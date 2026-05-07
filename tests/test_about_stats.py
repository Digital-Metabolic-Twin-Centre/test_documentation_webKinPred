from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class _FakeOutputFile:
    def __init__(self, path: str | None):
        self.path = path


class _FakeJob:
    def __init__(self, path: str | None):
        self.output_file = _FakeOutputFile(path) if path else None


class _FakeCompletedJobs:
    def __init__(self, jobs: list[_FakeJob], total_count: int):
        self._jobs = jobs
        self._total_count = total_count

    def only(self, *_args, **_kwargs):
        return self

    def iterator(self, chunk_size: int = 200):
        _ = chunk_size
        return iter(self._jobs)

    def count(self) -> int:
        return self._total_count


class AboutStatsServiceTests(unittest.TestCase):
    def _import_service_or_skip(self):
        try:
            from api.services import about_stats_service
        except ModuleNotFoundError as exc:
            if exc.name == "django":
                self.skipTest("Django is not installed in this Python environment.")
            raise
        return about_stats_service

    def _reset_cache(self, service_module):
        service_module._cache_payload = None
        service_module._cache_ts = 0.0

    def test_compute_stats_counts_mixed_outputs_and_unique_sequences(self):
        service = self._import_service_or_skip()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path_a = os.path.join(tmp_dir, "job_a.csv")
            path_b = os.path.join(tmp_dir, "job_b.csv")
            path_bad = os.path.join(tmp_dir, "bad.csv")
            path_missing = os.path.join(tmp_dir, "missing.csv")

            with open(path_a, "w", encoding="utf-8", newline="") as handle:
                handle.write("Protein Sequence,kcat (1/s),KM (mM)\n")
                handle.write("SEQ_A,1.1,\n")
                handle.write("SEQ_B,,0.2\n")
                handle.write("SEQ_A,,\n")
                handle.write(",3.2,\n")

            with open(path_b, "w", encoding="utf-8", newline="") as handle:
                handle.write("Protein Sequence,kcat/Km (1/(s*mM))\n")
                handle.write("SEQ_C,0.7\n")
                handle.write("SEQ_B,\n")
                handle.write(",1.3\n")

            with open(path_bad, "wb") as handle:
                handle.write(b"\xff\xfe\xff")

            fake_jobs = [
                _FakeJob(path_a),
                _FakeJob(path_b),
                _FakeJob(path_bad),
                _FakeJob(path_missing),
                _FakeJob(None),
            ]
            fake_qs = _FakeCompletedJobs(fake_jobs, total_count=7)

            with patch.object(service.Job.objects, "filter", return_value=fake_qs):
                payload = service._compute_about_stats_payload("2026-05-07T12:00:00+00:00")

        self.assertEqual(payload["scope"], "all_time")
        self.assertEqual(payload["generated_at"], "2026-05-07T12:00:00+00:00")
        self.assertEqual(payload["jobs_completed"], 7)
        self.assertEqual(payload["reactions_completed"], 5)
        self.assertEqual(payload["unique_protein_sequences"], 3)
        self.assertEqual(payload["kcat_predictions_completed"], 2)
        self.assertEqual(payload["km_predictions_completed"], 1)
        self.assertEqual(payload["kcat_km_predictions_completed"], 2)

    def test_get_about_stats_uses_ttl_cache_and_supports_force_refresh(self):
        service = self._import_service_or_skip()
        self._reset_cache(service)

        payload_a = {
            "scope": "all_time",
            "generated_at": "first",
            "jobs_completed": 1,
            "reactions_completed": 2,
            "unique_protein_sequences": 3,
            "kcat_predictions_completed": 4,
            "km_predictions_completed": 5,
            "kcat_km_predictions_completed": 6,
        }
        payload_b = {
            "scope": "all_time",
            "generated_at": "second",
            "jobs_completed": 10,
            "reactions_completed": 20,
            "unique_protein_sequences": 30,
            "kcat_predictions_completed": 40,
            "km_predictions_completed": 50,
            "kcat_km_predictions_completed": 60,
        }

        with patch.object(
            service,
            "_compute_about_stats_payload",
            side_effect=[payload_a, payload_b],
        ) as mocked_compute:
            with patch.object(service.time, "monotonic", side_effect=[100.0, 100.1, 101.0]):
                first = service.get_about_stats()
                second = service.get_about_stats()
                third = service.get_about_stats(force_refresh=True)

        self.assertEqual(first["generated_at"], "first")
        self.assertEqual(second["generated_at"], "first")
        self.assertEqual(third["generated_at"], "second")
        self.assertEqual(mocked_compute.call_count, 2)


class AboutStatsEndpointTests(unittest.TestCase):
    def _import_endpoint_or_skip(self):
        try:
            from django.test import RequestFactory
            from api.views import stats_views
        except ModuleNotFoundError as exc:
            if exc.name == "django":
                self.skipTest("Django is not installed in this Python environment.")
            raise
        return RequestFactory, stats_views

    def test_about_stats_endpoint_returns_expected_keys(self):
        RequestFactory, stats_views = self._import_endpoint_or_skip()

        payload = {
            "scope": "all_time",
            "generated_at": "2026-05-07T12:00:00+00:00",
            "jobs_completed": 1,
            "reactions_completed": 2,
            "unique_protein_sequences": 3,
            "kcat_predictions_completed": 4,
            "km_predictions_completed": 5,
            "kcat_km_predictions_completed": 6,
        }

        with patch.object(stats_views, "get_about_stats", return_value=payload):
            request = RequestFactory().get("/api/about-stats/")
            response = stats_views.about_stats(request)

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["scope"], "all_time")
        self.assertIn("generated_at", data)
        self.assertIsInstance(data["jobs_completed"], int)
        self.assertIsInstance(data["reactions_completed"], int)
        self.assertIsInstance(data["unique_protein_sequences"], int)
        self.assertIsInstance(data["kcat_predictions_completed"], int)
        self.assertIsInstance(data["km_predictions_completed"], int)
        self.assertIsInstance(data["kcat_km_predictions_completed"], int)


if __name__ == "__main__":
    unittest.main()
