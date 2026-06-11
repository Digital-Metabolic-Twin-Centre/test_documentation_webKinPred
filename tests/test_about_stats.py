from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class _FakeCacheRow:
    def __init__(self, payload: str = "", is_stale: bool = True):
        self.payload = payload
        self.is_stale = is_stale
        self.generated_at = None
        self.saved = []

    def save(self, update_fields=None):
        self.saved.append(tuple(update_fields or ()))


class AboutStatsServiceTests(unittest.TestCase):
    def _import_service_or_skip(self):
        try:
            from api.services import about_stats_service
        except ModuleNotFoundError as exc:
            if exc.name == "django":
                self.skipTest("Django is not installed in this Python environment.")
            raise
        return about_stats_service

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
        self.assertEqual(payload["parameter_predictions_completed"], 5)
        self.assertEqual(payload["kcat_predictions_completed"], 2)
        self.assertEqual(payload["km_predictions_completed"], 1)
        self.assertEqual(payload["kcat_km_predictions_completed"], 2)

    def test_get_about_stats_returns_fresh_cached_payload_without_recompute(self):
        service = self._import_service_or_skip()
        cached = {
            "scope": "all_time",
            "generated_at": "cached",
            "jobs_completed": 9,
            "reactions_completed": 8,
            "unique_protein_sequences": 7,
            "parameter_predictions_completed": 15,
            "kcat_predictions_completed": 6,
            "km_predictions_completed": 5,
            "kcat_km_predictions_completed": 4,
        }
        row = _FakeCacheRow(payload=json.dumps(cached), is_stale=False)

        with patch.object(service, "_get_or_create_cache_row", return_value=row):
            with patch.object(service, "refresh_about_stats_cache") as mocked_refresh:
                payload = service.get_about_stats()

        self.assertEqual(payload, cached)
        mocked_refresh.assert_not_called()

    def test_get_about_stats_recomputes_when_stale(self):
        service = self._import_service_or_skip()
        row = _FakeCacheRow(payload="", is_stale=True)
        fresh = {
            "scope": "all_time",
            "generated_at": "fresh",
            "jobs_completed": 1,
            "reactions_completed": 2,
            "unique_protein_sequences": 3,
            "parameter_predictions_completed": 15,
            "kcat_predictions_completed": 4,
            "km_predictions_completed": 5,
            "kcat_km_predictions_completed": 6,
        }

        with patch.object(service, "_get_or_create_cache_row", return_value=row):
            with patch.object(service, "refresh_about_stats_cache", return_value=fresh) as mocked_refresh:
                payload = service.get_about_stats()

        self.assertEqual(payload, fresh)
        mocked_refresh.assert_called_once_with(force=True)

    def test_mark_about_stats_cache_stale_sets_flag(self):
        service = self._import_service_or_skip()
        row = _FakeCacheRow(payload="{}", is_stale=False)

        with patch.object(service, "_get_or_create_cache_row", return_value=row):
            service.mark_about_stats_cache_stale()

        self.assertTrue(row.is_stale)
        self.assertTrue(any("is_stale" in fields for fields in row.saved))


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
            "parameter_predictions_completed": 15,
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
        self.assertIsInstance(data["parameter_predictions_completed"], int)
        self.assertIsInstance(data["kcat_predictions_completed"], int)
        self.assertIsInstance(data["km_predictions_completed"], int)
        self.assertIsInstance(data["kcat_km_predictions_completed"], int)


class JobFlowInvalidationTests(unittest.TestCase):
    def _import_job_service_or_skip(self):
        try:
            from api.services import job_service
        except ModuleNotFoundError as exc:
            if exc.name == "django":
                self.skipTest("Django is not installed in this Python environment.")
            raise
        return job_service

    def test_submission_marks_about_stats_cache_stale(self):
        job_service = self._import_job_service_or_skip()

        fake_df = MagicMock()
        fake_df.columns = ["Protein Sequence", "Substrate"]
        fake_df.__len__.return_value = 1

        params = {
            "targets": ["kcat"],
            "methods": {"kcat": "DLKcat"},
            "handle_long_sequences": "truncate",
            "use_experimental": False,
            "include_similarity_columns": True,
            "canonicalize_substrates": True,
        }

        with patch.object(job_service, "validate_prediction_parameters", return_value=None), \
            patch.object(job_service, "validate_sequence_handling_option", return_value=None), \
            patch.object(job_service, "parse_csv_file", return_value=fake_df), \
            patch.object(job_service, "validate_required_columns_for_methods", return_value=None), \
            patch.object(job_service, "validate_column_emptiness", return_value=None), \
            patch.object(job_service, "handle_quota_validation", return_value=None), \
            patch.object(job_service, "get_or_create_user", return_value=None), \
            patch.object(job_service, "get_experimental_results", return_value=None), \
            patch.object(job_service, "create_job_record") as mocked_create_job, \
            patch.object(job_service, "create_job_directory", return_value="/tmp/fake"), \
            patch.object(job_service, "save_job_input_file", return_value="/tmp/fake/input.csv"), \
            patch.object(job_service, "dispatch_prediction_task", return_value=None), \
            patch.object(job_service, "mark_about_stats_cache_stale") as mocked_mark_stale:
            mocked_create_job.return_value = MagicMock(public_id="abc123")
            error, success = job_service.process_job_submission_from_params(params, MagicMock(), "127.0.0.1")

        self.assertIsNone(error)
        self.assertEqual(success["public_id"], "abc123")
        mocked_mark_stale.assert_called_once()


if __name__ == "__main__":
    unittest.main()
