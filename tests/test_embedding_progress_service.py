#!/usr/bin/env python3
"""Unit tests for embedding progress cache-path planning logic."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class _FakeRedis:
    def __init__(self):
        self._store = {}

    def set(self, key, value, ex=None):
        self._store[key] = value

    def get(self, key):
        return self._store.get(key)

    def delete(self, *keys):
        for key in keys:
            self._store.pop(key, None)


fake_progress_service = types.ModuleType("api.services.progress_service")
fake_progress_service.redis_conn = _FakeRedis()
sys.modules["api.services.progress_service"] = fake_progress_service

from api.services import embedding_progress_service as eps  # noqa: E402


class EmbeddingProgressPlanningTests(unittest.TestCase):
    def test_prepare_plan_unikp_partial_cache(self):
        with tempfile.TemporaryDirectory(prefix="emb_prog_unikp_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            vec_dir = media / "sequence_info" / "prot_t5_last" / "mean_vecs"
            vec_dir.mkdir(parents=True, exist_ok=True)
            (vec_dir / "sid_a.npy").write_bytes(b"cached")

            with patch.object(eps, "_resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "_resolve_seq_ids_via_cli", return_value=["sid_a", "sid_b"]):
                    plan = eps._prepare_plan(
                        method_key="UniKP",
                        target="kcat",
                        sequences=["SEQ_A", "SEQ_B"],
                        env={},
                    )

            self.assertEqual(plan.total, 2)
            self.assertEqual(plan.cached_already, 1)
            self.assertEqual(plan.need_computation, 1)
            self.assertIn("sid_b", plan.missing_paths_by_seq)
            self.assertNotIn("sid_a", plan.missing_paths_by_seq)

    def test_prepare_plan_turnup_sequence_normalisation_and_dedupe(self):
        with tempfile.TemporaryDirectory(prefix="emb_prog_turnup_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            vec_dir = media / "sequence_info" / "esm1b_turnup"
            vec_dir.mkdir(parents=True, exist_ok=True)

            captured_sequences: list[str] = []

            def _capture_ids(seqs, *_args, **_kwargs):
                captured_sequences.extend(seqs)
                return ["sid_1", "sid_2"]

            with patch.object(eps, "_resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "_resolve_seq_ids_via_cli", side_effect=_capture_ids):
                    plan = eps._prepare_plan(
                        method_key="TurNup",
                        target="kcat",
                        sequences=["acdef", "ACDEF", "ghiklm"],
                        env={},
                    )

            # TurNup normalises to uppercase and de-duplicates.
            self.assertEqual(captured_sequences, ["ACDEF", "GHIKLM"])
            self.assertEqual(plan.total, 2)
            self.assertEqual(plan.cached_already, 0)
            self.assertEqual(plan.need_computation, 2)

    def test_catpred_expected_paths_require_all_checkpoint_variants(self):
        with tempfile.TemporaryDirectory(prefix="emb_prog_catpred_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            checkpoint_root = tmp_path / "checkpoints"
            seq_ids = ["seq_123"]

            # Mimic CatPred checkpoint layout discovered by adapter logic.
            (checkpoint_root / "kcat" / "foldA" / "seed1").mkdir(parents=True, exist_ok=True)
            (checkpoint_root / "kcat" / "foldB" / "seed2").mkdir(parents=True, exist_ok=True)
            (checkpoint_root / "kcat" / "foldA" / "seed1" / "model.pt").write_bytes(b"x")
            (checkpoint_root / "kcat" / "foldB" / "seed2" / "model.pt").write_bytes(b"y")

            expected = eps._expected_paths_by_seq(
                method_key="CatPred",
                target="kcat",
                seq_ids=seq_ids,
                media_path=media,
                env={"CATPRED_CHECKPOINT_ROOT": str(checkpoint_root)},
            )

            self.assertIn("seq_123", expected)
            self.assertEqual(len(expected["seq_123"]), 2)
            for path in expected["seq_123"]:
                self.assertTrue(path.endswith("seq_123.pt"))
                self.assertIn("catpred_esm2/kcat/", path)

    def test_omniesi_expected_paths_use_shared_seq_id_cache(self):
        with tempfile.TemporaryDirectory(prefix="emb_prog_omniesi_") as tmp:
            media = Path(tmp) / "media"
            expected = eps._expected_paths_by_seq(
                method_key="OmniESI",
                target="Km",
                seq_ids=["sid_123"],
                media_path=media,
                env={},
            )

            self.assertEqual(
                expected,
                {
                    "sid_123": {
                        str((media / "sequence_info" / "omniesi_esm2" / "sid_123.pt").resolve())
                    }
                },
            )

    def test_start_tracking_skips_dlkcat(self):
        fake_progress_service.redis_conn.set("job_embedding_progress:job_x", '{"enabled": true}')
        started = eps.start_embedding_tracking(
            job_public_id="job_x",
            method_key="DLKcat",
            target="kcat",
            valid_sequences=["AAAA"],
            env={},
        )
        self.assertFalse(started)
        self.assertIsNone(eps.get_embedding_progress("job_x"))

    def test_tracker_increments_per_file(self):
        # need_computation and total count individual files, not sequences.
        # For sid_1 with 2 missing files, need_computation=2 and computed
        # increments once per file as each one appears.
        plan = eps._PreparedPlan(
            method_key="CatPred",
            target="kcat",
            total=2,
            cached_already=0,
            need_computation=2,
            missing_paths_by_seq={"sid_1": {"/tmp/a.pt", "/tmp/b.pt"}},
            path_to_seqs={"/tmp/a.pt": {"sid_1"}, "/tmp/b.pt": {"sid_1"}},
            watch_dirs=set(),
        )
        tracker = eps._EmbeddingTracker(job_public_id="job_seq_done", plan=plan)

        first = tracker._mark_path_present("/tmp/a.pt")
        self.assertTrue(first)
        self.assertEqual(tracker.computed, 1)
        self.assertEqual(tracker.remaining, 1)

        second = tracker._mark_path_present("/tmp/b.pt")
        self.assertTrue(second)
        self.assertEqual(tracker.computed, 2)
        self.assertEqual(tracker.remaining, 0)

    def test_tracker_redis_write_failure_is_best_effort(self):
        plan = eps._PreparedPlan(
            method_key="OmniESI",
            target="Km",
            total=1,
            cached_already=0,
            need_computation=0,
            missing_paths_by_seq={},
            path_to_seqs={},
            watch_dirs=set(),
        )
        tracker = eps._EmbeddingTracker(job_public_id="job_redis_down", plan=plan)

        with patch.object(eps.redis_conn, "set", side_effect=RuntimeError("redis down")):
            tracker.start()
            tracker.stop()


if __name__ == "__main__":
    unittest.main()
