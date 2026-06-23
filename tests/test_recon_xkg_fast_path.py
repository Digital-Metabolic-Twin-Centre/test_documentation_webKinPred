"""Focused coverage for immediate ReconXKG full-cache completion."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-secret-key")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import django
    import pandas as pd

    django.setup()
    from api.services.job_service import (
        _try_complete_recon_xkg_job,
        process_job_submission_from_params,
    )
    from api.services.prediction_batch_service import (
        build_sequence_batch_plan,
        build_target_batch_plan,
    )
    from api.services.recon_xkg_preflight_service import (
        ReconXkgCacheSnapshot,
        ReconXkgPreflightResult,
        preflight_recon_xkg_cache,
    )
    from api.services.result_service import serialize_result_csv
    from api.services.similarity_service import append_kcat_similarity_columns_to_output_csv

    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


class _Descriptor:
    key = "PairMethod"
    display_name = "Pair method"
    max_seq_len = 4
    model_version = "1"
    col_to_kwarg = {"Substrate": "substrates"}
    target_kwargs = {"kcat": {"mode": "kcat"}, "Km": {"mode": "km"}}

    def __init__(self, behavior="expanded_pair", col_to_kwarg=None, key=None):
        self.behavior = behavior
        if col_to_kwarg is not None:
            self.col_to_kwarg = col_to_kwarg
        if key is not None:
            self.key = key

    def input_behavior(self, _target):
        return self.behavior


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class BatchPlanningTests(unittest.TestCase):
    def test_expanded_plan_preserves_positions_duplicates_and_truncation(self):
        dataframe = pd.DataFrame(
            {
                "Protein Sequence": ["ABCDE", "XYZ"],
                "Substrates": [" CCO ; O ; CCO ", "N"],
            },
            index=[10, 3],
        )
        descriptor = _Descriptor()
        sequences = build_sequence_batch_plan(dataframe, [descriptor], "truncate")
        batch = build_target_batch_plan(descriptor, "kcat", dataframe, sequences)

        self.assertEqual(sequences.valid_reaction_indices, (0, 1))
        self.assertEqual(batch.sequences, ("ABDE", "ABDE", "ABDE", "XYZ"))
        self.assertEqual(batch.call_kwargs["substrates"], ["CCO", "O", "CCO", "N"])
        self.assertEqual(batch.call_kwargs["mode"], "kcat")

    def test_native_multi_and_full_reaction_batches(self):
        dataframe = pd.DataFrame(
            {
                "Protein Sequence": ["AAA", "BBB"],
                "Substrates": ["CCO;O", "N"],
                "Products": ["CC=O;O", "N"],
            }
        )
        native_multi = _Descriptor("native_multi", key="CatPred")
        sequence_plan = build_sequence_batch_plan(dataframe, [native_multi], "skip")
        catpred = build_target_batch_plan(
            native_multi, "kcat", dataframe, sequence_plan
        )
        self.assertEqual(catpred.sequences, ("AAA", "BBB"))
        self.assertEqual(catpred.call_kwargs["substrates"], [["CCO", "O"], ["N"]])

        turnup = _Descriptor(
            "native_full_reaction",
            col_to_kwarg={"Substrates": "substrates", "Products": "products"},
            key="TurNup",
        )
        full = build_target_batch_plan(turnup, "kcat", dataframe, sequence_plan)
        self.assertEqual(full.call_kwargs["substrates"], ["CCO;O", "N"])
        self.assertEqual(full.call_kwargs["products"], ["CC=O;O", "N"])

    def test_skipped_sequences_create_no_target_units(self):
        dataframe = pd.DataFrame(
            {"Protein Sequence": ["TOOLONG", "AAA"], "Substrate": ["C", "O"]}
        )
        descriptor = _Descriptor()
        sequence_plan = build_sequence_batch_plan(dataframe, [descriptor], "skip")
        batch = build_target_batch_plan(descriptor, "Km", dataframe, sequence_plan)
        self.assertEqual(sequence_plan.valid_reaction_indices, (1,))
        self.assertEqual(batch.sequences, ("AAA",))
        self.assertEqual(batch.call_kwargs["substrates"], ["O"])


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
            {"Protein Sequence": ["AAA", "AAA"], "Substrates": ["C;O", "C"]}
        )
        self.descriptor = _Descriptor()

    def test_deduplicates_lookup_keys_and_captures_similarity_snapshot(self):
        key_batches = iter([(["a", "b", "a"], [None] * 3, "fp")])
        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            side_effect=lambda *_a, **_k: next(key_batches),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            return_value={"a": 1.0, "b": 2.0},
        ) as get_many, patch(
            "api.services.recon_xkg_preflight_service.similarity_cache_label_for_method",
            return_value="Pair data",
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_similarity_many",
            return_value={"AAA": (4.0, 8.0)},
        ):
            result = preflight_recon_xkg_cache(
                dataframe=self.dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=True,
                job_public_id="job-1",
            )

        self.assertTrue(result.complete)
        self.assertEqual(set(get_many.call_args.args[0]), {"a", "b"})
        self.assertEqual(result.snapshot.predictions, {"a": 1.0, "b": 2.0})
        self.assertEqual(result.snapshot.similarities, {"AAA": (4.0, 8.0)})

    def test_one_missing_or_uncacheable_unit_forces_queue_path(self):
        for keys, cached, expected_reason in [
            (["a", "b", "a"], {"a": 1.0}, "prediction-cache-miss"),
            (["a", "b", "a"], {"a": 1.0, "b": float("nan")}, "prediction-cache-miss"),
            (["a", None, "a"], {}, "uncacheable-prediction-unit"),
        ]:
            with self.subTest(reason=expected_reason), patch(
                "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
                return_value=(keys, [None] * len(keys), "fp"),
            ), patch(
                "api.services.recon_xkg_preflight_service.prediction_store.get_many",
                return_value=cached,
            ):
                result = preflight_recon_xkg_cache(
                    dataframe=self.dataframe,
                    targets=["kcat"],
                    descriptors={"kcat": self.descriptor},
                    handle_long_sequences="truncate",
                    canonicalize_substrates=True,
                    include_similarity_columns=False,
                    job_public_id="job-1",
                )
            self.assertFalse(result.complete)
            self.assertEqual(result.reason, expected_reason)

    def test_negative_prediction_outcome_satisfies_preflight(self):
        from api.services.prediction_store import CachedFailure

        reason = "Invalid protein sequence (unsupported amino acid characters)"
        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            return_value=(["a", "b", "a"], [None] * 3, "fp"),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            return_value={"a": 1.0, "b": CachedFailure(reason)},
        ):
            result = preflight_recon_xkg_cache(
                dataframe=self.dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=False,
                job_public_id="job-1",
            )
        self.assertTrue(result.complete)
        self.assertEqual(result.snapshot.predictions["b"], CachedFailure(reason))

    def test_missing_similarity_or_cache_exception_forces_queue_path(self):
        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            return_value=(["a", "b", "a"], [None] * 3, "fp"),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            return_value={"a": 1.0, "b": 2.0},
        ), patch(
            "api.services.recon_xkg_preflight_service.similarity_cache_label_for_method",
            return_value="Pair data",
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_similarity_many",
            return_value={},
        ):
            missing_similarity = preflight_recon_xkg_cache(
                dataframe=self.dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=True,
                job_public_id="job-1",
            )
        self.assertFalse(missing_similarity.complete)
        self.assertEqual(missing_similarity.reason, "similarity-cache-miss")

        with patch(
            "api.services.recon_xkg_preflight_service.prediction_store.build_unit_keys",
            return_value=(["a", "b", "a"], [None] * 3, "fp"),
        ), patch(
            "api.services.recon_xkg_preflight_service.prediction_store.get_many",
            side_effect=RuntimeError("store unavailable"),
        ):
            unreadable = preflight_recon_xkg_cache(
                dataframe=self.dataframe,
                targets=["kcat"],
                descriptors={"kcat": self.descriptor},
                handle_long_sequences="truncate",
                canonicalize_substrates=True,
                include_similarity_columns=False,
                job_public_id="job-1",
            )
        self.assertFalse(unreadable.complete)
        self.assertEqual(unreadable.reason, "cache-read-error")


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class ImmediateCompletionTests(unittest.TestCase):
    def setUp(self):
        self.job = SimpleNamespace(public_id="job-1")
        self.params = {
            "targets": ["kcat"],
            "methods": {"kcat": "DLKcat"},
            "handle_long_sequences": "truncate",
            "canonicalize_substrates": True,
            "include_similarity_columns": False,
            "disable_gpu_precompute": False,
        }
        self.snapshot = ReconXkgCacheSnapshot({"key": 1.0}, {})
        self.hit = ReconXkgPreflightResult(True, self.snapshot, "full-cache-hit", 1, 1, 0)

    def test_hit_runs_strict_executor_without_queueing(self):
        with patch("api.services.job_service.get_method", return_value=_Descriptor()), patch(
            "api.services.job_service.preflight_recon_xkg_cache", return_value=self.hit
        ), patch("api.services.job_service.execute_multi_prediction_job") as execute:
            completed = _try_complete_recon_xkg_job(
                job=self.job,
                params=self.params,
                dataframe=MagicMock(),
                experimental_results={},
            )

        self.assertTrue(completed)
        self.assertTrue(execute.call_args.kwargs["cache_only"])
        self.assertEqual(execute.call_args.kwargs["prediction_cache_snapshot"], {"key": 1.0})

    def test_assembly_failure_resets_job_for_one_normal_dispatch(self):
        with patch("api.services.job_service.get_method", return_value=_Descriptor()), patch(
            "api.services.job_service.preflight_recon_xkg_cache", return_value=self.hit
        ), patch(
            "api.services.job_service.execute_multi_prediction_job",
            side_effect=RuntimeError("assembly failed"),
        ), patch("api.services.job_service._reset_job_after_cache_only_failure") as reset:
            completed = _try_complete_recon_xkg_job(
                job=self.job,
                params=self.params,
                dataframe=MagicMock(),
                experimental_results={},
            )

        self.assertFalse(completed)
        reset.assert_called_once_with(self.job)


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class SubmissionDispatchTests(unittest.TestCase):
    def _submit(self, recon_xkg: bool, immediate: bool):
        dataframe = pd.DataFrame(
            {"Protein Sequence": ["AAA"], "Substrate": ["C"]}
        )
        params = {
            "targets": ["kcat"],
            "methods": {"kcat": "DLKcat"},
            "handle_long_sequences": "truncate",
            "use_experimental": False,
            "include_similarity_columns": False,
            "canonicalize_substrates": True,
            "disable_gpu_precompute": False,
            "recon_xkg": recon_xkg,
        }
        job = SimpleNamespace(public_id="job-1")
        with patch(
            "api.services.job_service.validate_prediction_parameters", return_value=None
        ), patch(
            "api.services.job_service.validate_sequence_handling_option", return_value=None
        ), patch("api.services.job_service.parse_csv_file", return_value=dataframe), patch(
            "api.services.job_service.validate_required_columns_for_methods",
            return_value=None,
        ), patch(
            "api.services.job_service.validate_column_emptiness", return_value=None
        ), patch(
            "api.services.job_service.validate_products_column", return_value=[]
        ), patch(
            "api.services.job_service.handle_quota_validation", return_value=None
        ), patch(
            "api.services.job_service.get_experimental_results", return_value={}
        ), patch("api.services.job_service.create_job_record", return_value=job), patch(
            "api.services.job_service.mark_about_stats_cache_stale"
        ), patch("api.services.job_service.create_job_directory", return_value="/tmp/job-1"), patch(
            "api.services.job_service.save_job_input_file"
        ), patch(
            "api.services.job_service._try_complete_recon_xkg_job",
            return_value=immediate,
        ) as attempt, patch("api.services.job_service.dispatch_prediction_task") as dispatch:
            error, success = process_job_submission_from_params(
                params,
                MagicMock(),
                "127.0.0.1",
                user=MagicMock(),
            )
        return error, success, attempt, dispatch

    def test_full_hit_skips_celery_dispatch(self):
        error, success, attempt, dispatch = self._submit(True, True)
        self.assertIsNone(error)
        self.assertTrue(success["completed_immediately"])
        attempt.assert_called_once()
        dispatch.assert_not_called()

    def test_miss_dispatches_exactly_once(self):
        error, success, attempt, dispatch = self._submit(True, False)
        self.assertIsNone(error)
        self.assertFalse(success["completed_immediately"])
        attempt.assert_called_once()
        dispatch.assert_called_once()

    def test_non_recon_submission_does_not_preflight(self):
        error, success, attempt, dispatch = self._submit(False, False)
        self.assertIsNone(error)
        self.assertFalse(success["completed_immediately"])
        attempt.assert_not_called()
        dispatch.assert_called_once()


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server dependencies unavailable: {_IMPORT_ERROR}")
class ResultSerializationTests(unittest.TestCase):
    def test_blank_cells_are_json_null(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv") as output:
            output.write("prediction,source\n1.0,model\n,\n")
            output.flush()
            result = serialize_result_csv(output.name)

        self.assertEqual(result["rowCount"], 2)
        self.assertIsNone(result["data"][1]["prediction"])
        self.assertIsNone(result["data"][1]["source"])

    def test_similarity_snapshot_never_invokes_mmseqs(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv") as output:
            output.write("Protein Sequence,kcat (1/s)\nAAA,1.0\n")
            output.flush()
            with patch(
                "api.services.similarity_service._resolve_similarity_dataset_for_method",
                return_value=("Pair data", None),
            ), patch("api.services.similarity_service._compute_mmseqs_similarity") as mmseqs:
                append_kcat_similarity_columns_to_output_csv(
                    output.name,
                    "PairMethod",
                    recon_xkg=True,
                    cached_similarity_snapshot={"AAA": (12.0, 34.0)},
                    cache_only=True,
                )
            result = serialize_result_csv(output.name)

        mmseqs.assert_not_called()
        self.assertEqual(
            result["data"][0]["mean similarity to PairMethod training data"],
            12.0,
        )
        self.assertEqual(
            result["data"][0]["max similarity to PairMethod training data"],
            34.0,
        )


if __name__ == "__main__":
    unittest.main()
