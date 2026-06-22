import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-secret-key")

try:
    import django
    import pandas as pd

    django.setup()
    from api.methods.catpred import descriptor as catpred
    from api.tasks import _execute_target_batch
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server test dependencies unavailable: {_IMPORT_ERROR}")
class TargetInputDispatchTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Protein Sequence": ["MKT", "AAA"],
                "Substrates": ["CCO;O", "C"],
                "Products": ["CC=O", "C"],
            }
        )
        self.common = {
            "job": SimpleNamespace(public_id="job-1"),
            "desc": catpred,
            "df": self.df,
            "sequences": ["MKT", "AAA"],
            "processed_by_reaction": ["MKT", "AAA"],
            "valid_reaction_indices": [0, 1],
            "experimental_results": [],
            "canonicalize_substrates": True,
            "disable_gpu_precompute": True,
            "extra_call_kwargs": {},
        }

    def test_catpred_kcat_receives_one_ordered_collection_per_reaction(self):
        expected = {
            "preds": [1.0, 2.0],
            "sources": ["a", "b"],
            "extra": ["", ""],
            "failed_reactions": {},
            "output_col": "kcat (1/s)",
        }
        with patch("api.tasks._execute_native_target_batch", return_value=expected) as native:
            result = _execute_target_batch(target="kcat", **self.common)

        self.assertEqual(result, expected)
        kwargs = native.call_args.kwargs
        self.assertEqual(kwargs["call_kwargs_override"]["substrates"], [["CCO", "O"], ["C"]])
        self.assertNotIn("products", kwargs["call_kwargs_override"])
        self.assertFalse(kwargs["apply_experimental_overrides"])

    def test_catpred_km_expands_and_returns_ordered_arrays(self):
        with patch(
            "api.tasks._invoke_method_prediction",
            return_value=([1.0, 2.0, 3.0], {}),
        ) as invoke:
            result = _execute_target_batch(target="Km", **self.common)

        self.assertEqual(invoke.call_args.kwargs["substrates"], ["CCO", "O", "C"])
        self.assertNotIn("products", invoke.call_args.kwargs)
        self.assertEqual(result["preds"], ["[1.0,2.0]", "[3.0]"])


if __name__ == "__main__":
    unittest.main()
