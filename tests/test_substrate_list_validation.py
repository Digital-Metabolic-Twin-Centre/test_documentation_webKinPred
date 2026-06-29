import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-secret-key")

try:
    import django
    import pandas as pd

    django.setup()
    from api.utils.job_utils import get_experimental_results, validate_required_columns_for_methods
    from api.utils.validation_utils import (
        validate_csv_structure,
        validate_column_emptiness,
        validate_protein_sequences,
        validate_products_column,
        validate_substrate_list_schema,
    )
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # Minimal developer shells may lack server dependencies.
    pd = None
    _IMPORT_ERROR = exc


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"Server test dependencies unavailable: {_IMPORT_ERROR}",
)
class SubstrateListValidationTests(unittest.TestCase):
    def test_structure_accepts_substrates_without_products(self):
        df = pd.DataFrame({"Protein Sequence": ["MKT"], "Substrates": ["CCO;O"]})
        self.assertIsNone(validate_csv_structure(df))

    def test_structure_rejects_ambiguous_or_orphan_columns(self):
        ambiguous = pd.DataFrame(
            {"Protein Sequence": ["MKT"], "Substrate": ["CCO"], "Substrates": ["O"]}
        )
        orphan_products = pd.DataFrame({"Protein Sequence": ["MKT"], "Products": ["O"]})
        self.assertIn("both", validate_csv_structure(ambiguous))
        self.assertIn("without", validate_csv_structure(orphan_products))

    def test_list_validation_reports_component_positions(self):
        df = pd.DataFrame(
            {
                "Protein Sequence": ["MKT"],
                "Substrates": ["CCO;not-a-molecule;O"],
            }
        )
        invalid = validate_substrate_list_schema(df)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["invalid_substrates"][0]["position"], 2)

    def test_products_are_validated_even_when_substrate_list_is_empty(self):
        df = pd.DataFrame(
            {
                "Protein Sequence": ["MKT"],
                "Substrates": [""],
                "Products": ["O;not-a-molecule"],
            }
        )
        invalid = validate_substrate_list_schema(df)
        self.assertEqual(invalid[0]["invalid_substrates"][0]["position"], None)
        self.assertEqual(invalid[0]["invalid_products"][0]["position"], 2)

    def test_product_preflight_reports_row_and_position(self):
        df = pd.DataFrame(
            {
                "Products": ["O;CC=O", "O;not-a-molecule"],
            }
        )
        invalid = validate_products_column(df)
        self.assertEqual(invalid[0]["row"], 2)
        self.assertEqual(invalid[0]["position"], 2)

    def test_pair_method_accepts_list_but_turnup_still_requires_products(self):
        df = pd.DataFrame({"Protein Sequence": ["MKT"], "Substrates": ["CCO;O"]})
        self.assertIsNone(
            validate_required_columns_for_methods(df, ["kcat"], {"kcat": "DLKcat"})
        )
        error = validate_required_columns_for_methods(
            df,
            ["kcat"],
            {"kcat": "TurNup"},
        )
        self.assertIn("Products", error)

    def test_catpred_uses_target_specific_list_behavior(self):
        list_df = pd.DataFrame(
            {"Protein Sequence": ["MKT"], "Substrates": ["CCO;O"]}
        )
        self.assertIsNone(
            validate_required_columns_for_methods(
                list_df,
                ["kcat", "Km"],
                {"kcat": "CatPred", "Km": "CatPred"},
            )
        )

        single_df = pd.DataFrame(
            {"Protein Sequence": ["MKT"], "Substrate": ["CCO.O"]}
        )
        self.assertIn(
            "Substrates",
            validate_required_columns_for_methods(
                single_df,
                ["kcat"],
                {"kcat": "CatPred"},
            ),
        )

    def test_catpred_kcat_rejects_scalar_single_substrate_schema(self):
        df = pd.DataFrame(
            {"Protein Sequence": ["MKT"], "Substrate": ["CCO"]}
        )
        error = validate_required_columns_for_methods(
            df,
            ["kcat"],
            {"kcat": "CatPred"},
        )
        self.assertIn("Substrates", error)

    @patch("api.utils.job_utils.get_experimental.lookup_experimental")
    def test_catpred_experimental_lookup_runs_only_for_expanded_km(self, lookup):
        lookup.return_value = [
            {"found": False},
            {"found": False},
        ]
        df = pd.DataFrame(
            {"Protein Sequence": ["MKT"], "Substrates": ["CCO;O"]}
        )
        result = get_experimental_results(
            True,
            {"kcat": "CatPred", "Km": "CatPred"},
            ["kcat", "Km"],
            df,
        )
        lookup.assert_called_once_with(["MKT", "MKT"], ["CCO", "O"], param_type="Km")
        self.assertIn("Km", result)
        self.assertNotIn("kcat", result)

    def test_full_reaction_is_accepted_by_pair_methods(self):
        df = pd.DataFrame(
            {
                "Protein Sequence": ["MKT"],
                "Substrates": ["CCO;O"],
                "Products": ["CC=O"],
            }
        )
        self.assertIsNone(
            validate_required_columns_for_methods(df, ["Km"], {"Km": "UniKP"})
        )
        self.assertIsNone(
            validate_required_columns_for_methods(
                df,
                ["kcat", "Km"],
                {"kcat": "TurNup", "Km": "UniKP"},
            )
        )

    def test_full_reaction_is_accepted_by_catpred_native_multi(self):
        df = pd.DataFrame(
            {
                "Protein Sequence": ["MKT"],
                "Substrates": ["CCO;O"],
                "Products": ["CC=O"],
            }
        )
        self.assertIsNone(
            validate_required_columns_for_methods(df, ["kcat"], {"kcat": "CatPred"})
        )

    def test_protein_sequence_lists_validate_each_position(self):
        df = pd.DataFrame({"Protein Sequence": ["AAA;AZC;CCC"]})
        invalid, lengths = validate_protein_sequences(df)
        self.assertEqual(invalid[0]["row"], 1)
        self.assertEqual(invalid[0]["position"], 2)
        self.assertEqual(invalid[0]["invalid_chars"], ["Z"])
        self.assertEqual(lengths["Server"], 0)

    def test_semicolon_only_protein_cells_are_empty(self):
        df = pd.DataFrame({"Protein Sequence": [";;", "AAA"], "Substrate": ["C", "O"]})
        self.assertIn("Rows", validate_column_emptiness(df, "Protein Sequence"))


if __name__ == "__main__":
    unittest.main()
