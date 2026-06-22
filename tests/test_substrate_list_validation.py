import unittest

try:
    import pandas as pd

    from api.utils.job_utils import validate_required_columns_for_methods
    from api.utils.validation_utils import (
        validate_csv_structure,
        validate_substrate_list_schema,
    )
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # Minimal developer shells may lack server dependencies.
    pd = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Server test dependencies unavailable: {_IMPORT_ERROR}")
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


if __name__ == "__main__":
    unittest.main()
