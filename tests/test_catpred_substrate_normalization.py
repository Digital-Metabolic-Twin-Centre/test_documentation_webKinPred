import unittest

try:
    from models.CatPred.catpred.integration.substrate_normalization import (
        normalize_catpred_substrates,
        substrate_components,
    )
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"CatPred normalization dependencies unavailable: {_IMPORT_ERROR}")
class CatPredSubstrateNormalizationTests(unittest.TestCase):
    def test_semicolon_input_becomes_catpred_dot_joined_smiles(self):
        self.assertEqual(normalize_catpred_substrates("CCO;O"), "CCO.O")
        self.assertEqual(substrate_components("CCO.O"), ["CCO.O"])

    def test_collection_preserves_order_duplicates_and_ignores_empty_fragments(self):
        self.assertEqual(substrate_components([" CCO ", "", "O", "CCO"]), ["CCO", "O", "CCO"])
        self.assertEqual(normalize_catpred_substrates(["CCO", "O", "CCO"]), "CCO.O.CCO")

    def test_one_substrate_and_noncanonical_smiles(self):
        self.assertEqual(normalize_catpred_substrates(["CCO"]), "CCO")
        self.assertEqual(
            normalize_catpred_substrates(["C(C)O", "O"], canonicalize=False),
            "C(C)O.O",
        )

    def test_inchi_is_converted_to_smiles(self):
        methane = "InChI=1S/CH4/h1H4"
        self.assertEqual(normalize_catpred_substrates([methane, "O"]), "C.O")

    def test_invalid_and_hydrogen_only_components_fail(self):
        with self.assertRaisesRegex(ValueError, "Invalid substrate component"):
            normalize_catpred_substrates("CCO;not-a-molecule")
        with self.assertRaisesRegex(ValueError, "contains no heavy atoms"):
            normalize_catpred_substrates("CCO;[H+]")
        with self.assertRaisesRegex(ValueError, "Missing substrate"):
            normalize_catpred_substrates(" ; ")


if __name__ == "__main__":
    unittest.main()
