import unittest

from api.methods.catpred import descriptor as catpred
from api.methods.dlkcat import descriptor as dlkcat
from api.methods.turnup import descriptor as turnup


class MethodInputBehaviorTests(unittest.TestCase):
    def test_catpred_behaviors_are_target_specific(self):
        self.assertEqual(catpred.input_behavior("kcat"), "native_multi")
        self.assertEqual(catpred.input_behavior("Km"), "expanded_pair")
        self.assertEqual(
            catpred.accepted_csv_types_for_target("kcat"),
            ["multi", "full_reaction"],
        )
        self.assertEqual(
            catpred.accepted_csv_types_for_target("Km"),
            ["single", "multi", "full_reaction"],
        )

    def test_defaults_cover_pair_and_full_reaction_methods(self):
        self.assertEqual(dlkcat.input_behavior("kcat"), "expanded_pair")
        self.assertEqual(
            dlkcat.accepted_csv_types_for_target("kcat"),
            ["single", "multi", "full_reaction"],
        )
        self.assertEqual(turnup.input_behavior("kcat"), "native_full_reaction")
        self.assertEqual(turnup.accepted_csv_types_for_target("kcat"), ["full_reaction"])


if __name__ == "__main__":
    unittest.main()
