import json
import math
import unittest

from api.utils.substrate_expansion import (
    SubstrateExpansionPlan,
    reduce_substrate_predictions,
    split_substrate_list,
)


class SubstrateExpansionTests(unittest.TestCase):
    def test_split_trims_ignores_empty_fragments_and_preserves_duplicates(self):
        self.assertEqual(split_substrate_list(" A ;; B ; A; "), ["A", "B", "A"])
        self.assertEqual(split_substrate_list(""), [])
        self.assertEqual(split_substrate_list(float("nan")), [])

    def test_plan_uses_positions_and_contiguous_slices(self):
        values = ["A;A", "B", "", "C;D;E"]
        plan = SubstrateExpansionPlan.build(values, [3, 0, 2])

        self.assertEqual(plan.reaction_positions, (3, 0, 2))
        self.assertEqual(plan.reaction_slices, ((3, 0, 3), (0, 3, 5), (2, 5, 5)))
        self.assertEqual(plan.expanded_substrates(), ["C", "D", "E", "A", "A"])
        self.assertEqual(
            plan.expanded_sequences(["seq-0", "seq-1", "seq-2", "seq-3"]),
            ["seq-3", "seq-3", "seq-3", "seq-0", "seq-0"],
        )

    def test_kcat_uses_successful_max_preserves_failures_and_first_tie(self):
        plan = SubstrateExpansionPlan.build(["A;B;C", "D", ""], [0, 1, 2])
        reduced = reduce_substrate_predictions(
            plan=plan,
            target="kcat",
            child_predictions=[5, None, "5.0", math.inf],
            child_sources=["model", "model", "experimental", "model"],
            child_errors={1: "Invalid substrate"},
            child_details=["", "", "reported value", ""],
            reaction_count=3,
        )

        self.assertEqual(reduced.predictions, [5.0, "", ""])
        self.assertEqual(reduced.sources[0], "model")
        self.assertEqual(set(reduced.failed_reactions), {1, 2})
        details = json.loads(reduced.extra_info[0])
        self.assertTrue(details[0]["selected"])
        self.assertFalse(details[2]["selected"])
        self.assertEqual(details[1]["error"], "Invalid substrate")
        self.assertEqual(details[2]["details"], "reported value")

    def test_non_kcat_outputs_ordered_json_with_nulls(self):
        plan = SubstrateExpansionPlan.build(["A", "B;C;D"], [0, 1])
        reduced = reduce_substrate_predictions(
            plan=plan,
            target="Km",
            child_predictions=["1.25", 2, None, "not-a-number"],
            child_sources=["model", "model", "", ""],
            child_errors={2: "Prediction failed"},
            child_details=None,
            reaction_count=2,
        )

        self.assertEqual(reduced.predictions, ["[1.25]", "[2.0,null,null]"])
        self.assertEqual(reduced.sources[0], "model (per substrate)")
        self.assertNotIn(1, reduced.failed_reactions)
        self.assertEqual(json.loads(reduced.extra_info[1])[2]["error"], "Prediction could not be made")

    def test_all_failed_array_result_is_blank(self):
        plan = SubstrateExpansionPlan.build(["A;B"], [0])
        reduced = reduce_substrate_predictions(
            plan=plan,
            target="kcat/Km",
            child_predictions=[None, float("nan")],
            child_sources=["", ""],
            child_errors={},
            child_details=None,
            reaction_count=1,
        )
        self.assertEqual(reduced.predictions, [""])
        self.assertEqual(set(reduced.failed_reactions), {0})

    def test_cardinality_mismatch_fails_loudly(self):
        plan = SubstrateExpansionPlan.build(["A;B"], [0])
        with self.assertRaisesRegex(ValueError, "1 prediction.*2 substrate"):
            reduce_substrate_predictions(
                plan=plan,
                target="kcat",
                child_predictions=[1],
                child_sources=["model"],
                child_errors={},
                child_details=None,
                reaction_count=1,
            )


if __name__ == "__main__":
    unittest.main()
