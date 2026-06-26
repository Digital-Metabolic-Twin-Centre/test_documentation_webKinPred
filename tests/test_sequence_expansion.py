import json
import math
import unittest

from api.utils.sequence_expansion import (
    SequenceExpansionPlan,
    TargetExpansionPlan,
    TargetPredictionUnit,
    count_multi_sequence_rows,
    reduce_sequence_predictions,
    split_sequence_list,
)


class SequenceExpansionTests(unittest.TestCase):
    def test_split_trims_ignores_empty_fragments_and_preserves_duplicates(self):
        self.assertEqual(split_sequence_list(" AAA ;; BBB ; AAA; "), ["AAA", "BBB", "AAA"])
        self.assertEqual(split_sequence_list(""), [])
        self.assertEqual(split_sequence_list(float("nan")), [])
        self.assertEqual(count_multi_sequence_rows(["AAA;BBB", "CCC", ";;"]), 1)

    def test_sequence_plan_handles_truncate_skip_and_empty_rows(self):
        truncate = SequenceExpansionPlan.build(
            ["ABCDE;XYZ", ";;", "ABCDEF"],
            range(3),
            limit=4,
            handle_long_sequences="truncate",
        )
        self.assertTrue(truncate.requires_reduction)
        self.assertEqual(
            [child.processed_sequence for child in truncate.children],
            ["ABDE", "XYZ", "ABEF"],
        )
        self.assertEqual(truncate.skipped_reactions, {1: "Missing protein sequence"})

        skip = SequenceExpansionPlan.build(
            ["ABCDE;XYZ"],
            [0],
            limit=4,
            handle_long_sequences="skip",
        )
        self.assertEqual(skip.children[0].processed_sequence, None)
        self.assertEqual(
            skip.children[0].skip_reason,
            "Sequence too long — sequence candidate was excluded",
        )
        self.assertEqual(skip.children[1].processed_sequence, "XYZ")
        self.assertEqual(skip.skipped_reactions, {})

    def test_scalar_targets_reduce_by_target_direction_and_first_tie(self):
        seq_plan = SequenceExpansionPlan.build(
            ["SEQ1;SEQ2;SEQ3"],
            [0],
            limit=99,
            handle_long_sequences="truncate",
        )
        plan = TargetExpansionPlan(
            sequence_plan=seq_plan,
            units=tuple(
                TargetPredictionUnit(0, i, i, seq_plan.children[i].sequence)
                for i in range(3)
            ),
            reaction_slices=((0, 0, 3),),
            substrate_tokens_by_reaction=(tuple(),),
        )

        kcat = reduce_sequence_predictions(
            plan=plan,
            target="kcat",
            child_predictions=[5, None, "5.0"],
            child_sources=["model", "", "experimental"],
            child_errors={1: "Invalid protein sequence"},
            child_details=None,
            reaction_count=1,
        )
        self.assertEqual(kcat.predictions, [5.0])
        self.assertEqual(kcat.sources, ["model"])
        self.assertEqual(kcat.selected_sequences, ["SEQ1"])
        details = json.loads(kcat.extra_info[0])
        self.assertTrue(details[0]["selected"])
        self.assertFalse(details[2]["selected"])

        km = reduce_sequence_predictions(
            plan=plan,
            target="Km",
            child_predictions=[5, 2, 3],
            child_sources=["model", "model", "model"],
            child_errors={},
            child_details=None,
            reaction_count=1,
        )
        self.assertEqual(km.predictions, [2.0])
        self.assertEqual(km.selected_sequences, ["SEQ2"])

    def test_substrate_arrays_reduce_per_substrate_across_sequences(self):
        seq_plan = SequenceExpansionPlan.build(
            ["SEQ1;SEQ2"],
            [0],
            limit=99,
            handle_long_sequences="truncate",
        )
        units = (
            TargetPredictionUnit(0, 0, 0, "SEQ1", 0, "A"),
            TargetPredictionUnit(0, 0, 0, "SEQ1", 1, "B"),
            TargetPredictionUnit(0, 1, 1, "SEQ2", 0, "A"),
            TargetPredictionUnit(0, 1, 1, "SEQ2", 1, "B"),
        )
        plan = TargetExpansionPlan(
            sequence_plan=seq_plan,
            units=units,
            reaction_slices=((0, 0, 4),),
            substrate_tokens_by_reaction=(("A", "B"),),
            uses_substrate_slots=True,
        )

        km = reduce_sequence_predictions(
            plan=plan,
            target="Km",
            child_predictions=[5, 9, 2, None],
            child_sources=["model", "model", "model", ""],
            child_errors={3: "Prediction failed"},
            child_details=["", "", "reported", ""],
            reaction_count=1,
        )
        self.assertEqual(km.predictions, ["[2.0,9.0]"])
        details = json.loads(km.extra_info[0])
        self.assertEqual(details[0]["substrates"][0]["selected"], False)
        self.assertEqual(details[1]["substrates"][0]["selected"], True)

        efficiency = reduce_sequence_predictions(
            plan=plan,
            target="kcat/Km",
            child_predictions=[5, 9, 2, math.nan],
            child_sources=["model", "model", "model", ""],
            child_errors={},
            child_details=None,
            reaction_count=1,
        )
        self.assertEqual(efficiency.predictions, ["[5.0,9.0]"])

    def test_all_failed_or_empty_reactions_are_blank(self):
        seq_plan = SequenceExpansionPlan.build(
            [";;"],
            [0],
            limit=99,
            handle_long_sequences="truncate",
        )
        plan = TargetExpansionPlan(
            sequence_plan=seq_plan,
            units=tuple(),
            reaction_slices=((0, 0, 0),),
            substrate_tokens_by_reaction=(tuple(),),
        )
        reduced = reduce_sequence_predictions(
            plan=plan,
            target="kcat",
            child_predictions=[],
            child_sources=[],
            child_errors={},
            child_details=None,
            reaction_count=1,
        )
        self.assertEqual(reduced.predictions, [""])
        self.assertEqual(reduced.failed_reactions, {0: "Missing protein sequence"})


if __name__ == "__main__":
    unittest.main()
