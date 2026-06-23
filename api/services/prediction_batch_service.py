"""Pure positional planning for prediction engine batches.

The planner is deliberately free of database, cache, model, and progress side
effects.  Both the request-time ReconXKG preflight and the worker use these
objects, so a cache hit is proved against the exact units that execution will
consume.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from api.methods.base import PredictionError
from api.utils.handle_long import get_valid_indices, truncate_sequences
from api.utils.substrate_expansion import SubstrateExpansionPlan

try:
    from webKinPred.config_docker import SERVER_LIMIT
except ImportError:
    from webKinPred.config_local import SERVER_LIMIT


@dataclass(frozen=True)
class SequenceBatchPlan:
    """Position-safe sequence handling shared by all selected targets."""

    original_sequences: tuple[Any, ...]
    processed_by_reaction: tuple[Any | None, ...]
    valid_reaction_indices: tuple[int, ...]
    skipped_reactions: dict[int, str]


@dataclass(frozen=True)
class TargetBatchPlan:
    """One method/target engine call after input-schema adaptation."""

    target: str
    input_behavior: str
    sequences: tuple[Any, ...]
    call_kwargs: dict[str, Any]
    expansion: SubstrateExpansionPlan | None = None


def build_sequence_batch_plan(
    dataframe: pd.DataFrame,
    descriptors: Iterable[Any],
    handle_long_sequences: str,
) -> SequenceBatchPlan:
    sequences = dataframe["Protein Sequence"].tolist()
    limits = [min(SERVER_LIMIT, desc.max_seq_len) for desc in descriptors]
    limit = min(limits) if limits else SERVER_LIMIT

    if handle_long_sequences == "truncate":
        processed, valid_indices = truncate_sequences(sequences, limit)
    else:
        valid_indices = get_valid_indices(sequences, limit, mode="skip")
        processed = [sequences[index] for index in valid_indices]

    processed_by_reaction: list[Any | None] = [None] * len(sequences)
    for local_index, reaction_index in enumerate(valid_indices):
        processed_by_reaction[reaction_index] = processed[local_index]

    valid_set = set(valid_indices)
    skipped = {
        index: "Sequence too long — row was excluded"
        for index in range(len(sequences))
        if index not in valid_set
    }
    return SequenceBatchPlan(
        original_sequences=tuple(sequences),
        processed_by_reaction=tuple(processed_by_reaction),
        valid_reaction_indices=tuple(valid_indices),
        skipped_reactions=skipped,
    )


def build_target_batch_plan(
    descriptor: Any,
    target: str,
    dataframe: pd.DataFrame,
    sequence_plan: SequenceBatchPlan,
) -> TargetBatchPlan:
    """Build the exact positional engine batch for one selected target."""
    behavior = descriptor.input_behavior(target)
    valid_indices = sequence_plan.valid_reaction_indices

    if (
        behavior == "expanded_pair"
        and "Substrate" in descriptor.col_to_kwarg
        and "Substrates" in dataframe.columns
    ):
        expansion = SubstrateExpansionPlan.build(
            dataframe["Substrates"].tolist(),
            valid_indices,
        )
        kwargs = _expanded_call_kwargs(descriptor, dataframe, expansion)
        kwargs.update(descriptor.target_kwargs.get(target, {}))
        return TargetBatchPlan(
            target=target,
            input_behavior=behavior,
            sequences=tuple(expansion.expanded_sequences(sequence_plan.processed_by_reaction)),
            call_kwargs=kwargs,
            expansion=expansion,
        )

    expansion = None
    if behavior == "native_multi" and "Substrates" in dataframe.columns:
        expansion = SubstrateExpansionPlan.build(
            dataframe["Substrates"].tolist(),
            valid_indices,
        )
        kwargs = _native_multi_call_kwargs(descriptor, dataframe, expansion)
    else:
        kwargs = _native_call_kwargs(descriptor, dataframe, valid_indices)

    kwargs.update(descriptor.target_kwargs.get(target, {}))
    return TargetBatchPlan(
        target=target,
        input_behavior=behavior,
        sequences=tuple(
            sequence_plan.processed_by_reaction[index] for index in valid_indices
        ),
        call_kwargs=kwargs,
        expansion=expansion,
    )


def _expanded_call_kwargs(
    descriptor: Any,
    dataframe: pd.DataFrame,
    expansion: SubstrateExpansionPlan,
) -> dict[str, list[Any]]:
    kwargs: dict[str, list[Any]] = {}
    for column, kwarg_name in descriptor.col_to_kwarg.items():
        if column == "Substrate":
            kwargs[kwarg_name] = expansion.expanded_substrates()
        else:
            _require_column(descriptor, dataframe, column)
            kwargs[kwarg_name] = expansion.expanded_parent_values(
                dataframe[column].tolist()
            )
    return kwargs


def _native_multi_call_kwargs(
    descriptor: Any,
    dataframe: pd.DataFrame,
    expansion: SubstrateExpansionPlan,
) -> dict[str, list[Any]]:
    kwargs: dict[str, list[Any]] = {}
    for column, kwarg_name in descriptor.col_to_kwarg.items():
        if column == "Substrate":
            kwargs[kwarg_name] = [
                [expansion.children[index].substrate for index in range(start, end)]
                for _reaction_position, start, end in expansion.reaction_slices
            ]
        else:
            _require_column(descriptor, dataframe, column)
            kwargs[kwarg_name] = [
                dataframe[column].iloc[reaction_position]
                for reaction_position in expansion.reaction_positions
            ]
    return kwargs


def _native_call_kwargs(
    descriptor: Any,
    dataframe: pd.DataFrame,
    reaction_indices: tuple[int, ...],
) -> dict[str, list[Any]]:
    kwargs: dict[str, list[Any]] = {}
    for column, kwarg_name in descriptor.col_to_kwarg.items():
        _require_column(descriptor, dataframe, column)
        kwargs[kwarg_name] = [
            dataframe[column].iloc[reaction_index]
            for reaction_index in reaction_indices
        ]
    return kwargs


def _require_column(descriptor: Any, dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        raise PredictionError(
            f"Missing column required for {descriptor.display_name}: {column}"
        )
