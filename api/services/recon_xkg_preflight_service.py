"""Request-time, cache-only qualification for ReconXKG jobs."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any

import pandas as pd
from api.services import prediction_store
from api.services.prediction_batch_service import (
    build_sequence_batch_plan,
    build_target_batch_plan,
)
from api.services.similarity_service import similarity_cache_label_for_method

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconXkgCacheSnapshot:
    """Values captured by a successful full-cache preflight."""

    predictions: dict[str, Any]
    similarities: dict[str, tuple[float | None, float | None]]


@dataclass(frozen=True)
class ReconXkgPreflightResult:
    complete: bool
    snapshot: ReconXkgCacheSnapshot | None
    reason: str
    prediction_units: int
    unique_prediction_keys: int
    similarity_sequences: int


def preflight_recon_xkg_cache(
    *,
    dataframe: pd.DataFrame,
    targets: list[str],
    descriptors: dict[str, Any],
    handle_long_sequences: str,
    canonicalize_substrates: bool,
    include_similarity_columns: bool,
    job_public_id: str,
) -> ReconXkgPreflightResult:
    """Return a snapshot only when every required cached value is present."""
    started = time.monotonic()
    prediction_units = 0
    similarity_count = 0
    unique_keys: set[str] = set()
    reason = "cache-read-error"

    try:
        sequence_plan = build_sequence_batch_plan(
            dataframe,
            descriptors.values(),
            handle_long_sequences,
        )
        for target in targets:
            descriptor = descriptors[target]
            batch = build_target_batch_plan(
                descriptor,
                target,
                dataframe,
                sequence_plan,
            )
            keys, _components, _params_fp = prediction_store.build_unit_keys(
                descriptor,
                target,
                batch.sequences,
                batch.call_kwargs,
                canonicalize_substrates,
            )
            prediction_units += len(keys)
            unique_keys.update(key for key in keys if key is not None)
            if any(key is None for key in keys):
                reason = "uncacheable-prediction-unit"
                return _logged_result(
                    False,
                    None,
                    reason,
                    prediction_units,
                    len(unique_keys),
                    0,
                    job_public_id,
                    started,
                )

        prediction_values = prediction_store.get_many(unique_keys)
        if any(
            key not in prediction_values
            or not prediction_store.cached_outcome_is_valid(prediction_values[key])
            for key in unique_keys
        ):
            reason = "prediction-cache-miss"
            return _logged_result(
                False,
                None,
                reason,
                prediction_units,
                len(unique_keys),
                0,
                job_public_id,
                started,
            )

        similarity_values: dict[str, tuple[float | None, float | None]] = {}
        if include_similarity_columns and "kcat" in targets:
            method_key = descriptors["kcat"].key
            cache_label = similarity_cache_label_for_method(method_key)
            if not cache_label:
                reason = "similarity-cache-unavailable"
                return _logged_result(
                    False,
                    None,
                    reason,
                    prediction_units,
                    len(unique_keys),
                    0,
                    job_public_id,
                    started,
                )

            raw_sequences = [
                str(sequence).strip()
                for sequence in dataframe["Protein Sequence"].fillna("").tolist()
            ]
            unique_sequences = list(dict.fromkeys(seq for seq in raw_sequences if seq))
            similarity_count = len(unique_sequences)
            sequence_hashes = {
                sequence: prediction_store.sha256_text(sequence)
                for sequence in unique_sequences
            }
            similarity_values = prediction_store.get_similarity_many(
                sequence_hashes,
                cache_label,
            )
            if any(
                sequence not in similarity_values
                or not _valid_similarity_entry(similarity_values[sequence])
                for sequence in unique_sequences
            ):
                reason = "similarity-cache-miss"
                return _logged_result(
                    False,
                    None,
                    reason,
                    prediction_units,
                    len(unique_keys),
                    similarity_count,
                    job_public_id,
                    started,
                )

        snapshot = ReconXkgCacheSnapshot(
            predictions=dict(prediction_values),
            similarities=dict(similarity_values),
        )
        return _logged_result(
            True,
            snapshot,
            "full-cache-hit",
            prediction_units,
            len(unique_keys),
            similarity_count,
            job_public_id,
            started,
        )
    except Exception:
        _log.warning(
            "ReconXKG immediate cache preflight failed; queueing normally",
            extra={
                "event": "recon_xkg.preflight_error",
                "job_public_id": job_public_id,
                "prediction_units": prediction_units,
                "unique_prediction_keys": len(unique_keys),
                "similarity_sequences": similarity_count,
            },
            exc_info=True,
        )
        return _logged_result(
            False,
            None,
            reason,
            prediction_units,
            len(unique_keys),
            similarity_count,
            job_public_id,
            started,
        )


def _logged_result(
    complete: bool,
    snapshot: ReconXkgCacheSnapshot | None,
    reason: str,
    prediction_units: int,
    unique_prediction_keys: int,
    similarity_sequences: int,
    job_public_id: str,
    started: float,
) -> ReconXkgPreflightResult:
    _log.info(
        "ReconXKG immediate cache preflight completed",
        extra={
            "event": "recon_xkg.preflight_hit" if complete else "recon_xkg.preflight_miss",
            "job_public_id": job_public_id,
            "reason": reason,
            "prediction_units": prediction_units,
            "unique_prediction_keys": unique_prediction_keys,
            "similarity_sequences": similarity_sequences,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        },
    )
    return ReconXkgPreflightResult(
        complete=complete,
        snapshot=snapshot,
        reason=reason,
        prediction_units=prediction_units,
        unique_prediction_keys=unique_prediction_keys,
        similarity_sequences=similarity_sequences,
    )


def _valid_similarity_entry(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(
        item is None or (isinstance(item, Real) and math.isfinite(float(item)))
        for item in value
    )
