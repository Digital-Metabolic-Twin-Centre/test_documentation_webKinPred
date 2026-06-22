# api/tasks.py
#
# Celery tasks for running kinetic-parameter predictions.
#
# There are two entry-point tasks:
#
#   run_prediction(public_id, method_key, target, experimental_results)
#       Used for single-target jobs (prediction_type = "kcat" or "Km").
#
#   run_both_prediction(public_id, kcat_key, km_key, experimental_results)
#       Legacy dual-target helper kept for compatibility with internal tools.
#       New submissions use run_multi_prediction.
#
#   run_multi_prediction(public_id, targets, methods, experimental_results)
#       Used by the current submission flow for one or more selected targets.
#
# Both tasks delegate to internal helpers (_execute_prediction /
# _execute_both_prediction) that contain the shared prediction logic.
# Adding a new method requires no changes here — it is picked up automatically
# by the method registry.

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd
from celery import shared_task
from django.conf import settings
from django.utils import timezone

from api.methods.base import PredictionError
from api.methods.registry import get as get_method
from api.models import Job
from api.observability.context import log_context
from api.prediction_engines.generic_subprocess import run_generic_subprocess_prediction
from api.services.gpu_precompute_status_service import clear_gpu_precompute_status
from api.services.job_progress_service import (
    initialise_job_progress_stages,
    mark_running_stage_failed,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_running,
    set_stage_prediction_snapshot,
)
from api.services.similarity_service import append_kcat_similarity_columns_to_output_csv
from api.services.about_stats_service import refresh_about_stats_cache
from api.utils.extra_info import _source, build_extra_info
from api.utils.handle_long import get_valid_indices, truncate_sequences
from api.utils.job_utils import canonicalise_targets
from api.utils.quotas import credit_back
from api.utils.safe_read import safe_read_csv
from api.utils.substrate_expansion import (
    SubstrateExpansionPlan,
    reduce_substrate_predictions,
)

try:
    from webKinPred.config_docker import SERVER_LIMIT
except ImportError:
    from webKinPred.config_local import SERVER_LIMIT

_log = logging.getLogger(__name__)

_REALKCAT_CLASS_RANGES: dict[str, dict[int, tuple[float, float]]] = {
    "kcat": {
        0: (0.0, 3.32e-8),
        1: (3.33e-8, 1.0e-2),
        2: (1.01e-2, 1.0e-1),
        3: (1.01e-1, 1.0),
        4: (1.001, 10.0),
        5: (1.004e1, 1.0e2),
        6: (1.0025e2, 1.0e3),
        7: (1.002e3, 7.0e7),
    },
    "Km": {
        0: (1.0e-10, 1.0e-5),
        1: (1.01e-5, 1.0e-4),
        2: (1.002e-4, 1.0e-3),
        3: (1.002e-3, 1.0e-2),
        4: (1.008e-2, 1.0e-1),
        5: (1.01e-1, 1.02e2),
    },
}


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------


def _safe_clear_gpu_precompute_status(public_id: str) -> None:
    try:
        clear_gpu_precompute_status(public_id)
    except Exception:
        # Redis telemetry cleanup is best-effort only.
        pass


def _safe_refresh_about_stats_cache() -> None:
    try:
        refresh_about_stats_cache(force=True)
    except Exception:
        # About stats are non-critical and should never break jobs.
        pass


def _format_prediction_value_and_source(desc, target: str, prediction: Any) -> tuple[Any, str]:
    """
    Return the output value and source string for one predicted row.

    For RealKcat only, convert class-index outputs to class-range mean values
    and include classifier context in the source text.
    """
    default_source = f"Prediction from {desc.display_name}"

    if desc.key != "RealKcat":
        return prediction, default_source

    if prediction is None:
        return None, default_source

    ranges = _REALKCAT_CLASS_RANGES.get(target)
    if not ranges:
        return prediction, default_source

    try:
        class_idx = int(prediction)
    except (TypeError, ValueError):
        return prediction, default_source

    class_range = ranges.get(class_idx)
    if class_range is None:
        return prediction, default_source

    low, high = class_range
    mean_value = (low + high) / 2.0
    source = (
        "Prediction from RealKcat (classifier): "
        f"class {class_idx}, range {low:.3e} to {high:.3e}; "
        "value is class-range mean."
    )
    return mean_value, source


@shared_task
def run_prediction(
    public_id: str,
    method_key: str,
    target: str,
    experimental_results: list | None = None,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """
    Run a single-target prediction job.

    Parameters
    ----------
    public_id : str
        The job's public identifier.
    method_key : str
        Registry key of the prediction method (e.g. ``"DLKcat"``).
    target : str
        Prediction target: ``"kcat"`` or ``"Km"``.
    experimental_results : list | None
        Pre-fetched experimental values to merge into the output, or None.
    """
    job = Job.objects.get(public_id=public_id)
    _safe_clear_gpu_precompute_status(public_id)
    job.status = "Processing"
    job.start_time = timezone.now()
    job.save(update_fields=["status", "start_time"])

    desc = get_method(method_key)

    try:
        df = _load_input(job)
        _execute_prediction(
            job,
            desc,
            df,
            target,
            experimental_results or [],
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
        )
        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
        )
        _safe_refresh_about_stats_cache()

    except PredictionError as e:
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=str(e),
            completion_time=timezone.now(),
        )

    except MemoryError:
        _handle_oom(job, desc.display_name)

    except Exception as e:
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=_sanitise_unexpected(e, desc.display_name),
            completion_time=timezone.now(),
        )


@shared_task
def run_both_prediction(
    public_id: str,
    kcat_key: str,
    km_key: str,
    experimental_results: list | None = None,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """
    Run a dual-target prediction job (kcat and KM in sequence).

    Parameters
    ----------
    public_id : str
        The job's public identifier.
    kcat_key : str
        Registry key of the kcat prediction method.
    km_key : str
        Registry key of the KM prediction method.
    experimental_results : list | None
        Pre-fetched experimental values to merge into the output, or None.
    """
    job = Job.objects.get(public_id=public_id)
    _safe_clear_gpu_precompute_status(public_id)
    job.status = "Processing"
    job.start_time = timezone.now()
    job.predictions_made = 0
    job.total_predictions = 0
    job.save(update_fields=["status", "start_time", "predictions_made", "total_predictions"])

    kcat_desc = get_method(kcat_key)
    km_desc = get_method(km_key)

    try:
        df = _load_input(job)
        _execute_both_prediction(
            job,
            kcat_desc,
            km_desc,
            df,
            experimental_results or [],
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
        )
        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
        )
        _safe_refresh_about_stats_cache()

    except PredictionError as e:
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=str(e),
            completion_time=timezone.now(),
        )

    except MemoryError:
        _handle_oom(job, f"{kcat_desc.display_name}/{km_desc.display_name}")

    except Exception as e:
        label = f"{kcat_desc.display_name}/{km_desc.display_name}"
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=_sanitise_unexpected(e, label),
            completion_time=timezone.now(),
        )


@shared_task
def run_multi_prediction(
    public_id: str,
    targets: list[str],
    methods: dict[str, str],
    experimental_results: dict | None = None,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """
    Run a multi-target prediction job.

    Parameters
    ----------
    public_id : str
        The job's public identifier.
    targets : list[str]
        Selected targets, subset of ``["kcat", "Km", "kcat/Km"]``.
    methods : dict[str, str]
        Mapping target -> method key.
    experimental_results : dict | None
        Optional pre-fetched experimental rows keyed by target.
    """
    job = Job.objects.get(public_id=public_id)
    if job.status == "Completed":
        _log.info(
            "Skipping duplicate task execution — job already completed",
            extra={"event": "task.duplicate_skipped", "job_public_id": public_id},
        )
        return
    _safe_clear_gpu_precompute_status(public_id)
    job.status = "Processing"
    job.start_time = timezone.now()
    job.predictions_made = 0
    job.total_predictions = 0
    job.save(update_fields=["status", "start_time", "predictions_made", "total_predictions"])

    ordered_targets = canonicalise_targets(targets)
    if not ordered_targets:
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message="No prediction targets were provided.",
            completion_time=timezone.now(),
        )
        return

    try:
        desc_by_target = {target: get_method(methods[target]) for target in ordered_targets}
    except Exception as e:
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=f"Invalid method selection: {e}",
            completion_time=timezone.now(),
        )
        return

    initialise_job_progress_stages(job, ordered_targets, desc_by_target)

    try:
        df = _load_input(job)
        _execute_multi_prediction(
            job=job,
            targets=ordered_targets,
            desc_by_target=desc_by_target,
            df=df,
            experimental_results=experimental_results or {},
            canonicalize_substrates=canonicalize_substrates,
            include_similarity_columns=include_similarity_columns,
            disable_gpu_precompute=disable_gpu_precompute,
        )
        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
        )
        _safe_refresh_about_stats_cache()

    except PredictionError as e:
        mark_running_stage_failed(public_id, message=str(e))
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=str(e),
            completion_time=timezone.now(),
        )

    except MemoryError:
        mark_running_stage_failed(public_id, message="Out of memory.")
        label = "/".join(desc.display_name for desc in desc_by_target.values())
        _handle_oom(job, label)

    except Exception as e:
        mark_running_stage_failed(public_id, message=str(e))
        label = "/".join(desc.display_name for desc in desc_by_target.values())
        Job.objects.filter(pk=job.pk).update(
            status="Failed",
            error_message=_sanitise_unexpected(e, label),
            completion_time=timezone.now(),
        )


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------


def _execute_multi_prediction(
    job: Job,
    targets: list[str],
    desc_by_target: dict,
    df: pd.DataFrame,
    experimental_results: dict[str, list],
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """Run targets through the canonical reaction/child orchestration path."""
    sequences = df["Protein Sequence"].tolist()
    n_rows = len(sequences)
    limits = [min(SERVER_LIMIT, desc.max_seq_len) for desc in desc_by_target.values()]
    limit = min(limits) if limits else SERVER_LIMIT

    if job.handle_long_sequences == "truncate":
        sequences_proc, valid_idx = truncate_sequences(sequences, limit)
    else:
        valid_idx = get_valid_indices(sequences, limit, mode="skip")
        sequences_proc = [sequences[index] for index in valid_idx]

    processed_by_reaction: list[Any] = [None] * n_rows
    for local_index, reaction_index in enumerate(valid_idx):
        processed_by_reaction[reaction_index] = sequences_proc[local_index]

    sequence_skips: dict[int, str] = {
        index: "Sequence too long — row was excluded"
        for index in set(range(n_rows)) - set(valid_idx)
    }
    reported_skips = dict(sequence_skips)
    target_results: dict[str, dict[str, Any]] = {}

    eitlem_targets = [target for target in targets if desc_by_target[target].key == "EITLEM"]
    last_eitlem_target = eitlem_targets[-1] if eitlem_targets else None
    omniesi_targets = [target for target in targets if desc_by_target[target].key == "OmniESI"]
    last_omniesi_target = omniesi_targets[-1] if omniesi_targets else None

    for target in targets:
        desc = desc_by_target[target]
        mark_stage_running(job.public_id, target, desc.key)
        extra_call_kwargs: dict[str, Any] = {}
        if desc.key == "EITLEM":
            extra_call_kwargs["cleanup_esm1v_embeddings"] = target == last_eitlem_target
        if desc.key == "OmniESI":
            extra_call_kwargs["cleanup_embeddings_after_run"] = target == last_omniesi_target

        try:
            result = _execute_target_batch(
                job=job,
                desc=desc,
                df=df,
                target=target,
                sequences=sequences,
                processed_by_reaction=processed_by_reaction,
                valid_reaction_indices=valid_idx,
                experimental_results=experimental_results.get(target, []),
                canonicalize_substrates=canonicalize_substrates,
                disable_gpu_precompute=disable_gpu_precompute,
                extra_call_kwargs=extra_call_kwargs,
            )
        except Exception as exc:
            mark_stage_failed(job.public_id, target, desc.key, message=str(exc))
            raise

        for reaction_index, reason in result["failed_reactions"].items():
            _append_skip_reason(reported_skips, reaction_index, f"{target}: {reason}")
        target_results[target] = result
        mark_stage_completed(job.public_id, target, desc.key)

    for reaction_index, reason in sequence_skips.items():
        for result in target_results.values():
            result["preds"][reaction_index] = ""
            result["sources"][reaction_index] = reason
            result["extra"][reaction_index] = ""

    results_df = df.copy()
    preferred_cols: list[str] = []
    for target in targets:
        result = target_results[target]
        pred_col = result["output_col"]
        source_col = f"Source {target}"
        extra_col = f"Extra Info {target}"
        results_df[pred_col] = result["preds"]
        results_df[source_col] = result["sources"]
        results_df[extra_col] = result["extra"]
        preferred_cols.extend([pred_col, source_col, extra_col])

    results_df = results_df[
        preferred_cols + [column for column in results_df.columns if column not in preferred_cols]
    ]
    out_path = _output_path(job.public_id)
    results_df.to_csv(out_path, index=False)
    if include_similarity_columns and "kcat" in targets:
        append_kcat_similarity_columns_to_output_csv(out_path, desc_by_target["kcat"].key)

    fully_predicted = pd.Series(True, index=results_df.index)
    for target in targets:
        pred_col = target_results[target]["output_col"]
        fully_predicted = (
            fully_predicted & (results_df[pred_col] != "") & results_df[pred_col].notna()
        )
    processed_reactions = int(fully_predicted.sum())
    to_refund = max(0, int(job.requested_rows) - processed_reactions)
    if to_refund:
        credit_back(_job_quota_subject(job), to_refund)

    Job.objects.filter(pk=job.pk).update(
        output_file=os.path.relpath(out_path, settings.MEDIA_ROOT),
        error_message=_build_skipped_message(reported_skips),
    )


def _execute_target_batch(
    *,
    job: Job,
    desc,
    df: pd.DataFrame,
    target: str,
    sequences: list[Any],
    processed_by_reaction: list[Any],
    valid_reaction_indices: list[int],
    experimental_results: list[dict[str, Any]],
    canonicalize_substrates: bool,
    disable_gpu_precompute: bool,
    extra_call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Execute one method/target either natively or as an expanded child batch."""
    n_rows = len(sequences)
    if (
        desc.input_format == "single"
        and "Substrate" in desc.col_to_kwarg
        and "Substrates" in df.columns
    ):
        plan = SubstrateExpansionPlan.build(
            df["Substrates"].tolist(),
            valid_reaction_indices,
        )
        child_sequences = plan.expanded_sequences(processed_by_reaction)
        call_kwargs = _build_expanded_call_kwargs(desc, df, plan)
        call_kwargs.update(desc.target_kwargs.get(target, {}))
        call_kwargs.update(extra_call_kwargs)

        if child_sequences:
            child_predictions, child_errors = _invoke_method_prediction(
                desc=desc,
                sequences=child_sequences,
                public_id=job.public_id,
                target=target,
                canonicalize_substrates=canonicalize_substrates,
                disable_gpu_precompute=disable_gpu_precompute,
                **call_kwargs,
            )
        else:
            child_predictions, child_errors = [], {}
            set_stage_prediction_snapshot(
                job_public_id=job.public_id,
                target=target,
                method_key=desc.key,
                molecules_total=0,
                molecules_processed=0,
                invalid_rows=0,
                predictions_total=0,
                predictions_made=0,
            )

        formatted_values: list[Any] = []
        formatted_sources: list[str] = []
        child_details: list[str] = [""] * len(plan.children)
        for child_index in range(len(plan.children)):
            value, source = _format_prediction_value_and_source(
                desc,
                target,
                child_predictions[child_index],
            )
            formatted_values.append(value)
            formatted_sources.append(source if value is not None else "")

        _apply_expanded_experimental_overrides(
            job=job,
            desc=desc,
            target=target,
            sequences=sequences,
            plan=plan,
            experimental_results=experimental_results,
            values=formatted_values,
            sources=formatted_sources,
            details=child_details,
            child_errors=child_errors,
        )
        try:
            reduced = reduce_substrate_predictions(
                plan=plan,
                target=target,
                child_predictions=formatted_values,
                child_sources=formatted_sources,
                child_errors=child_errors,
                child_details=child_details,
                reaction_count=n_rows,
            )
        except ValueError as exc:
            raise PredictionError(f"{desc.display_name} result mapping failed: {exc}") from exc
        return {
            "preds": reduced.predictions,
            "sources": reduced.sources,
            "extra": reduced.extra_info,
            "failed_reactions": reduced.failed_reactions,
            "output_col": desc.output_cols[target],
        }

    return _execute_native_target_batch(
        job=job,
        desc=desc,
        df=df,
        target=target,
        sequences=sequences,
        processed_by_reaction=processed_by_reaction,
        valid_reaction_indices=valid_reaction_indices,
        experimental_results=experimental_results,
        canonicalize_substrates=canonicalize_substrates,
        disable_gpu_precompute=disable_gpu_precompute,
        extra_call_kwargs=extra_call_kwargs,
    )


def _build_expanded_call_kwargs(
    desc,
    df: pd.DataFrame,
    plan: SubstrateExpansionPlan,
) -> dict[str, list[Any]]:
    call_kwargs: dict[str, list[Any]] = {}
    for column, kwarg_name in desc.col_to_kwarg.items():
        if column == "Substrate":
            call_kwargs[kwarg_name] = plan.expanded_substrates()
            continue
        if column not in df.columns:
            raise PredictionError(
                f"Missing column required for {desc.display_name}: {column}"
            )
        call_kwargs[kwarg_name] = plan.expanded_parent_values(df[column].tolist())
    return call_kwargs


def _execute_native_target_batch(
    *,
    job: Job,
    desc,
    df: pd.DataFrame,
    target: str,
    sequences: list[Any],
    processed_by_reaction: list[Any],
    valid_reaction_indices: list[int],
    experimental_results: list[dict[str, Any]],
    canonicalize_substrates: bool,
    disable_gpu_precompute: bool,
    extra_call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    n_rows = len(sequences)
    predictions: list[Any] = [""] * n_rows
    sources: list[str] = [""] * n_rows
    extra: list[str] = [""] * n_rows
    failed_reactions: dict[int, str] = {}
    call_kwargs: dict[str, Any] = {}
    for column, kwarg_name in desc.col_to_kwarg.items():
        if column not in df.columns:
            raise PredictionError(
                f"Missing column required for {desc.display_name}: {column}"
            )
        call_kwargs[kwarg_name] = [
            df[column].iloc[reaction_index] for reaction_index in valid_reaction_indices
        ]
    call_kwargs.update(desc.target_kwargs.get(target, {}))
    call_kwargs.update(extra_call_kwargs)
    processed_sequences = [processed_by_reaction[index] for index in valid_reaction_indices]

    if processed_sequences:
        subset, invalid_subset = _invoke_method_prediction(
            desc=desc,
            sequences=processed_sequences,
            public_id=job.public_id,
            target=target,
            canonicalize_substrates=canonicalize_substrates,
            disable_gpu_precompute=disable_gpu_precompute,
            **call_kwargs,
        )
        for local_index, reaction_index in enumerate(valid_reaction_indices):
            value, source = _format_prediction_value_and_source(
                desc,
                target,
                subset[local_index],
            )
            if _prediction_is_missing(value):
                reason = invalid_subset.get(local_index, "Prediction could not be made")
                sources[reaction_index] = reason
                failed_reactions[reaction_index] = reason
            else:
                predictions[reaction_index] = value
                sources[reaction_index] = source

        for local_index, reason in invalid_subset.items():
            if 0 <= local_index < len(valid_reaction_indices):
                reaction_index = valid_reaction_indices[local_index]
                predictions[reaction_index] = ""
                sources[reaction_index] = reason
                failed_reactions[reaction_index] = reason
    else:
        set_stage_prediction_snapshot(
            job_public_id=job.public_id,
            target=target,
            method_key=desc.key,
            molecules_total=n_rows,
            molecules_processed=0,
            invalid_rows=0,
            predictions_total=0,
            predictions_made=0,
        )

    _apply_native_experimental_overrides(
        job=job,
        desc=desc,
        target=target,
        sequences=sequences,
        experimental_results=experimental_results,
        predictions=predictions,
        sources=sources,
        extra=extra,
        failed_reactions=failed_reactions,
    )
    return {
        "preds": predictions,
        "sources": sources,
        "extra": extra,
        "failed_reactions": failed_reactions,
        "output_col": desc.output_cols[target],
    }


def _apply_expanded_experimental_overrides(
    *,
    job: Job,
    desc,
    target: str,
    sequences: list[Any],
    plan: SubstrateExpansionPlan,
    experimental_results: list[dict[str, Any]],
    values: list[Any],
    sources: list[str],
    details: list[str],
    child_errors: dict[int, str],
) -> None:
    if target not in {"kcat", "Km"}:
        return
    exp_key = "kcat_value" if target == "kcat" else "km_value"
    by_position = {
        (item.get("reaction_idx"), item.get("substrate_idx")): item
        for item in experimental_results
        if item.get("found")
    }
    for child_index, child in enumerate(plan.children):
        exp = by_position.get((child.reaction_position, child.substrate_position))
        if not exp or exp_key not in exp:
            continue
        if not _experimental_sequence_matches(exp, sequences[child.reaction_position]):
            _log_experimental_mismatch(job, desc, target, child.reaction_position)
            continue
        previous = values[child_index]
        values[child_index] = exp[exp_key]
        sources[child_index] = _source(exp)
        details[child_index] = build_extra_info(exp, target, previous, desc.display_name)
        child_errors.pop(child_index, None)


def _apply_native_experimental_overrides(
    *,
    job: Job,
    desc,
    target: str,
    sequences: list[Any],
    experimental_results: list[dict[str, Any]],
    predictions: list[Any],
    sources: list[str],
    extra: list[str],
    failed_reactions: dict[int, str],
) -> None:
    if target not in {"kcat", "Km"}:
        return
    exp_key = "kcat_value" if target == "kcat" else "km_value"
    for exp in experimental_results:
        if not exp.get("found") or exp_key not in exp:
            continue
        reaction_index = exp.get("reaction_idx", exp.get("idx"))
        if not isinstance(reaction_index, int) or not 0 <= reaction_index < len(sequences):
            continue
        if not _experimental_sequence_matches(exp, sequences[reaction_index]):
            _log_experimental_mismatch(job, desc, target, reaction_index)
            continue
        previous = predictions[reaction_index]
        predictions[reaction_index] = exp[exp_key]
        sources[reaction_index] = _source(exp)
        extra[reaction_index] = build_extra_info(exp, target, previous, desc.display_name)
        failed_reactions.pop(reaction_index, None)


def _experimental_sequence_matches(exp: dict[str, Any], sequence: Any) -> bool:
    recorded = exp.get("protein_sequence")
    return recorded is None or recorded == sequence


def _log_experimental_mismatch(job: Job, desc, target: str, reaction_index: int) -> None:
    _log.warning(
        "Skipping experimental overwrite because protein sequence mismatched",
        extra={
            "event": "prediction.experimental_overwrite_mismatch",
            "job_public_id": job.public_id,
            "method_key": desc.key,
            "target": target,
            "row_index": reaction_index,
        },
    )


def _prediction_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"none", "nan"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _append_skip_reason(reasons: dict[int, str], row_index: int, reason: str) -> None:
    previous = reasons.get(row_index, "")
    if reason and reason not in previous:
        reasons[row_index] = f"{previous}; {reason}" if previous else reason


def _execute_prediction(
    job: Job,
    desc,
    df: pd.DataFrame,
    target: str,
    experimental_results: list,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """Compatibility wrapper around the canonical multi-target executor."""
    _execute_multi_prediction(
        job=job,
        targets=[target],
        desc_by_target={target: desc},
        df=df,
        experimental_results={target: experimental_results},
        canonicalize_substrates=canonicalize_substrates,
        include_similarity_columns=include_similarity_columns,
        disable_gpu_precompute=disable_gpu_precompute,
    )


def _execute_both_prediction(
    job: Job,
    kcat_desc,
    km_desc,
    df: pd.DataFrame,
    experimental_results: list,
    canonicalize_substrates: bool = True,
    include_similarity_columns: bool = True,
    disable_gpu_precompute: bool = False,
) -> None:
    """Compatibility wrapper around the canonical multi-target executor."""
    _execute_multi_prediction(
        job=job,
        targets=["kcat", "Km"],
        desc_by_target={"kcat": kcat_desc, "Km": km_desc},
        df=df,
        experimental_results={
            "kcat": [item for item in experimental_results if "kcat_value" in item],
            "Km": [item for item in experimental_results if "km_value" in item],
        },
        canonicalize_substrates=canonicalize_substrates,
        include_similarity_columns=include_similarity_columns,
        disable_gpu_precompute=disable_gpu_precompute,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _invoke_method_prediction(
    desc,
    sequences: list[str],
    public_id: str,
    target: str,
    canonicalize_substrates: bool = True,
    disable_gpu_precompute: bool = False,
    **call_kwargs,
) -> tuple[list, dict[int, str]]:
    """
    Invoke a method using either:

    1) a custom `pred_func` (legacy/current methods), or
    2) the built-in generic subprocess engine (recommended for new methods).

    Always returns (predictions, invalid_reasons) where invalid_reasons maps
    local indices (into sequences) to human-readable skip reasons.
    """
    call_kwargs = dict(call_kwargs)
    call_kwargs.setdefault("canonicalize_substrates", canonicalize_substrates)
    call_kwargs.setdefault("disable_gpu_precompute", disable_gpu_precompute)

    with log_context(job_public_id=public_id, method_key=desc.key, target=target):
        if desc.pred_func is not None:
            preds, invalid_result = desc.pred_func(
                sequences=sequences,
                public_id=public_id,
                **call_kwargs,
            )
            if isinstance(invalid_result, dict):
                invalid_reasons = invalid_result
            else:
                invalid_reasons = {
                    idx: "Prediction could not be made" for idx in (invalid_result or [])
                }
            return _validate_method_result(desc, sequences, preds, invalid_reasons)

        if desc.subprocess is not None:
            preds, invalid_reasons = run_generic_subprocess_prediction(
                desc=desc,
                sequences=sequences,
                public_id=public_id,
                target=target,
                **call_kwargs,
            )
            return _validate_method_result(desc, sequences, preds, invalid_reasons)

    raise PredictionError(f"{desc.display_name} is not configured with a prediction engine.")


def _validate_method_result(
    desc,
    sequences: list[Any],
    predictions: Any,
    invalid_reasons: Any,
) -> tuple[list, dict[int, str]]:
    """Enforce the positional prediction-engine contract at one boundary."""
    if not isinstance(predictions, (list, tuple)):
        raise PredictionError(
            f"{desc.display_name} returned an invalid prediction result."
        )
    if len(predictions) != len(sequences):
        raise PredictionError(
            f"{desc.display_name} produced {len(predictions)} prediction(s) "
            f"for {len(sequences)} input(s)."
        )

    normalised_invalid: dict[int, str] = {}
    if isinstance(invalid_reasons, dict):
        for raw_index, raw_reason in invalid_reasons.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(sequences):
                reason = str(raw_reason or "").strip()
                normalised_invalid[index] = reason or "Prediction could not be made"
            else:
                _log.warning(
                    "Prediction method returned an out-of-range invalid index",
                    extra={
                        "event": "prediction.invalid_index_out_of_range",
                        "method_key": desc.key,
                        "row_index": index,
                    },
                )
    return list(predictions), normalised_invalid


def _map_subset_invalid_reasons(
    global_indices: list[int],
    invalid_reasons: dict[int, str],
) -> dict[int, str]:
    """Map local invalid reasons (keyed by position in sequences subset) to global row indices."""
    mapped: dict[int, str] = {}
    for local_idx, reason in invalid_reasons.items():
        try:
            idx = int(local_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(global_indices):
            mapped[global_indices[idx]] = reason
    return mapped


def _build_skipped_message(skipped_reasons: dict[int, str]) -> str:
    """Serialize per-row skip reasons as a JSON array grouped by reason."""
    if not skipped_reasons:
        return ""
    groups: dict[str, list[int]] = {}
    for idx, reason in skipped_reasons.items():
        groups.setdefault(reason, []).append(idx)
    return json.dumps([{"rows": sorted(rows), "reason": reason} for reason, rows in groups.items()])


def _load_input(job: Job) -> pd.DataFrame:
    """Read the job's input CSV, crediting back quota on failure."""
    path = os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv")
    df = safe_read_csv(path, _job_quota_subject(job), job.requested_rows)
    if df is None:
        raise PredictionError(
            "The uploaded CSV file could not be read. "
            "Please ensure it is a valid CSV and try again."
        )
    return df


def _output_path(public_id: str) -> str:
    return os.path.join(settings.MEDIA_ROOT, "jobs", str(public_id), "output.csv")


def _handle_oom(job: Job, label: str) -> None:
    """Mark job as failed with an out-of-memory message and credit back quota."""
    msg = (
        f"{label} prediction terminated due to insufficient memory. "
        "Try reducing the number of rows or the sequence lengths."
    )
    Job.objects.filter(pk=job.pk).update(
        status="Failed",
        error_message=msg,
        completion_time=timezone.now(),
    )
    credit_back(_job_quota_subject(job), job.requested_rows)


def _job_quota_subject(job: Job) -> str:
    """
    Resolve the quota subject used for this job.

    Older rows may not have ``quota_subject`` populated; fallback to the legacy
    IP-based key in that case.
    """
    return job.quota_subject or job.ip_address


def _sanitise_unexpected(exc: Exception, label: str) -> str:
    """
    Convert an unexpected (non-PredictionError) exception to a user-facing
    message, stripping internal paths and stack traces.
    """
    import re

    msg = str(exc)
    # If the message contains file paths or the word "Traceback", it's too
    # technical for a user.  Replace with a generic fallback.
    if re.search(r"/[a-z_/]+\.[a-z]+", msg, re.IGNORECASE) or "Traceback" in msg:
        return (
            f"{label} prediction encountered an unexpected error. "
            "Please verify your input and try again."
        )
    return msg or (
        f"{label} prediction encountered an unexpected error. "
        "Please verify your input and try again."
    )
