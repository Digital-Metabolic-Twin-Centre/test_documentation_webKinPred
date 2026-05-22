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
    """
    Run a single-target prediction and write output.csv.

    Validates required columns, applies sequence-length handling, invokes the
    method's pred_func, merges experimental overwrites, writes the output CSV,
    and credits back rows that produced no prediction.
    """
    # ── 1. Validate required columns ──────────────────────────────────────────
    required_cols = ["Protein Sequence"] + list(desc.col_to_kwarg.keys())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s) required for {desc.display_name}: " + ", ".join(missing)
        )

    # ── 2. Extract sequences and apply length limit ───────────────────────────
    sequences = df["Protein Sequence"].tolist()
    limit = min(SERVER_LIMIT, desc.max_seq_len)

    if job.handle_long_sequences == "truncate":
        sequences_proc, valid_idx = truncate_sequences(sequences, limit)
    else:
        valid_idx = get_valid_indices(sequences, limit, mode="skip")
        sequences_proc = [sequences[i] for i in valid_idx]

    # ── 3. Build pred_func kwargs from descriptor ─────────────────────────────
    call_kwargs: dict = {}
    for col, kwarg_name in desc.col_to_kwarg.items():
        call_kwargs[kwarg_name] = [df[col].iloc[i] for i in valid_idx]
    call_kwargs.update(desc.target_kwargs.get(target, {}))
    if desc.key == "OmniESI":
        call_kwargs["cleanup_embeddings_after_run"] = True

    # ── 4. Run predictions ────────────────────────────────────────────────────
    output_col = desc.output_cols[target]
    n_rows = len(sequences)
    full_preds: list = [""] * n_rows
    sources: list[str] = [""] * n_rows
    extra_info: list[str] = [""] * n_rows

    skipped_reasons: dict[int, str] = {
        idx: "Sequence too long — row was excluded" for idx in set(range(n_rows)) - set(valid_idx)
    }

    if valid_idx:
        pred_subset, invalid_reasons = _invoke_method_prediction(
            desc=desc,
            sequences=sequences_proc,
            public_id=job.public_id,
            target=target,
            canonicalize_substrates=canonicalize_substrates,
            disable_gpu_precompute=disable_gpu_precompute,
            **call_kwargs,
        )
        for global_i, pred in zip(valid_idx, pred_subset):
            value_out, source_out = _format_prediction_value_and_source(desc, target, pred)
            full_preds[global_i] = value_out if value_out is not None else ""
            sources[global_i] = source_out
        skipped_reasons.update(_map_subset_invalid_reasons(valid_idx, invalid_reasons))

    for idx, reason in skipped_reasons.items():
        sources[idx] = reason
        full_preds[idx] = ""

    # ── 5. Apply experimental overwrites ──────────────────────────────────────
    exp_key = "km_value" if target == "Km" else "kcat_value"
    for exp in experimental_results:
        if not exp.get("found"):
            continue
        idx = exp["idx"]
        if exp.get("protein_sequence") != sequences[idx]:
            _log.warning(
                "Skipping experimental overwrite because protein sequence mismatched",
                extra={
                    "event": "prediction.experimental_overwrite_mismatch",
                    "job_public_id": job.public_id,
                    "method_key": desc.key,
                    "target": target,
                    "row_index": idx,
                },
            )
            continue
        prev = full_preds[idx]
        full_preds[idx] = exp[exp_key]
        sources[idx] = _source(exp)
        extra_info[idx] = build_extra_info(exp, target, prev, desc.display_name)

    # ── 6. Write output CSV ───────────────────────────────────────────────────
    df.insert(0, "Extra Info", extra_info)
    df.insert(0, "Source", sources)
    df.insert(0, output_col, full_preds)

    out_path = _output_path(job.public_id)
    df.to_csv(out_path, index=False)
    if target == "kcat" and include_similarity_columns:
        append_kcat_similarity_columns_to_output_csv(out_path, desc.key)

    # ── 7. Credit back empty rows and update job ──────────────────────────────
    empty = int((df[output_col] == "").sum()) + int(df[output_col].isna().sum())
    credit_back(_job_quota_subject(job), min(max(0, empty), int(job.requested_rows)))

    job.output_file.name = os.path.relpath(out_path, settings.MEDIA_ROOT)
    job.error_message = _build_skipped_message(skipped_reasons)
    job.save(update_fields=["output_file", "error_message"])


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
    """
    Run kcat and KM predictions in sequence and write a combined output.csv.

    Uses the stricter of the two methods' sequence-length limits.
    """
    sequences = df["Protein Sequence"].tolist()
    n_rows = len(sequences)

    # ── 1. Length handling — apply the stricter of the two limits ─────────────
    kcat_limit = min(SERVER_LIMIT, kcat_desc.max_seq_len)
    km_limit = min(SERVER_LIMIT, km_desc.max_seq_len)
    limit = min(kcat_limit, km_limit)

    if job.handle_long_sequences == "truncate":
        sequences_proc, valid_idx = truncate_sequences(sequences, limit)
    else:
        valid_idx = get_valid_indices(sequences, limit, mode="skip")
        sequences_proc = [sequences[i] for i in valid_idx]

    # Rows skipped by length handling
    skipped_reasons: dict[int, str] = {
        idx: "Sequence too long — row was excluded" for idx in set(range(n_rows)) - set(valid_idx)
    }

    # ── 2. Initialise result arrays ───────────────────────────────────────────
    kcat_preds: list = [""] * n_rows
    kcat_src: list[str] = [""] * n_rows
    kcat_extra: list[str] = [""] * n_rows
    km_preds: list = [""] * n_rows
    km_src: list[str] = [""] * n_rows
    km_extra: list[str] = [""] * n_rows

    eitlem_shared_cache = kcat_desc.key == "EITLEM" and km_desc.key == "EITLEM"
    omniesi_shared_cache = kcat_desc.key == "OmniESI" and km_desc.key == "OmniESI"

    # ── 3. kcat predictions ───────────────────────────────────────────────────
    if valid_idx:
        kcat_call_kwargs: dict = {}
        for col, kwarg_name in kcat_desc.col_to_kwarg.items():
            kcat_call_kwargs[kwarg_name] = [df[col].iloc[i] for i in valid_idx]
        kcat_call_kwargs.update(kcat_desc.target_kwargs.get("kcat", {}))
        if kcat_desc.key == "EITLEM":
            kcat_call_kwargs["cleanup_esm1v_embeddings"] = not eitlem_shared_cache
        if kcat_desc.key == "OmniESI":
            kcat_call_kwargs["cleanup_embeddings_after_run"] = not omniesi_shared_cache

        kcat_subset, kcat_bad = _invoke_method_prediction(
            desc=kcat_desc,
            sequences=sequences_proc,
            public_id=job.public_id,
            target="kcat",
            canonicalize_substrates=canonicalize_substrates,
            disable_gpu_precompute=disable_gpu_precompute,
            **kcat_call_kwargs,
        )
        for global_i, pred in zip(valid_idx, kcat_subset):
            value_out, source_out = _format_prediction_value_and_source(kcat_desc, "kcat", pred)
            kcat_preds[global_i] = value_out if value_out is not None else ""
            if value_out is not None:
                kcat_src[global_i] = source_out
        skipped_reasons.update(_map_subset_invalid_reasons(valid_idx, kcat_bad))

    # ── 4. KM predictions ─────────────────────────────────────────────────────
    if valid_idx:
        km_call_kwargs = {}
        for col, kwarg_name in km_desc.col_to_kwarg.items():
            km_call_kwargs[kwarg_name] = [df[col].iloc[i] for i in valid_idx]
        km_call_kwargs.update(km_desc.target_kwargs.get("Km", {}))
        if km_desc.key == "EITLEM":
            km_call_kwargs["cleanup_esm1v_embeddings"] = True
        if km_desc.key == "OmniESI":
            km_call_kwargs["cleanup_embeddings_after_run"] = True

        km_subset, km_bad = _invoke_method_prediction(
            desc=km_desc,
            sequences=sequences_proc,
            public_id=job.public_id,
            target="Km",
            canonicalize_substrates=canonicalize_substrates,
            disable_gpu_precompute=disable_gpu_precompute,
            **km_call_kwargs,
        )
        for global_i, pred in zip(valid_idx, km_subset):
            value_out, source_out = _format_prediction_value_and_source(km_desc, "Km", pred)
            km_preds[global_i] = value_out if value_out is not None else ""
            if value_out is not None:
                km_src[global_i] = source_out
        skipped_reasons.update(_map_subset_invalid_reasons(valid_idx, km_bad))

    for idx, reason in skipped_reasons.items():
        kcat_src[idx] = reason
        kcat_preds[idx] = ""
        km_src[idx] = reason
        km_preds[idx] = ""

    # ── 5. Experimental overwrites ────────────────────────────────────────────
    for exp in experimental_results:
        if not exp.get("found"):
            continue
        idx = exp["idx"]
        if exp.get("protein_sequence") != sequences[idx]:
            _log.warning(
                "Skipping experimental overwrite because protein sequence mismatched",
                extra={
                    "event": "prediction.experimental_overwrite_mismatch",
                    "job_public_id": job.public_id,
                    "method_key": f"{kcat_desc.key}/{km_desc.key}",
                    "target": "kcat/Km",
                    "row_index": idx,
                },
            )
            continue
        if "kcat_value" in exp:
            prev = kcat_preds[idx]
            kcat_preds[idx] = exp["kcat_value"]
            kcat_src[idx] = _source(exp)
            kcat_extra[idx] = build_extra_info(exp, "kcat", prev, kcat_desc.display_name)
        elif "km_value" in exp:
            prev = km_preds[idx]
            km_preds[idx] = exp["km_value"]
            km_src[idx] = _source(exp)
            km_extra[idx] = build_extra_info(exp, "Km", prev, km_desc.display_name)

    # ── 6. Assemble output DataFrame ──────────────────────────────────────────
    results_df = df.copy()
    results_df["kcat (1/s)"] = kcat_preds
    results_df["KM (mM)"] = km_preds
    results_df.insert(0, "Extra Info KM", km_extra)
    results_df.insert(0, "Source KM", km_src)
    results_df.insert(0, "Extra Info kcat", kcat_extra)
    results_df.insert(0, "Source kcat", kcat_src)

    preferred = [
        "kcat (1/s)",
        "Source kcat",
        "Extra Info kcat",
        "KM (mM)",
        "Source KM",
        "Extra Info KM",
    ]
    results_df = results_df[preferred + [c for c in results_df.columns if c not in preferred]]

    # ── 7. Write CSV, credit back, update job ─────────────────────────────────
    out_path = _output_path(job.public_id)
    results_df.to_csv(out_path, index=False)
    if include_similarity_columns:
        append_kcat_similarity_columns_to_output_csv(out_path, kcat_desc.key)

    fully_predicted = (
        (results_df["kcat (1/s)"] != "")
        & results_df["kcat (1/s)"].notna()
        & (results_df["KM (mM)"] != "")
        & results_df["KM (mM)"].notna()
    )
    processed = int(fully_predicted.sum())
    to_refund = max(0, int(job.requested_rows) - processed)
    if to_refund > 0:
        credit_back(_job_quota_subject(job), to_refund)

    Job.objects.filter(pk=job.pk).update(
        output_file=os.path.relpath(out_path, settings.MEDIA_ROOT),
        error_message=_build_skipped_message(skipped_reasons),
    )


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
    """
    Run one or more prediction targets and write a combined output.csv.

    Targets are executed in canonical order and each contributes a column bundle:
    prediction values + source + extra info.
    """
    sequences = df["Protein Sequence"].tolist()
    n_rows = len(sequences)

    limits = [min(SERVER_LIMIT, desc.max_seq_len) for desc in desc_by_target.values()]
    limit = min(limits) if limits else SERVER_LIMIT

    if job.handle_long_sequences == "truncate":
        sequences_proc, valid_idx = truncate_sequences(sequences, limit)
    else:
        valid_idx = get_valid_indices(sequences, limit, mode="skip")
        sequences_proc = [sequences[i] for i in valid_idx]

    skipped_reasons: dict[int, str] = {
        idx: "Sequence too long — row was excluded" for idx in set(range(n_rows)) - set(valid_idx)
    }

    target_results: dict[str, dict] = {}
    for target in targets:
        desc = desc_by_target[target]
        target_results[target] = {
            "desc": desc,
            "preds": [""] * n_rows,
            "sources": [""] * n_rows,
            "extra": [""] * n_rows,
            "output_col": desc.output_cols[target],
        }

    eitlem_targets = [target for target in targets if desc_by_target[target].key == "EITLEM"]
    last_eitlem_target = eitlem_targets[-1] if eitlem_targets else None
    omniesi_targets = [target for target in targets if desc_by_target[target].key == "OmniESI"]
    last_omniesi_target = omniesi_targets[-1] if omniesi_targets else None

    for target in targets:
        desc = desc_by_target[target]
        results = target_results[target]
        mark_stage_running(job.public_id, target, desc.key)

        if not valid_idx:
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
            mark_stage_completed(job.public_id, target, desc.key)
            continue

        call_kwargs = {}
        for col, kwarg_name in desc.col_to_kwarg.items():
            call_kwargs[kwarg_name] = [df[col].iloc[i] for i in valid_idx]
        call_kwargs.update(desc.target_kwargs.get(target, {}))
        if desc.key == "EITLEM":
            call_kwargs["cleanup_esm1v_embeddings"] = target == last_eitlem_target
        if desc.key == "OmniESI":
            call_kwargs["cleanup_embeddings_after_run"] = target == last_omniesi_target

        try:
            pred_subset, invalid_subset = _invoke_method_prediction(
                desc=desc,
                sequences=sequences_proc,
                public_id=job.public_id,
                target=target,
                canonicalize_substrates=canonicalize_substrates,
                disable_gpu_precompute=disable_gpu_precompute,
                **call_kwargs,
            )
        except Exception as exc:
            mark_stage_failed(job.public_id, target, desc.key, message=str(exc))
            raise

        for global_i, pred in zip(valid_idx, pred_subset):
            value_out, source_out = _format_prediction_value_and_source(desc, target, pred)
            results["preds"][global_i] = value_out if value_out is not None else ""
            results["sources"][global_i] = source_out
        skipped_reasons.update(_map_subset_invalid_reasons(valid_idx, invalid_subset))
        mark_stage_completed(job.public_id, target, desc.key)

    # Experimental overrides are only available for kcat and Km.
    for target, exp_key in (("kcat", "kcat_value"), ("Km", "km_value")):
        if target not in target_results:
            continue

        for exp in experimental_results.get(target, []):
            if not exp.get("found"):
                continue

            idx = exp.get("idx")
            if not isinstance(idx, int) or idx < 0 or idx >= n_rows:
                continue

            if exp.get("protein_sequence") != sequences[idx]:
                _log.warning(
                    "Skipping experimental overwrite because protein sequence mismatched",
                    extra={
                        "event": "prediction.experimental_overwrite_mismatch",
                        "job_public_id": job.public_id,
                        "method_key": target_results[target]["desc"].key,
                        "target": target,
                        "row_index": idx,
                    },
                )
                continue

            if exp_key not in exp:
                continue

            prev = target_results[target]["preds"][idx]
            target_results[target]["preds"][idx] = exp[exp_key]
            target_results[target]["sources"][idx] = _source(exp)
            target_results[target]["extra"][idx] = build_extra_info(
                exp,
                target,
                prev,
                target_results[target]["desc"].display_name,
            )

    for idx, reason in skipped_reasons.items():
        for result in target_results.values():
            result["sources"][idx] = reason
            result["preds"][idx] = ""

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
        preferred_cols + [c for c in results_df.columns if c not in preferred_cols]
    ]

    out_path = _output_path(job.public_id)
    results_df.to_csv(out_path, index=False)
    if include_similarity_columns and "kcat" in targets and "kcat" in desc_by_target:
        append_kcat_similarity_columns_to_output_csv(
            out_path,
            desc_by_target["kcat"].key,
        )

    fully_predicted = pd.Series(True, index=results_df.index)
    for target in targets:
        pred_col = target_results[target]["output_col"]
        fully_predicted = (
            fully_predicted & (results_df[pred_col] != "") & results_df[pred_col].notna()
        )

    processed = int(fully_predicted.sum())
    to_refund = max(0, int(job.requested_rows) - processed)
    if to_refund > 0:
        credit_back(_job_quota_subject(job), to_refund)

    Job.objects.filter(pk=job.pk).update(
        output_file=os.path.relpath(out_path, settings.MEDIA_ROOT),
        error_message=_build_skipped_message(skipped_reasons),
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
                return preds, invalid_result
            return preds, {idx: "Prediction could not be made" for idx in (invalid_result or [])}

        if desc.subprocess is not None:
            return run_generic_subprocess_prediction(
                desc=desc,
                sequences=sequences,
                public_id=public_id,
                target=target,
                **call_kwargs,
            )

    raise PredictionError(f"{desc.display_name} is not configured with a prediction engine.")


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
