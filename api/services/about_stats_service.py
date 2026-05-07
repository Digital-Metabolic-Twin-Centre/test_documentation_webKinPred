"""
About-page usage metrics aggregation.

Computes all-time counts used by the frontend About page. Metrics are derived
from completed jobs and, for row-level counters, from available output CSV
files only.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from typing import Any

from django.utils import timezone

from api.models import Job

KCAT_COL = "kcat (1/s)"
KM_COL = "KM (mM)"
KCAT_KM_COL = "kcat/Km (1/(s*mM))"
PROTEIN_SEQUENCE_COL = "Protein Sequence"

CACHE_TTL_SECONDS = 300

_cache_lock = threading.Lock()
_cache_payload: dict[str, Any] | None = None
_cache_ts = 0.0


def _has_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip() != ""


def _iter_available_completed_output_paths() -> tuple[int, list[str]]:
    completed_jobs = Job.objects.filter(status="Completed").only("output_file")
    jobs_completed = int(completed_jobs.count())

    output_paths: list[str] = []
    for job in completed_jobs.iterator(chunk_size=200):
        output_file = getattr(job, "output_file", None)
        if not output_file:
            continue
        try:
            file_path = output_file.path
        except Exception:
            continue
        if file_path and os.path.isfile(file_path):
            output_paths.append(file_path)

    return jobs_completed, output_paths


def _compute_about_stats_payload(now_iso: str) -> dict[str, Any]:
    jobs_completed, output_paths = _iter_available_completed_output_paths()

    reactions_completed = 0
    kcat_predictions_completed = 0
    km_predictions_completed = 0
    kcat_km_predictions_completed = 0
    unique_sequences: set[str] = set()

    for file_path in output_paths:
        try:
            with open(file_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    kcat_has = _has_non_empty_value(row.get(KCAT_COL))
                    km_has = _has_non_empty_value(row.get(KM_COL))
                    kcat_km_has = _has_non_empty_value(row.get(KCAT_KM_COL))

                    if kcat_has:
                        kcat_predictions_completed += 1
                    if km_has:
                        km_predictions_completed += 1
                    if kcat_km_has:
                        kcat_km_predictions_completed += 1
                    if kcat_has or km_has or kcat_km_has:
                        reactions_completed += 1

                    sequence = str(row.get(PROTEIN_SEQUENCE_COL, "")).strip()
                    if sequence:
                        unique_sequences.add(sequence)
        except Exception:
            # Policy choice: silently skip missing/unreadable/invalid output files.
            continue

    return {
        "scope": "all_time",
        "generated_at": now_iso,
        "jobs_completed": jobs_completed,
        "reactions_completed": reactions_completed,
        "unique_protein_sequences": len(unique_sequences),
        "kcat_predictions_completed": kcat_predictions_completed,
        "km_predictions_completed": km_predictions_completed,
        "kcat_km_predictions_completed": kcat_km_predictions_completed,
    }


def get_about_stats(*, force_refresh: bool = False) -> dict[str, Any]:
    """Return cached About metrics, refreshing them when TTL expires."""
    global _cache_payload, _cache_ts

    now_monotonic = time.monotonic()
    if not force_refresh:
        with _cache_lock:
            if _cache_payload is not None and (now_monotonic - _cache_ts) < CACHE_TTL_SECONDS:
                return dict(_cache_payload)

    payload = _compute_about_stats_payload(timezone.now().isoformat())

    with _cache_lock:
        _cache_payload = dict(payload)
        _cache_ts = now_monotonic

    return dict(payload)
