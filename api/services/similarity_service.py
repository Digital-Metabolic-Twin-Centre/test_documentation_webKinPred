"""
Similarity analysis service that orchestrates the similarity workflow.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd

from api.services.progress_service import push_line
from api.utils.similarity_config import SIMILARITY_DATASETS, TARGET_DBS
from api.utils.similarity_utils import (
    TMP_DIR,
    calculate_average_similarity,
    calculate_identity_histogram,
    cleanup_temporary_files,
    create_fasta_file,
    create_mmseqs_database,
    create_unique_sequence_mapping,
    extract_protein_sequences_from_csv,
    map_results_to_original_sequences,
    parse_mmseqs_results,
    parse_mmseqs_results_raw,
    run_mmseqs_search,
    _mmseqs_cmd,
)

_log = logging.getLogger(__name__)


def _find_merged_db() -> tuple[str, str] | tuple[None, None]:
    """Return (merged_db_path, membership_json_path) if both exist, else (None, None)."""
    datasets = SIMILARITY_DATASETS or {}
    any_dataset = next(iter(datasets.values()), {})
    any_target_db = any_dataset.get("target_db", "")
    if not any_target_db:
        return None, None
    dbs_dir = os.path.dirname(any_target_db)
    merged_db = os.path.join(dbs_dir, "targetdb_merged")
    membership_path = f"{merged_db}_membership.json"
    if (os.path.exists(merged_db) or os.path.exists(f"{merged_db}.dbtype")) and os.path.exists(
        membership_path
    ):
        return merged_db, membership_path
    return None, None


def _load_membership(membership_path: str) -> Dict[str, List[str]]:
    with open(membership_path) as f:
        return json.load(f)


def analyze_sequence_similarity(csv_file, session_id: str = "default") -> Dict[str, Any]:
    """
    Analyze sequence similarity against target databases.

    Args:
        csv_file: Uploaded CSV file containing protein sequences
        session_id: Session ID for logging

    Returns:
        Dictionary containing similarity analysis results

    Raises:
        ValueError: If CSV is invalid or contains no sequences
        Exception: If analysis fails
    """
    input_sequences = extract_protein_sequences_from_csv(csv_file)
    unique_sequences, seq_to_unique_id = create_unique_sequence_mapping(input_sequences)
    query_file_path = create_fasta_file(unique_sequences, seq_to_unique_id)
    temp_files_to_cleanup = [query_file_path]

    try:
        query_db, temp_query_dir = create_mmseqs_database(query_file_path, session_id)
        temp_files_to_cleanup.append(temp_query_dir)

        method_histograms = {}

        datasets = SIMILARITY_DATASETS or {
            label: {"label": label, "target_db": path} for label, path in TARGET_DBS.items()
        }

        # Filter to datasets whose DBs are present on disk
        active_datasets: Dict[str, Any] = {}
        for _dataset_key, dataset in datasets.items():
            label = dataset.get("label") or _dataset_key
            target_db = dataset.get("target_db")
            if not target_db:
                push_line(session_id, f"[WARN] Skipping dataset '{label}' (missing target_db path)")
                continue
            if not (os.path.exists(target_db) or os.path.exists(f"{target_db}.dbtype")):
                push_line(session_id, f"[WARN] Skipping dataset '{label}' (DB files not found)")
                continue
            active_datasets[label] = dataset

        merged_db, membership_path = _find_merged_db()

        if merged_db and active_datasets:
            membership = _load_membership(membership_path)

            # Build reverse index: label -> set of mseq IDs
            label_to_targets: Dict[str, set] = {}
            for mseq_id, labels in membership.items():
                for lbl in labels:
                    label_to_targets.setdefault(lbl, set()).add(mseq_id)

            covered = [lbl for lbl in active_datasets if lbl in label_to_targets]
            uncovered = [lbl for lbl in active_datasets if lbl not in label_to_targets]

            if uncovered:
                push_line(
                    session_id,
                    f"[WARN] Not in merged DB, searching individually: {', '.join(uncovered)}",
                )

            if covered:
                # Single search against the merged DB; no --max-seqs truncation so
                # every hit that would be found in any individual search is present.
                push_line(session_id, "==> Running single merged search")
                result_file = run_mmseqs_search(
                    query_db, merged_db, "merged", session_id, max_seqs=len(membership)
                )
                temp_files_to_cleanup.append(os.path.dirname(result_file))

                raw_hits = parse_mmseqs_results_raw(result_file)

                for label in covered:
                    push_line(session_id, f"==> Processing DB: {label}")
                    target_ids = label_to_targets[label]

                    # Filter hits to this DB's sequences; 0.0 for queries with no hits
                    identity_lists: Dict[str, List[float]] = {}
                    for query_id, hits in raw_hits.items():
                        filtered = [p for t, p in hits if t in target_ids]
                        if filtered:
                            identity_lists[query_id] = filtered
                    for unique_id in seq_to_unique_id.values():
                        if unique_id not in identity_lists:
                            identity_lists[unique_id] = [0.0]

                    unique_max = {k: max(v) for k, v in identity_lists.items()}
                    unique_mean = {k: sum(v) / len(v) for k, v in identity_lists.items()}

                    query_to_max, query_to_mean = map_results_to_original_sequences(
                        unique_max, unique_mean, input_sequences, seq_to_unique_id
                    )
                    histogram_max_counts, histogram_max_perc = calculate_identity_histogram(
                        query_to_max
                    )
                    histogram_mean_counts, histogram_mean_perc = calculate_identity_histogram(
                        query_to_mean
                    )
                    method_histograms[label] = {
                        "histogram_max": histogram_max_perc,
                        "histogram_mean": histogram_mean_perc,
                        "average_max_similarity": calculate_average_similarity(query_to_max),
                        "average_mean_similarity": calculate_average_similarity(query_to_mean),
                        "count_max": histogram_max_counts,
                        "count_mean": histogram_mean_counts,
                    }
                    push_line(session_id, f"--> [{label}] Aggregated {len(query_to_max)} sequences")

            # Individual fallback for any dataset not covered by the merged DB
            for label in uncovered:
                target_db = active_datasets[label].get("target_db")
                push_line(session_id, f"==> Processing DB: {label}")
                method_histograms[label] = analyze_similarity_for_method(
                    query_db, target_db, query_file_path, label,
                    input_sequences, seq_to_unique_id, session_id,
                )

        else:
            # No merged DB available — original per-DB loop
            for label, dataset in active_datasets.items():
                target_db = dataset.get("target_db")
                push_line(session_id, f"==> Processing DB: {label}")
                method_histograms[label] = analyze_similarity_for_method(
                    query_db, target_db, query_file_path, label,
                    input_sequences, seq_to_unique_id, session_id,
                )

        if not method_histograms:
            raise ValueError(
                "No similarity datasets are available. "
                "Add a dataset in similarity config and build its MMseqs DB."
            )

        return method_histograms

    finally:
        cleanup_temporary_files(*temp_files_to_cleanup)


def analyze_similarity_for_method(
    query_db: str,
    target_db: str,
    query_file_path: str,
    method_name: str,
    original_sequences: List[str],
    seq_to_unique_id: Dict[str, str],
    session_id: str,
) -> Dict[str, Any]:
    """
    Analyze similarity for a specific method/database.

    Args:
        query_db: Path to query database
        target_db: Path to target database
        query_file_path: Path to original FASTA file
        method_name: Name of the method
        original_sequences: Original sequence list
        seq_to_unique_id: Sequence to unique ID mapping
        session_id: Session ID for logging

    Returns:
        Dictionary containing method-specific results
    """
    result_file = None

    try:
        # Run MMseqs2 search
        result_file = run_mmseqs_search(query_db, target_db, method_name, session_id)

        # Parse results to get identity scores
        unique_max_identity, unique_mean_identity = parse_mmseqs_results(
            result_file, query_file_path
        )

        # Map results back to original sequences
        query_to_max, query_to_mean = map_results_to_original_sequences(
            unique_max_identity, unique_mean_identity, original_sequences, seq_to_unique_id
        )

        # Calculate histograms
        histogram_max_counts, histogram_max_perc = calculate_identity_histogram(query_to_max)
        histogram_mean_counts, histogram_mean_perc = calculate_identity_histogram(query_to_mean)

        # Calculate averages
        average_max_similarity = calculate_average_similarity(query_to_max)
        average_mean_similarity = calculate_average_similarity(query_to_mean)

        push_line(session_id, f"--> [{method_name}] Aggregated {len(query_to_max)} sequences")

        return {
            "histogram_max": histogram_max_perc,
            "histogram_mean": histogram_mean_perc,
            "average_max_similarity": average_max_similarity,
            "average_mean_similarity": average_mean_similarity,
            "count_max": histogram_max_counts,
            "count_mean": histogram_mean_counts,
        }

    finally:
        # Clean up result file and its parent directory
        if result_file and os.path.exists(result_file):
            result_dir = os.path.dirname(result_file)
            cleanup_temporary_files(result_dir)


def _similarity_column_names(method_key: str) -> tuple[str, str]:
    return (
        f"mean similarity to {method_key} training data",
        f"max similarity to {method_key} training data",
    )


def _resolve_similarity_dataset_for_method(method_key: str) -> tuple[Optional[str], Optional[str]]:
    datasets = SIMILARITY_DATASETS or {
        label: {"label": label, "target_db": path} for label, path in TARGET_DBS.items()
    }

    for dataset_key, dataset in datasets.items():
        dataset_methods = set(dataset.get("method_keys") or [])
        if method_key in dataset_methods:
            label = dataset.get("label") or dataset_key
            return label, dataset.get("target_db")

    return None, None


def _run_mmseqs_command(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        output = (proc.stdout or "").strip()
        if output:
            output = output.splitlines()[-1]
            raise RuntimeError(f"MMseqs command failed: {' '.join(cmd)} :: {output}")
        raise RuntimeError(f"MMseqs command failed: {' '.join(cmd)}")


def _write_blank_similarity_columns(
    output_csv_path: str,
    mean_col: str,
    max_col: str,
) -> None:
    try:
        df = pd.read_csv(output_csv_path)
    except Exception as exc:
        _log.warning(
            "Could not read output CSV to add blank similarity columns",
            extra={
                "event": "similarity.output_blank_columns_read_failed",
                "output_csv_path": output_csv_path,
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )
        return

    df[mean_col] = ""
    df[max_col] = ""
    try:
        df.to_csv(output_csv_path, index=False)
    except Exception as exc:
        _log.warning(
            "Could not write blank similarity columns to output CSV",
            extra={
                "event": "similarity.output_blank_columns_write_failed",
                "output_csv_path": output_csv_path,
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )


def append_kcat_similarity_columns_to_output_csv(
    output_csv_path: str,
    kcat_method_key: str,
) -> None:
    """
    Best-effort enrichment for completed kcat jobs.

    Adds two per-row columns to output.csv:
      - mean similarity to {method} training data
      - max similarity to {method} training data

    On any error, both columns are still created with blank values.
    """
    mean_col, max_col = _similarity_column_names(kcat_method_key)
    temp_files_to_cleanup: list[str] = []

    try:
        df = pd.read_csv(output_csv_path)
        if "Protein Sequence" not in df.columns:
            raise ValueError('Output CSV is missing required "Protein Sequence" column')

        dataset_label, target_db = _resolve_similarity_dataset_for_method(kcat_method_key)
        if not target_db:
            raise ValueError(f"No similarity dataset is configured for method '{kcat_method_key}'")
        if not (os.path.exists(target_db) or os.path.exists(f"{target_db}.dbtype")):
            raise FileNotFoundError(
                f"Similarity DB for '{dataset_label or kcat_method_key}' not found at {target_db}"
            )

        raw_sequences = [str(seq).strip() for seq in df["Protein Sequence"].fillna("").tolist()]
        unique_sequences: list[str] = []
        seq_to_unique_id: dict[str, str] = {}
        for seq in raw_sequences:
            if not seq:
                continue
            if seq not in seq_to_unique_id:
                seq_to_unique_id[seq] = f"useq{len(seq_to_unique_id)}"
                unique_sequences.append(seq)

        if not unique_sequences:
            raise ValueError("No non-empty protein sequences available for similarity analysis")

        query_fasta_path = create_fasta_file(unique_sequences, seq_to_unique_id)
        temp_files_to_cleanup.append(query_fasta_path)

        query_dir = tempfile.mkdtemp(dir=TMP_DIR)
        temp_files_to_cleanup.append(query_dir)
        query_db = os.path.join(query_dir, "queryDB")
        _run_mmseqs_command(_mmseqs_cmd("createdb", query_fasta_path, query_db))

        result_dir = tempfile.mkdtemp(dir=TMP_DIR)
        temp_files_to_cleanup.append(result_dir)
        result_db = os.path.join(result_dir, "resultDB")
        result_file = os.path.join(result_dir, "result.m8")

        _run_mmseqs_command(
            _mmseqs_cmd(
                "search",
                query_db,
                target_db,
                result_db,
                result_dir,
                "--max-seqs",
                "1000",
                "-s",
                "7.5",
                "-e",
                "0.001",
                "-v",
                "0",
            )
        )
        _run_mmseqs_command(
            _mmseqs_cmd(
                "convertalis",
                query_db,
                target_db,
                result_db,
                result_file,
                "--format-output",
                "query,target,pident",
            )
        )

        unique_max_identity, unique_mean_identity = parse_mmseqs_results(
            result_file,
            query_fasta_path,
        )

        sequence_to_max = {
            seq: round(float(unique_max_identity.get(unique_id, 0.0)), 2)
            for seq, unique_id in seq_to_unique_id.items()
        }
        sequence_to_mean = {
            seq: round(float(unique_mean_identity.get(unique_id, 0.0)), 2)
            for seq, unique_id in seq_to_unique_id.items()
        }

        mean_values: list[float | str] = []
        max_values: list[float | str] = []
        for seq in raw_sequences:
            if not seq:
                mean_values.append("")
                max_values.append("")
                continue
            mean_values.append(sequence_to_mean.get(seq, 0.0))
            max_values.append(sequence_to_max.get(seq, 0.0))

        df[mean_col] = mean_values
        df[max_col] = max_values
        df.to_csv(output_csv_path, index=False)

    except Exception as exc:
        _log.warning(
            "Could not enrich output CSV with kcat similarity columns",
            extra={
                "event": "similarity.output_enrichment_failed",
                "output_csv_path": output_csv_path,
                "method_key": kcat_method_key,
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )
        _write_blank_similarity_columns(output_csv_path, mean_col, max_col)
    finally:
        cleanup_temporary_files(*temp_files_to_cleanup)
