#!/usr/bin/env python3
"""
Build a merged MMseqs2 target database from all configured similarity datasets.

Reads sequences from existing MMseqs2 databases (not the original FASTA files),
deduplicates across all databases, and creates a single merged database alongside
a membership JSON mapping each merged sequence ID to its source database label(s).

The membership JSON is used at query time to split hits back into per-database
results without re-running MMseqs2 multiple times.

Examples
--------
Build merged DB (output: fastas/dbs/targetdb_merged):
    python tools/build_merged_similarity_db.py

Custom output name:
    python tools/build_merged_similarity_db.py --output-name targetdb_all

Preview which databases would be merged without running:
    python tools/build_merged_similarity_db.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_conda_path() -> str | None:
    env_override = os.environ.get("WEBKINPRED_CONDA_PATH")
    if env_override:
        return env_override
    default = Path.home() / "anaconda3" / "bin" / "conda"
    return str(default) if default.exists() else None


def _load_similarity_registry() -> dict[str, dict]:
    registry_path = REPO_ROOT / "webKinPred" / "similarity_dataset_registry.py"
    spec = importlib.util.spec_from_file_location("similarity_dataset_registry", registry_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load registry file: {registry_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = getattr(module, "SIMILARITY_DATASET_REGISTRY", {})
    if not isinstance(registry, dict):
        raise RuntimeError("SIMILARITY_DATASET_REGISTRY must be a dict.")
    return registry


CONDA_PATH = _default_conda_path()
FASTAS_DIR = Path(os.environ.get("WEBKINPRED_FASTAS_DIR", str(REPO_ROOT / "fastas"))).resolve()
DBS_DIR = FASTAS_DIR / "dbs"


def _mmseqs_cmd(*args: str) -> list[str]:
    if CONDA_PATH:
        return [CONDA_PATH, "run", "-n", "mmseqs2_env", "mmseqs", *args]
    return ["mmseqs", *args]


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _db_exists(db_path: str) -> bool:
    return os.path.exists(db_path) or os.path.exists(f"{db_path}.dbtype")


def _extract_sequences_from_db(db_path: str, label: str, tmp_dir: str) -> list[str]:
    """
    Dump an MMseqs2 sequence DB to a temporary FASTA via convert2fasta and return
    all non-empty sequences as a list of amino-acid strings.
    """
    safe_label = label.replace("/", "_").replace(" ", "_")
    fasta_out = os.path.join(tmp_dir, f"{safe_label}.fasta")
    _run(_mmseqs_cmd("convert2fasta", db_path, fasta_out))

    sequences: list[str] = []
    current_parts: list[str] = []

    with open(fasta_out) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_parts:
                    seq = "".join(current_parts)
                    if seq:
                        sequences.append(seq)
                    current_parts = []
            else:
                current_parts.append(line)
        if current_parts:
            seq = "".join(current_parts)
            if seq:
                sequences.append(seq)

    return sequences


def _collect_available_dbs(registry: dict[str, dict]) -> list[tuple[str, str]]:
    available: list[tuple[str, str]] = []
    for label, meta in registry.items():
        db_name = meta.get("db_name")
        if not db_name:
            print(f"[WARN] Skipping '{label}': no db_name in registry.")
            continue
        db_path = str(DBS_DIR / db_name)
        if not _db_exists(db_path):
            print(f"[WARN] Skipping '{label}': DB not found at {db_path}")
            continue
        available.append((label, db_path))
    return available


def build_merged_db(output_name: str, dry_run: bool) -> int:
    registry = _load_similarity_registry()
    available = _collect_available_dbs(registry)

    if not available:
        print(
            "ERROR: No existing MMseqs2 databases found. Build individual DBs first with:\n"
            "  python tools/build_similarity_dbs.py --all",
            file=sys.stderr,
        )
        return 1

    print(f"\nDatabases to merge ({len(available)}):")
    for label, db_path in available:
        print(f"  - {label}")
        print(f"    {db_path}")

    if dry_run:
        print("\n[DRY RUN] Exiting without building.")
        return 0

    # sequence_string -> [label, label, ...]  (preserves insertion order for labels)
    membership_by_seq: dict[str, list[str]] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for label, db_path in available:
            print(f"\n==> Extracting: {label}")
            sequences = _extract_sequences_from_db(db_path, label, tmp_dir)
            print(f"    {len(sequences)} sequences read")
            for seq in sequences:
                membership_by_seq.setdefault(seq, []).append(label)

        unique_sequences = list(membership_by_seq.keys())
        n_cross_db = sum(1 for v in membership_by_seq.values() if len(v) > 1)
        print(f"\nUnique sequences:          {len(unique_sequences)}")
        if n_cross_db:
            print(f"Shared across >1 database: {n_cross_db}")

        # Assign new sequential IDs and write merged FASTA
        merged_id_to_labels: dict[str, list[str]] = {}
        merged_fasta = os.path.join(tmp_dir, "merged.fasta")

        with open(merged_fasta, "w") as fh:
            for idx, seq in enumerate(unique_sequences):
                seq_id = f"mseq{idx}"
                merged_id_to_labels[seq_id] = membership_by_seq[seq]
                fh.write(f">{seq_id}\n{seq}\n")

        # Build the merged MMseqs2 DB
        os.makedirs(DBS_DIR, exist_ok=True)
        output_db = str(DBS_DIR / output_name)

        print(f"\n==> Building merged MMseqs2 DB: {output_db}")
        _run(_mmseqs_cmd("createdb", merged_fasta, output_db))

        # Write membership sidecar JSON next to the DB
        membership_path = f"{output_db}_membership.json"
        with open(membership_path, "w") as fh:
            json.dump(merged_id_to_labels, fh)
        print(f"[OK] Membership JSON: {membership_path}")

    print(f"\n{'='*50}")
    print(f"Merged DB:          {output_db}")
    print(f"Membership JSON:    {membership_path}")
    print(f"Databases merged:   {len(available)}")
    print(f"Unique sequences:   {len(unique_sequences)}")
    print()
    label_counts: dict[str, int] = {}
    for labels in merged_id_to_labels.values():
        for lbl in labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    for label, count in label_counts.items():
        print(f"  {label}: {count} sequences")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a merged MMseqs2 database from all configured similarity datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-name",
        default="targetdb_merged",
        metavar="NAME",
        help="Filename for the output DB inside fastas/dbs/ (default: targetdb_merged).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List databases that would be merged without running anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build_merged_db(args.output_name, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
