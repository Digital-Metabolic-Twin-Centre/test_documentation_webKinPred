"""EITLEM-Kinetics prediction script.

ESM1v per-residue representations (seq_len × 1280, float32) are used
exactly as the original EITLEM model requires — the full matrix is passed
as `pro_emb` to the GNN, which aggregates it internally.

Embedding resolution order for each sequence
─────────────────────────────────────────────
1. Read from  <EITLEM_MEDIA_PATH>/sequence_info/esm1v/<seq_id>.npy
   (pre-computed by the GPU embedding step when a GPU server is available).
2. If absent, compute on CPU using the local ESM1v checkpoint and save the
   full per-residue matrix to the same path.

After all predictions are complete the script deletes every esm1v file it
touched (GPU-precomputed and CPU-computed alike), so the directory does not
accumulate stale data between jobs.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from torch_geometric.data import Batch, Data

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import (
    SpoolAsyncCommitter,
    remove_manifest_entries,
    resolve_missing_ids,
)
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor

# ── Path resolution ────────────────────────────────────────────────────────────

_MEDIA_PATH = os.environ.get("EITLEM_MEDIA_PATH", "/home/saleh/webKinPred/media")
_TOOLS_PATH = os.environ.get("EITLEM_TOOLS_PATH", "/home/saleh/webKinPred/tools")
_DELETE_EMBEDDINGS_AFTER_RUN = (
    str(os.environ.get("EITLEM_DELETE_EMBEDDINGS_AFTER_RUN", "1")).strip().lower()
    in {"1", "true", "yes", "on"}
)

ESM_EMB_DIR = Path(_MEDIA_PATH) / "sequence_info" / "esm1v"

SEQMAP_CLI = Path(_TOOLS_PATH) / "seqmap" / "main.py"
SEQMAP_DB  = Path(_MEDIA_PATH) / "sequence_info" / "seqmap.sqlite3"

# Use the running interpreter in Docker; fall back to the local venv otherwise.
SEQMAP_PY = sys.executable if os.environ.get("EITLEM_MEDIA_PATH") else "/home/saleh/webKinPredEnv/bin/python"

if os.environ.get("EITLEM_MEDIA_PATH"):
    _ESM_MODEL_PATH = "/app/models/EITLEM/Weights/esm1v/esm1v_t33_650M_UR90S_1.pt"
    _WEIGHT_PATHS = {
        "KCAT": "/app/models/EITLEM/Weights/KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787",
        "KM":   "/app/models/EITLEM/Weights/KM/iter8_trainR2_0.9303_devR2_0.7163_RMSE_0.6960_MAE_0.4802",
    }
else:
    _ESM_MODEL_PATH = "/home/saleh/webKinPred/models/EITLEM/Weights/esm1v/esm1v_t33_650M_UR90S_1.pt"
    _WEIGHT_PATHS = {
        "KCAT": "/home/saleh/webKinPred/models/EITLEM/Weights/KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787",
        "KM":   "/home/saleh/webKinPred/models/EITLEM/Weights/KM/iter8_trainR2_0.9303_devR2_0.7163_RMSE_0.6960_MAE_0.4802",
    }

# ── Seqmap ────────────────────────────────────────────────────────────────────

def resolve_seq_ids_via_cli(sequences: list[str]) -> list[str]:
    payload = "\n".join(sequences) + "\n"
    cmd = [SEQMAP_PY, str(SEQMAP_CLI), "--db", str(SEQMAP_DB), "batch-get-or-create", "--stdin"]
    proc = subprocess.run(cmd, input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"seqmap CLI failed (rc={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    ids = proc.stdout.strip().splitlines()
    if len(ids) != len(sequences):
        raise RuntimeError(f"seqmap returned {len(ids)} ids for {len(sequences)} sequences")
    return ids


# ── ESM1v (lazy singleton) ────────────────────────────────────────────────────

_esm_model = None
_alphabet = None
_batch_converter = None


def _load_esm_once() -> None:
    global _esm_model, _alphabet, _batch_converter
    if _esm_model is not None:
        return
    import esm as esm_lib
    _esm_model, _alphabet = esm_lib.pretrained.load_model_and_alphabet_local(_ESM_MODEL_PATH)
    _batch_converter = _alphabet.get_batch_converter()
    _esm_model.eval()


def _compute_residue_repr(sequence: str) -> np.ndarray:
    """Return ESM1v layer-33 per-residue representations on CPU.

    Returns an ndarray of shape (seq_len, 1280) — the full matrix that the
    EITLEM GNN receives as `pro_emb`.
    """
    _load_esm_once()

    # ESM1v maximum context is 1024 tokens (incl. <cls>/<eos>) → seq ≤ 1022.
    if len(sequence) > 1022:
        sequence = sequence[:500] + sequence[-500:]

    _, _, batch_tokens = _batch_converter([("protein", sequence)])
    batch_lens = (batch_tokens != _alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = _esm_model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_repr = results["representations"][33]
    tokens_len = batch_lens[0]
    # Strip <cls> (position 0) and <eos> (position tokens_len-1).
    return token_repr[0, 1 : tokens_len - 1].cpu().numpy()  # (seq_len, 1280)


# ── ESM1v cache (esm1v/) ──────────────────────────────────────────────────────

def _emb_path(seq_id: str) -> Path:
    return ESM_EMB_DIR / f"{seq_id}.npy"


def _free_esm() -> None:
    global _esm_model, _alphabet, _batch_converter
    _esm_model = None
    _alphabet = None
    _batch_converter = None


def _cleanup_embeddings(seq_ids: list[str]) -> None:
    cleanup_seq_ids = {str(seq_id).strip() for seq_id in seq_ids if str(seq_id).strip()}
    if not cleanup_seq_ids:
        return

    for seq_id in cleanup_seq_ids:
        try:
            _emb_path(seq_id).unlink(missing_ok=True)
        except OSError:
            pass

    try:
        remove_manifest_entries(ESM_EMB_DIR, cleanup_seq_ids)
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python eitlem_prediction_script.py input.csv output.csv KCAT|KM")
        sys.exit(1)

    input_file    = sys.argv[1]
    output_file   = sys.argv[2]
    kinetics_type = sys.argv[3].upper()

    if kinetics_type not in _WEIGHT_PATHS:
        print(f"Invalid kinetics type: {kinetics_type}")
        sys.exit(1)

    df_input   = pd.read_csv(input_file)
    sequences  = df_input["Protein Sequence"].tolist()
    substrates = df_input["Substrate SMILES"].tolist()

    seq_ids = resolve_seq_ids_via_cli(sequences)
    run_completed = False

    try:
        # ── Phase 1: ensure all ESM1v embeddings are on disk ─────────────────
        # GPU case: files already written by the GPU server before this script runs.
        # CPU fallback: compute them now with ESM1v loaded once, then free the model
        # so it doesn't compete with the GNN for RAM during Phase 2.
        ESM_EMB_DIR.mkdir(parents=True, exist_ok=True)
        missing_ids, _ready_ids = resolve_missing_ids(
            seq_ids,
            cache_dir=ESM_EMB_DIR,
            suffix=".npy",
        )
        if missing_ids:
            seq_by_id: dict[str, str] = {}
            for seq_id, sequence in zip(seq_ids, sequences):
                seq_by_id.setdefault(seq_id, sequence)
            async_workers = max(
                1,
                int(
                    os.environ.get(
                        "EITLEM_CACHE_ASYNC_WORKERS",
                        os.environ.get("GPU_EMBED_CACHE_ASYNC_WORKERS", "4"),
                    )
                ),
            )
            spool_dir = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_DIR", "/dev/shm/webkinpred-gpu-cache"))
            spool_fallback = Path(os.environ.get("GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR", "/tmp/webkinpred-gpu-cache"))
            with SpoolAsyncCommitter(
                max_workers=async_workers,
                spool_dir=spool_dir,
                spool_fallback_dir=spool_fallback,
            ) as committer:
                for seq_id in missing_ids:
                    sequence = seq_by_id[seq_id]
                    rep = _compute_residue_repr(sequence)
                    committer.submit_numpy(cache_dir=ESM_EMB_DIR, seq_id=seq_id, array=rep)
        _free_esm()

        # Load EITLEM prediction model only after ESM1v has been freed.
        if kinetics_type == "KCAT":
            eitlem_model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
        else:
            eitlem_model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)

        eitlem_model.load_state_dict(
            torch.load(_WEIGHT_PATHS[kinetics_type], map_location=torch.device("cpu"))
        )
        eitlem_model.eval()

        # ── Phase 2: GNN inference, one sample at a time ─────────────────────
        _BATCH_SIZE = 1
        total = len(sequences)
        predictions: list[float | None] = [None] * total

        pending_indices: list[int] = []
        pending_data:    list[Data] = []

        def _flush_batch() -> None:
            if not pending_indices:
                return
            batch = Batch.from_data_list(pending_data, follow_batch=["pro_emb"])
            with torch.no_grad():
                res = eitlem_model(batch)
            for local_i, global_idx in enumerate(pending_indices):
                predictions[global_idx] = math.pow(10, res[local_i].item())
            pending_indices.clear()
            pending_data.clear()

        for idx, (sequence, substrate, seq_id) in enumerate(zip(sequences, substrates, seq_ids)):
            try:
                mol = Chem.MolFromSmiles(substrate)
                if mol is None:
                    raise ValueError(f"Invalid substrate SMILES: {substrate}")

                mol_feature = torch.FloatTensor(MACCSkeys.GenMACCSKeys(mol).ToList())
                sequence_rep = torch.FloatTensor(np.load(str(_emb_path(seq_id))))

                pending_indices.append(idx)
                pending_data.append(Data(x=mol_feature.unsqueeze(0), pro_emb=sequence_rep))

                if len(pending_indices) == _BATCH_SIZE:
                    _flush_batch()

            except Exception as exc:
                print(f"Error processing sample {idx}: {exc}")

            print(f"Progress: {idx + 1}/{total}", flush=True)

        _flush_batch()

        # ── Write output CSV ─────────────────────────────────────────────────
        pd.DataFrame(
            {
                "Substrate SMILES": substrates,
                "Protein Sequence": sequences,
                "Predicted Value":  predictions,
            }
        ).to_csv(output_file, index=False)
        run_completed = True
    finally:
        # Delete every esm1v file for sequences in this job unless this run is
        # asked to preserve embeddings for another target stage in the same job.
        # Failed runs always clean up their staged full matrices.
        _free_esm()
        if _DELETE_EMBEDDINGS_AFTER_RUN or not run_completed:
            _cleanup_embeddings(seq_ids)


if __name__ == "__main__":
    main()
