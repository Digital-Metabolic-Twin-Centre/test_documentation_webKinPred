"""
IECata prediction script for webKinPred (ephemeral per-residue ProtT5 cache).

Usage:
    python predict_webkinpred.py --input <input.json> --output <output.json>

Environment variables injected by SubprocessEngineConfig:
    IECATA_DATA      — path to models/IECata (model weights + configs)
    IECATA_EMBED_DIR — path to media/sequence_info/iecata_prot_t5_residues/

Input JSON (webKinPred contract):
    {
      "method": "IECata",
      "target": "kcat/Km",
      "public_id": "abc1234",
      "rows": [
        {
          "sequence": "MKTLL...",
          "substrates": "CC(=O)O",
          "seq_id": "sha256_or_similar_id"
        }
      ],
      "params": {}
    }

Output JSON:
    {
      "predictions": [1.23e-4, null, ...],
      "invalid_indices": [2, 5]
    }

Predictions are in native units (kcat/KM s^-1 M^-1), NOT log10.
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from functools import partial

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models import DrugBAN
from utils import set_seed, prottrans_graph_collate_func
from configs import get_cfg_defaults
from torch.utils.data import Dataset, DataLoader
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

DATA_ROOT   = os.environ.get("IECATA_DATA", SCRIPT_DIR)
EMBED_DIR   = os.environ.get("IECATA_EMBED_DIR", "")
CONFIG_PATH = os.path.join(DATA_ROOT, "configs", "independent_test.yaml")
MODEL_PATH  = os.path.join(
    DATA_ROOT, "result", "loss_lamba0.2_seed100", "best_model_epoch_80.pth"
)
MAX_PROT_LEN = 1200   # IECata pads protein dim to this in the dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------
# Dataset — bypasses Prot_fasta_Feature_Extraction; loads cached .npy instead
# ---------------------------------------------------------------------------

class CachedDTIDataset(Dataset):
    """
    Like IECata's DTIDataset but reads protein embeddings from pre-computed
    .npy files (shape: seq_len x 1024) rather than running ProtT5 inline.
    """

    def __init__(self, rows, embed_dir, max_drug_nodes=290):
        self.rows = rows
        self.embed_dir = embed_dir
        self.max_drug_nodes = max_drug_nodes
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row    = self.rows[idx]
        smiles = row["substrates"]
        seq_id = row["seq_id"]

        # ── Drug graph ───────────────────────────────────────────────────────
        v_d = self.fc(
            smiles=smiles,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )
        actual_node_feats  = v_d.ndata.pop("h")
        num_actual_nodes   = actual_node_feats.shape[0]
        num_virtual_nodes  = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit   = torch.zeros([num_actual_nodes, 1])
        actual_node_feats  = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata["h"]     = actual_node_feats
        virtual_node_feat  = torch.cat(
            (torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1
        )
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        # ── Protein embedding from cache ──────────────────────────────────────
        npy_path = os.path.join(self.embed_dir, f"{seq_id}.npy")
        residue_matrix = np.load(npy_path)          # (seq_len, 1024)

        # Pad to (MAX_PROT_LEN, 1024) — same as original dataloader
        seq_len = residue_matrix.shape[0]
        pad_len = MAX_PROT_LEN - seq_len
        if pad_len > 0:
            pad = np.zeros((pad_len, residue_matrix.shape[1]), dtype=residue_matrix.dtype)
            residue_matrix = np.vstack([residue_matrix, pad])
        elif pad_len < 0:
            residue_matrix = residue_matrix[:MAX_PROT_LEN, :]   # truncate if over limit

        v_p = residue_matrix   # shape: (MAX_PROT_LEN, 1024)

        # Placeholder label and weight — not used during inference
        y = np.float32(0.0)
        w = np.float32(1.0)
        return v_d, v_p, y, w


def load_model(cfg):
    model = DrugBAN(**cfg).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def run_inference(model, dataset, batch_size=16):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=prottrans_graph_collate_func,
    )
    n = len(dataset)
    all_preds = []

    with torch.no_grad():
        for v_d, v_p, labels, weights in loader:
            v_d     = v_d.to(DEVICE)
            v_p     = v_p.to(DEVICE)
            _, _, score, _ = model(v_d, v_p, mode="eval")
            # score layout: [mean, lambda, alpha, beta] x n_heads — extract means
            means   = score[:, [j for j in range(score.shape[1]) if j % 4 == 0]]
            batch   = means.mean(dim=1).cpu().numpy().tolist()
            all_preds.extend(batch)
            print(f"Progress: {len(all_preds)}/{n}", flush=True)

    return all_preds   # log10(kcat/KM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        job = json.load(f)

    rows = job.get("rows", [])
    n    = len(rows)

    if n == 0:
        with open(args.output, "w") as f:
            json.dump({"predictions": [], "invalid_indices": []}, f)
        return

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = get_cfg_defaults()
    cfg.merge_from_file(CONFIG_PATH)
    cfg.defrost()
    cfg.RESULT.OUTPUT_DIR  = "/tmp/iecata_inference"
    cfg.SOLVER.NUM_WORKERS = 0
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    # ── Separate valid rows from structurally invalid ones ───────────────────
    valid_rows      = []
    invalid_indices = []
    index_map       = []   # valid_rows[i] came from rows[index_map[i]]

    for idx, row in enumerate(rows):
        seq_id = row.get("seq_id", "")
        smiles = row.get("substrates", "")
        npy    = os.path.join(EMBED_DIR, f"{seq_id}.npy") if seq_id else ""

        if not seq_id or not smiles or not os.path.exists(npy):
            invalid_indices.append(idx)
        else:
            valid_rows.append(row)
            index_map.append(idx)

    # ── Load model and run inference ─────────────────────────────────────────
    predictions = [None] * n

    if valid_rows:
        model   = load_model(cfg)
        dataset = CachedDTIDataset(valid_rows, embed_dir=EMBED_DIR)
        log10_preds = run_inference(model, dataset, batch_size=cfg.SOLVER.BATCH_SIZE)

        touched_npy = set()
        for i, log_val in enumerate(log10_preds):
            orig_idx = index_map[i]
            try:
                predictions[orig_idx] = float(10 ** log_val)
                touched_npy.add(
                    os.path.join(EMBED_DIR, f"{valid_rows[i]['seq_id']}.npy")
                )
            except Exception:
                predictions[orig_idx] = None
                if orig_idx not in invalid_indices:
                    invalid_indices.append(orig_idx)

        # ── Ephemeral cleanup (EITLEM pattern) ───────────────────────────────
        for path in touched_npy:
            try:
                os.remove(path)
            except OSError:
                pass

    with open(args.output, "w") as f:
        json.dump(
            {
                "predictions":     predictions,
                "invalid_indices": sorted(invalid_indices),
            },
            f,
        )


if __name__ == "__main__":
    main()

