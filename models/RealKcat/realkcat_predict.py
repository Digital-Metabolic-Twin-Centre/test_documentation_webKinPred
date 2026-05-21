#!/usr/bin/env python
from __future__ import annotations
"""
RealKcat prediction script for webKinPred subprocess engine.

Input: JSON with sequences, substrates, target
Output: JSON with predictions and invalid indices

Usage:
    python predict.py --input <input.json> --output <output.json>
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
import esm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
_ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
_ESM2_LAYER = 33
_CHEMBERTA_MODEL = "seyonec/PubChem10M_SMILES_BPE_450k"
_ESM2_MODEL_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
_MAX_SEQ_LEN = 1022
_MAX_SMILES_LEN = 512

# Global standardization params and class ranges from RealKcat notebook inference flow.
_GLOBAL_MEAN_1 = -0.0006011285004206002
_GLOBAL_STD_1 = 0.18902993202209473
_GLOBAL_MEAN_2 = -0.00015002528380136937
_GLOBAL_STD_2 = 0.6113553047180176

_CLASS_RANGES_KCAT = {
    0: {"low": 0.0, "high": 3.32e-8},
    1: {"low": 3.33e-8, "high": 1.0e-2},
    2: {"low": 1.01e-2, "high": 1.0e-1},
    3: {"low": 1.01e-1, "high": 1.0},
    4: {"low": 1.001, "high": 10.0},
    5: {"low": 1.004e1, "high": 1.0e2},
    6: {"low": 1.0025e2, "high": 1.0e3},
    7: {"low": 1.002e3, "high": 7.0e7},
}

_CLASS_RANGES_KM = {
    0: {"low": 1.0e-10, "high": 1.0e-5},
    1: {"low": 1.01e-5, "high": 1.0e-4},
    2: {"low": 1.002e-4, "high": 1.0e-3},
    3: {"low": 1.002e-3, "high": 1.0e-2},
    4: {"low": 1.008e-2, "high": 1.0e-1},
    5: {"low": 1.01e-1, "high": 1.02e2},
}


def resolve_data_dir() -> str:
    """Resolve model artifact directory across local and Docker runs."""
    script_dir = Path(__file__).resolve().parent
    candidates = []

    env_dir = os.getenv("REALKCAT_DATA")
    if env_dir:
        candidates.append(Path(env_dir))

    # Local repo default for direct CLI usage.
    candidates.append(script_dir / "model_weights")
    # Historical path used by some RealKcat docs.
    candidates.append(script_dir / "data")
    # Docker default path used in containerized deployments.
    candidates.append(Path("/app/models/RealKcat/data"))

    required = ["kcat_model.pkl", "km_model.pkl"]
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return str(candidate)

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "RealKcat model artifacts not found. Expected files "
        f"{required} in one of: {checked}. "
        "Set REALKCAT_DATA to the directory containing these files."
    )


def build_hardcoded_stats(device: torch.device) -> dict:
    """Build the stats payload expected by the prediction pipeline."""
    return {
        "esm2_mean": torch.tensor(_GLOBAL_MEAN_1, device=device),
        "esm2_std": torch.tensor(_GLOBAL_STD_1, device=device),
        "chemberta_mean": torch.tensor(_GLOBAL_MEAN_2, device=device),
        "chemberta_std": torch.tensor(_GLOBAL_STD_2, device=device),
        "class_ranges_kcat": _CLASS_RANGES_KCAT,
        "class_ranges_km": _CLASS_RANGES_KM,
    }


def load_esm2_model(device: torch.device):
    """Load ESM2 model across fair-esm versions with different APIs."""
    try:
        # Newer/alternate fair-esm variants may support this direct call.
        return esm.pretrained.load_model_and_alphabet_core(_ESM2_MODEL_NAME)
    except TypeError:
        # Older fair-esm variants require model_data as the second argument.
        model_data = torch.hub.load_state_dict_from_url(
            _ESM2_MODEL_URL,
            progress=True,
            map_location=device,
        )
        return esm.pretrained.load_model_and_alphabet_core(_ESM2_MODEL_NAME, model_data)


def load_models(device: torch.device, data_dir: str):
    """Load RealKcat models and standardization parameters."""
    # Load XGBoost models
    kcat_model = joblib.load(os.path.join(data_dir, "kcat_model.pkl"))
    km_model = joblib.load(os.path.join(data_dir, "km_model.pkl"))
    
    # Use notebook-derived constants to avoid an extra artifact dependency.
    stats = build_hardcoded_stats(device)
    
    # Load ESM2
    esm_model, alphabet = load_esm2_model(device)
    esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    
    # Load ChemBERTa
    chemberta_tokenizer = AutoTokenizer.from_pretrained(_CHEMBERTA_MODEL)
    chemberta_model = AutoModel.from_pretrained(_CHEMBERTA_MODEL)
    chemberta_model.eval().to(device)
    
    return {
        "kcat_model": kcat_model,
        "km_model": km_model,
        "stats": stats,
        "esm_model": esm_model,
        "alphabet": alphabet,
        "batch_converter": batch_converter,
        "chemberta_tokenizer": chemberta_tokenizer,
        "chemberta_model": chemberta_model,
    }


def get_esm2_embedding_cached(seq: str, seq_id: str, device: torch.device, 
                               models: dict, cache_dir: str = None) -> torch.Tensor:
    """
    Get ESM2 embedding, using webKinPred cache if available.
    
    Cache path: media/sequence_info/omniesi_esm2/{seq_id}.pt
    """
    if cache_dir:
        cache_path = Path(cache_dir) / "omniesi_esm2" / f"{seq_id}.pt"
        if cache_path.exists():
            try:
                embedding = torch.load(cache_path, map_location=device)
                # Mean-pool if full matrix
                if embedding.dim() == 2:  # [seq_len, 1280]
                    return embedding.mean(dim=0)
                return embedding  # Already pooled [1280]
            except Exception as e:
                logger.warning(f"Cache read failed for {seq_id}: {e}")
    
    # Compute embedding
    batch_labels, batch_strs, batch_tokens = models["batch_converter"]([("seq", seq)])
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != models["alphabet"].padding_idx).sum(1)
    
    with torch.no_grad():
        results = models["esm_model"](batch_tokens, repr_layers=[_ESM2_LAYER])
        token_repr = results["representations"][_ESM2_LAYER][0, 1:batch_lens[0]-1]
        embedding = token_repr.mean(dim=0).type(torch.float32)  # [1280]
    
    # Save to cache if path provided
    if cache_dir and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding.cpu(), cache_path)
    
    return embedding


def get_chemberta_embedding(smiles: str, device: torch.device, models: dict) -> torch.Tensor:
    """Get ChemBERTa mean-pooled embedding for a SMILES string."""
    inputs = models["chemberta_tokenizer"]([smiles], return_tensors="pt", 
                                           padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if inputs["input_ids"].shape[1] > _MAX_SMILES_LEN:
        raise ValueError(f"SMILES too long: {inputs['input_ids'].shape[1]} > {_MAX_SMILES_LEN}")
    
    with torch.no_grad():
        outputs = models["chemberta_model"](**inputs)
        # Mean-pool last hidden state, excluding padding
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size())
        masked_hidden = outputs.last_hidden_state * attention_mask
        embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
        return embedding.squeeze(0).type(torch.float32)  # [~768]


def standardize_features(esm2_emb: torch.Tensor, chemberta_emb: torch.Tensor, 
                         stats: dict, device: torch.device) -> np.ndarray:
    """Apply global standardization and concatenate embeddings."""
    # Split stats
    mean1, std1 = stats["esm2_mean"].to(device), stats["esm2_std"].to(device)
    mean2, std2 = stats["chemberta_mean"].to(device), stats["chemberta_std"].to(device)
    
    # Standardize each stream
    std1 = torch.clamp(std1, min=1e-7)
    std2 = torch.clamp(std2, min=1e-7)
    esm2_std = (esm2_emb - mean1) / std1
    chem_std = (chemberta_emb - mean2) / std2
    
    # Concatenate and convert to numpy
    combined = torch.cat([esm2_std, chem_std]).cpu().numpy()
    return combined.reshape(1, -1)  # XGBoost expects 2D array


def validate_input(seq: str, smiles: str) -> tuple[bool, str | None]:
    """Validate sequence and substrate."""
    if not all(c in _AMINO_ACIDS for c in seq):
        return False, "Invalid protein sequence (unsupported amino acid characters)"
    if len(seq) > _MAX_SEQ_LEN:
        return False, f"Sequence too long: {len(seq)} > {_MAX_SEQ_LEN}"
    
    mol = Chem.MolFromSmiles(smiles) or Chem.MolFromInchi(smiles)
    if mol is None:
        return False, "Invalid substrate (not a valid SMILES or InChI)"
    if len(Chem.GetMolFrags(mol)) > 1:
        return False, "Substrate contains multiple disconnected fragments"
    
    return True, None


def predict_batch(rows: list[dict], target: str, models: dict, 
                  device: torch.device, cache_dir: str = None) -> tuple[list, list[int]]:
    """Run predictions on a batch of inputs."""
    predictions = []
    invalid_indices = []

    target_norm = str(target).lower()
    if target_norm not in {"kcat", "km"}:
        raise ValueError(f"Unsupported target '{target}'. Expected 'kcat' or 'Km'.")
    
    model = models["kcat_model"] if target_norm == "kcat" else models["km_model"]
    class_ranges = models["stats"][f"class_ranges_{target_norm}"]
    
    for idx, row in enumerate(rows):
        seq, smiles = row["sequence"], row["substrates"]
        seq_id = row.get("seq_id", f"job_{idx}")
        
        # Validate
        valid, reason = validate_input(seq, smiles)
        if not valid:
            predictions.append(None)
            invalid_indices.append(idx)
            logger.debug(f"Row {idx} invalid: {reason}")
            continue
        
        try:
            # Get embeddings
            esm2_emb = get_esm2_embedding_cached(seq, seq_id, device, models, cache_dir)
            chemberta_emb = get_chemberta_embedding(smiles, device, models)
            
            # Standardize and predict
            features = standardize_features(esm2_emb, chemberta_emb, models["stats"], device)
            pred_class = int(model.predict(features)[0])
            
            # Return the predicted class
            predictions.append(pred_class)
            
            # Progress reporting for webKinPred
            if (idx + 1) % 10 == 0:
                print(f"Progress: {idx + 1}/{len(rows)}", flush=True)
                
        except Exception as e:
            logger.error(f"Prediction failed for row {idx}: {e}")
            predictions.append(None)
            invalid_indices.append(idx)
    
    return predictions, invalid_indices


def main():
    parser = argparse.ArgumentParser(description="RealKcat prediction script")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--cache-dir", default=None, help="Optional PLM embedding cache directory")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    with open(args.input, "r") as f:
        job_data = json.load(f)
    
    rows = job_data["rows"]
    target = job_data["target"]  # "kcat" or "Km"
    data_dir = resolve_data_dir()
    
    if not rows:
        with open(args.output, "w") as f:
            json.dump({"predictions": [], "invalid_indices": []}, f)
        return
    
    # Load models (once per subprocess)
    logger.info("Loading RealKcat models...")
    models = load_models(device, data_dir)
    
    # Predict
    logger.info(f"Running {target} predictions on {len(rows)} samples...")
    predictions, invalid_indices = predict_batch(
        rows, target, models, device, 
        cache_dir=args.cache_dir or os.path.join(os.getenv("MEDIA_ROOT", "media"), "sequence_info")
    )
    
    # Write output
    output = {
        "predictions": predictions,
        "invalid_indices": invalid_indices
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Done. {len(predictions) - len(invalid_indices)}/{len(predictions)} successful predictions.")


if __name__ == "__main__":
    main()