"""Normalize webKinPred substrate collections for CatPred model input."""

from __future__ import annotations

from typing import Any

from rdkit import Chem, rdBase


def substrate_components(value: Any) -> list[str]:
    """Return ordered components from semicolon-separated strings or collections."""
    if isinstance(value, (list, tuple)):
        components: list[str] = []
        for item in value:
            components.extend(substrate_components(item))
        return components

    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan"}:
        return []
    if ";" in text:
        components: list[str] = []
        for item in text.split(";"):
            components.extend(substrate_components(item))
        return components
    if text.startswith("InChI="):
        return [text]
    return [text]


def normalize_catpred_substrates(value: Any, *, canonicalize: bool = True) -> str:
    """Validate components and create CatPred's internal dot-joined SMILES."""
    components = substrate_components(value)
    if not components:
        raise ValueError("Missing substrate")

    smiles_components: list[str] = []
    for component in components:
        with rdBase.BlockLogs():
            if component.startswith("InChI="):
                mol = Chem.MolFromInchi(component)
                preserve_raw = False
            else:
                mol = Chem.MolFromSmiles(component)
                preserve_raw = not canonicalize
        if mol is None:
            raise ValueError(f"Invalid substrate component: {component}")
        if mol.GetNumHeavyAtoms() == 0:
            raise ValueError(
                f"Unsupported substrate for CatPred: {component} contains no heavy atoms"
            )
        smiles_components.append(
            component
            if preserve_raw
            else Chem.MolToSmiles(mol, canonical=canonicalize)
        )
    return ".".join(smiles_components)
