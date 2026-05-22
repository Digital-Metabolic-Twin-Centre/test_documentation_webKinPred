"""
Central registry for MMseqs similarity datasets.

Add new datasets here once, then both local and Docker configs will pick them up.
"""

from __future__ import annotations


SIMILARITY_DATASET_REGISTRY: dict[str, dict[str, object]] = {
    # label -> metadata
    "DLKcat/UniKP": {
        "fasta_filename": "dlkcat_sequences.fasta",
        "db_name": "targetdb_dlkcat",
        "method_keys": ["DLKcat", "UniKP"],
    },
    "EITLEM/KinForm": {
        "fasta_filename": "EITLEM_sequences.fasta",
        "db_name": "targetdb_EITLEM",
        "method_keys": ["EITLEM", "KinForm-H", "KinForm-L"],
    },
    "TurNup": {
        "fasta_filename": "turnup_sequences.fasta",
        "db_name": "targetdb_turnup",
        "method_keys": ["TurNup"],
    },
    "CataPro": {
        "fasta_filename": "catapro_sequences.fasta",
        "db_name": "targetdb_catapro",
        "method_keys": ["CataPro"],
    },
    "CatPred/OmniESI": {
        "fasta_filename": "catpred_kcat_sequences.fasta",
        "db_name": "targetdb_catpred_kcat",
        "method_keys": ["CatPred", "OmniESI"],
    },
    "IECata": {
        "fasta_filename": "iecata_sequences.fasta",
        "db_name": "targetdb_iecata",
        "method_keys": ["IECata"],
    },
}
