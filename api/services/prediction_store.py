"""
ReconXKG memoization store.

This module is the persistence + keying layer behind the (undocumented)
``recon_xkg`` submit mode. It caches **raw model prediction values** at the
granularity of a single prediction unit, plus per-sequence training-set
similarity, in a dedicated SQLite database (``prediction_store``) using WAL mode
so cache traffic never contends with the primary application DB.

Design notes
------------
* The cache is keyed at the *unit* level — one (sequence, single/native
  substrate set, products, target, method, model_version, params) tuple — rather
  than at the (row, target) level. This is what makes Km / kcat-Km ordered-array
  outputs correct under substrate reordering: per-substrate units are looked up
  individually and reassembled in the caller's input order.
* We store the **raw** model output (before RealKcat class-range formatting,
  substrate reduction, or experimental overrides). Everything downstream of the
  model runs unchanged on merged hit+miss values, so a cached row is
  byte-for-byte identical to a freshly predicted one.
* Every operation is best-effort: any failure logs and degrades to normal
  computation rather than raising into the prediction pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Iterable, Sequence
from functools import lru_cache
from numbers import Real
from typing import Any

from api.utils.convert_to_mol import (
    clean_molecule_text,
    convert_to_mol,
    substrate_as_smiles,
)
from api.utils.substrate_expansion import split_substrate_list
from django.utils import timezone

_log = logging.getLogger(__name__)

# SQLite has a default limit of 999 host parameters per statement (older builds)
# and many more on newer ones; 900 keeps batched ``IN`` lookups safe everywhere.
_IN_CHUNK = 900

# Field separator for hash inputs — a control char that cannot appear in SMILES,
# method names, or hex digests, so distinct field tuples never collide.
_SEP = "\x1f"


# ---------------------------------------------------------------------------
# Hashing / keying
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=100_000)
def _canon_token(token: str, canonicalize: bool) -> str | None:
    """
    Canonicalize one molecule token for keying (process-wide memoized).

    When ``canonicalize`` is True we use the RDKit canonical SMILES — matching
    what the engine feeds the model. When False we keep the raw (validated)
    text, because the model then sees the raw string and two raw strings that
    happen to canonicalize alike could yield different values.

    Returns None when the token is not a parseable molecule (caller treats the
    whole unit as uncacheable).
    """
    text = clean_molecule_text(token)
    if not text:
        return None
    if canonicalize:
        return substrate_as_smiles(text, canonicalize=True)
    return text if convert_to_mol(text) is not None else None


def canonical_unit(value: Any, canonicalize: bool) -> str | None:
    """
    Build the order-independent canonical form of a substrate/product unit.

    ``value`` may be a single token, a ``;``-separated list (the ``Substrates``
    schema), or an already-split list/tuple (native-multi grouping). Tokens are
    canonicalized, sorted (set semantics — order does not change the prediction)
    and re-joined with ``;``. Returns None if any token is unparseable or the
    unit is empty.
    """
    if isinstance(value, (list, tuple)):
        tokens = [str(token) for token in value]
    else:
        tokens = split_substrate_list(value)

    canon_tokens: list[str] = []
    for token in tokens:
        canon = _canon_token(token, canonicalize)
        if canon is None:
            return None
        canon_tokens.append(canon)

    if not canon_tokens:
        return None
    return ";".join(sorted(canon_tokens))


def params_fingerprint(canonicalize_substrates: bool, target_kwargs: dict | None) -> str:
    """
    Hash the result-affecting parameters that are not already captured by the
    sequence hash / canonical substrate fields.

    ``handle_long_sequences`` is intentionally absent: the key uses the actual
    (post-truncation) sequence, so truncation is already reflected. ``skip`` only
    omits rows, it does not change a value for a given sequence.
    """
    payload = _SEP.join(
        [
            "canon=1" if canonicalize_substrates else "canon=0",
            "tk=" + _stable_mapping(target_kwargs or {}),
        ]
    )
    return sha256_text(payload)


def _stable_mapping(mapping: dict) -> str:
    return ",".join(f"{k}={mapping[k]}" for k in sorted(mapping, key=str))


def make_lookup_key(
    *,
    target: str,
    method: str,
    model_version: str,
    params_fp: str,
    sequence_sha256: str,
    substrate_canon: str,
    products_canon: str,
) -> str:
    """Compose the SHA-256 lookup key over all prediction-affecting fields."""
    return sha256_text(
        _SEP.join(
            [
                target,
                method,
                model_version,
                params_fp,
                sequence_sha256,
                substrate_canon,
                products_canon or "",
            ]
        )
    )


def build_unit_keys(
    descriptor: Any,
    target: str,
    sequences: Sequence[Any],
    call_kwargs: dict[str, Any],
    canonicalize_substrates: bool,
) -> tuple[list[str | None], list[tuple[str, str, str] | None], str]:
    """Build position-aligned ReconXKG keys for one planned engine batch."""
    count = len(sequences)
    model_version = getattr(descriptor, "model_version", "1")
    params_fp = params_fingerprint(
        canonicalize_substrates,
        descriptor.target_kwargs.get(target, {}),
    )
    substrate_kwarg = descriptor.col_to_kwarg.get(
        "Substrate"
    ) or descriptor.col_to_kwarg.get("Substrates")
    products_kwarg = descriptor.col_to_kwarg.get("Products")
    substrate_values = call_kwargs.get(substrate_kwarg) if substrate_kwarg else None
    product_values = call_kwargs.get(products_kwarg) if products_kwarg else None

    keys: list[str | None] = [None] * count
    components: list[tuple[str, str, str] | None] = [None] * count
    for index in range(count):
        sequence = sequences[index]
        if sequence is None:
            continue
        sequence_text = str(sequence).strip()
        if not sequence_text or any(
            residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in sequence_text
        ):
            continue

        substrate = (
            substrate_values[index]
            if isinstance(substrate_values, (list, tuple))
            and index < len(substrate_values)
            else None
        )
        if substrate is None:
            continue
        substrate_canon = canonical_unit(substrate, canonicalize_substrates)
        if substrate_canon is None:
            continue

        products_canon = ""
        if product_values is not None:
            products = (
                product_values[index]
                if isinstance(product_values, (list, tuple))
                and index < len(product_values)
                else None
            )
            if products is not None and str(products).strip():
                products_canon = canonical_unit(products, canonicalize_substrates)
                if products_canon is None:
                    continue

        sequence_sha = sha256_text(str(sequence))
        keys[index] = make_lookup_key(
            target=target,
            method=descriptor.key,
            model_version=model_version,
            params_fp=params_fp,
            sequence_sha256=sequence_sha,
            substrate_canon=substrate_canon,
            products_canon=products_canon,
        )
        components[index] = (sequence_sha, substrate_canon, products_canon)

    return keys, components, params_fp


def coerce_value(raw: Any) -> float | None:
    """Coerce a raw model output to a finite float, or None if not storable."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, Real):
        number = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() in {"none", "nan", "inf", "+inf", "-inf"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Prediction-unit cache I/O
# ---------------------------------------------------------------------------


def get_many(keys: Iterable[str]) -> dict[str, float]:
    """
    Batch-fetch cached prediction values for ``keys``.

    Returns a ``{lookup_key: value}`` dict containing only the hits. Reads are
    chunked to stay within SQLite's host-parameter limit. Best-effort: any error
    returns an empty dict (callers then treat everything as a miss).
    """
    from api.models import PredictionStore

    unique_keys = list({k for k in keys if k})
    if not unique_keys:
        return {}

    hits: dict[str, float] = {}
    try:
        for start in range(0, len(unique_keys), _IN_CHUNK):
            chunk = unique_keys[start : start + _IN_CHUNK]
            rows = PredictionStore.objects.filter(lookup_key__in=chunk).values_list(
                "lookup_key", "value"
            )
            for lookup_key, value in rows:
                hits[lookup_key] = value
    except Exception:
        _log.warning(
            "ReconXKG prediction cache read failed; treating as full miss",
            extra={"event": "recon_xkg.cache_read_failed", "requested": len(unique_keys)},
            exc_info=True,
        )
        return {}
    return hits


def upsert_many(rows: Sequence[dict[str, Any]]) -> int:
    """
    Append/overwrite prediction-unit rows by ``lookup_key`` (write-through).

    ``rows`` is a list of dicts with the full set of store columns. Uses a single
    ``INSERT ... ON CONFLICT(lookup_key) DO UPDATE`` statement so concurrent jobs
    that compute the same miss converge on one row. Best-effort.
    """
    from api.models import PredictionStore

    if not rows:
        return 0

    now = timezone.now()
    objects = [
        PredictionStore(
            lookup_key=row["lookup_key"],
            target=row["target"],
            method=row["method"],
            model_version=row["model_version"],
            params_fingerprint=row["params_fingerprint"],
            sequence_sha256=row["sequence_sha256"],
            substrate_canon=row["substrate_canon"],
            products_canon=row.get("products_canon", ""),
            value=row["value"],
            created_at=now,
            updated_at=now,
        )
        for row in rows
    ]
    try:
        PredictionStore.objects.bulk_create(
            objects,
            update_conflicts=True,
            unique_fields=["lookup_key"],
            update_fields=["value", "updated_at"],
        )
    except Exception:
        _log.warning(
            "ReconXKG prediction cache write failed; results still returned",
            extra={"event": "recon_xkg.cache_write_failed", "rows": len(objects)},
            exc_info=True,
        )
        return 0
    return len(objects)


# ---------------------------------------------------------------------------
# Similarity cache I/O
# ---------------------------------------------------------------------------


def similarity_key(sequence_sha256: str, dataset_label: str) -> str:
    return sha256_text(_SEP.join(["sim", dataset_label, sequence_sha256]))


def get_similarity_many(
    sequence_sha_by_seq: dict[str, str],
    dataset_label: str,
) -> dict[str, tuple[float | None, float | None]]:
    """
    Fetch cached (mean, max) similarity for a set of sequences.

    ``sequence_sha_by_seq`` maps raw sequence -> its sha256. Returns a dict keyed
    by raw sequence for the hits only.
    """
    from api.models import SimilarityStore

    if not sequence_sha_by_seq:
        return {}

    key_to_seq: dict[str, str] = {}
    for seq, seq_sha in sequence_sha_by_seq.items():
        key_to_seq[similarity_key(seq_sha, dataset_label)] = seq

    out: dict[str, tuple[float | None, float | None]] = {}
    keys = list(key_to_seq.keys())
    try:
        for start in range(0, len(keys), _IN_CHUNK):
            chunk = keys[start : start + _IN_CHUNK]
            rows = SimilarityStore.objects.filter(lookup_key__in=chunk).values_list(
                "lookup_key", "mean_similarity", "max_similarity"
            )
            for lookup_key, mean_sim, max_sim in rows:
                out[key_to_seq[lookup_key]] = (mean_sim, max_sim)
    except Exception:
        _log.warning(
            "ReconXKG similarity cache read failed; recomputing",
            extra={"event": "recon_xkg.sim_read_failed"},
            exc_info=True,
        )
        return {}
    return out


def upsert_similarity_many(
    entries: Sequence[tuple[str, str, float | None, float | None]],
    dataset_label: str,
) -> int:
    """
    Upsert per-sequence similarity rows.

    ``entries`` is a list of (sequence, sequence_sha256, mean, max). Best-effort.
    """
    from api.models import SimilarityStore

    if not entries:
        return 0

    now = timezone.now()
    objects = [
        SimilarityStore(
            lookup_key=similarity_key(seq_sha, dataset_label),
            sequence_sha256=seq_sha,
            dataset_label=dataset_label,
            mean_similarity=mean_sim,
            max_similarity=max_sim,
            created_at=now,
            updated_at=now,
        )
        for _seq, seq_sha, mean_sim, max_sim in entries
    ]
    try:
        SimilarityStore.objects.bulk_create(
            objects,
            update_conflicts=True,
            unique_fields=["lookup_key"],
            update_fields=["mean_similarity", "max_similarity", "updated_at"],
        )
    except Exception:
        _log.warning(
            "ReconXKG similarity cache write failed",
            extra={"event": "recon_xkg.sim_write_failed", "rows": len(objects)},
            exc_info=True,
        )
        return 0
    return len(objects)
