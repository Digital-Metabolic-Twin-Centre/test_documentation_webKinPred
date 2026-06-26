"""
test_api.py — Comprehensive API test suite for the Open Kinetics Predictor REST API.

Tests every endpoint, every expected success response, and every expected error
response.  Designed to run against a local Django dev server.

Usage:
    python tests/test_api.py                        # uses the default key below
    python tests/test_api.py --key ak_yourkey       # pass a key on the command line
    python tests/test_api.py --base http://host:8000/api/v1   # different server
    python tests/test_api.py --test-multi-sub-for-single-sub-methods  # slow value checks
    python tests/test_api.py --only-multimer        # focused multi-sequence checks
"""

import argparse
import hashlib
import io
import json
import math
import sys
import textwrap
import time

import requests

# ---------------------------------------------------------------------------
# Configuration — edit or pass via CLI flags
# ---------------------------------------------------------------------------

DEFAULT_BASE = "http://127.0.0.1:8000/api/v1"
DEFAULT_KEY = "ak_17f90e7c1f6ac3f5fc861d8cec4667a2b888c358a333bb81f75b631a9b50066b"

# ---------------------------------------------------------------------------
# Known method IDs (normalised to lowercase for comparison)
# ---------------------------------------------------------------------------

# kcat-capable methods
KCAT_METHOD_IDS = [
    "DLKcat",
    "TurNup",
    "EITLEM",
    "UniKP",
    "KinForm-H",
    "KinForm-L",
    "CataPro",
    "CatPred",
    "OmniESI",
    "OmniESI-O2DENet",
    "RealKcat",
]
# Km-capable methods
KM_METHOD_IDS = [
    "EITLEM",
    "UniKP",
    "KinForm-H",
    "CataPro",
    "CatPred",
    "OmniESI",
    "OmniESI-O2DENet",
    "RealKcat",
    "MMISA-KM",
]
# kcat/Km-capable methods
KCAT_KM_METHOD_IDS = ["CataPro", "IECata"]
# All recognised method IDs (de-duplicated, lowercase)
ALL_METHOD_IDS = sorted({m.lower() for m in KCAT_METHOD_IDS + KM_METHOD_IDS + KCAT_KM_METHOD_IDS})

# GPU-offload-capable methods (subset of the above)
GPU_SUPPORTED_KCAT_METHOD_IDS = [
    "KinForm-H",
    "KinForm-L",
    "UniKP",
    "TurNup",
    "CataPro",
    "OmniESI",
    "OmniESI-O2DENet",
    "EITLEM",
    "CatPred",
    "RealKcat",
]
GPU_SUPPORTED_KM_METHOD_IDS = [
    "KinForm-H",
    "UniKP",
    "CataPro",
    "OmniESI",
    "OmniESI-O2DENet",
    "EITLEM",
    "CatPred",
    "RealKcat",
]
GPU_SUPPORTED_KCAT_KM_METHOD_IDS = ["CataPro", "IECata"]



def sel(methods: set, *names: str) -> bool:
    """Return True if *any* of the given method names appear in the selected set."""
    return any(n.lower() in methods for n in names)


def selected_kcat_methods(methods: set) -> list[str]:
    """Return selected kcat-capable methods in canonical order."""
    return [m for m in KCAT_METHOD_IDS if m.lower() in methods]


def selected_km_methods(methods: set) -> list[str]:
    """Return selected Km-capable methods in canonical order."""
    return [m for m in KM_METHOD_IDS if m.lower() in methods]


def selected_kcat_km_methods(methods: set) -> list[str]:
    """Return selected kcat/Km-capable methods in canonical order."""
    return [m for m in KCAT_KM_METHOD_IDS if m.lower() in methods]


# ---------------------------------------------------------------------------
# Tiny inline CSV fixtures
# ---------------------------------------------------------------------------

# Standard single-substrate CSV (used by DLKcat, EITLEM, UniKP, KinForm-*)
# Substrates use simple SMILES strings (no commas) to avoid CSV parsing issues.
SINGLE_SUBSTRATE_CSV = textwrap.dedent("""\
    Protein Sequence,Substrate
    MAAAALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,CC(=O)O
    MEMLEEHRCFEGWQQRWRHDSSTLNCPMTFSIFLPPPRDHTPPPVLYWLSGLTCNDENFTTKAGAQRVAAELGIVLVMPDTSPRGEKVANDDGYDLGQGAGFYLNATQPPWATHYRMYDYLRDELPALVQSQFNVSDRCAISGHSMGGHGALIMALKNPGKYTSVSAFAPIVNPCSVPWGIKAFSSYLGEDKNAWLEWDSCALMYASNAQDAIPTLIAQGDNDQFLADQLQPAVLAEAARQKAWPMTLRIQPGYDHSYYFIASFIEDHLRFHAQYLLK,c1ccccc1
    MCTAITLNGNSNYFGRNLDLDFSYGEEVIITPAEYEFKFRKEKAIKNHKSLIGVGIVANDYPLYFDAINEDGLGMAGLNFPGNAYYSDALENDKDNITPFEFIPWILGQCSDVNEARNLVEKINLINLSFSEQLPLAGLHWLIADREKSIVVEVTKSGVHIYDNPIGILTNNPEFNYQMYNLNKYRNLSISTPQNTFSDSVDLKVDGTGFGGIGLPGDVSPESRFVRATFSKLNSSKGMTVEEDITQFFHILGTVEQIKGVNKTESGKEEYTVYSNCYDLDNKTLYYTTYENRQIVAVTLNKDKDGNRLVTYPFERKQIINKLN,OCC(O)CO
""")

# CatPred compatibility fixture: the historical single "Substrate" column
# with dot-joined co-substrates remains accepted for CatPred kcat.
CATPRED_DOTJOIN_SUBSTRATE_CSV = textwrap.dedent("""\
    Protein Sequence,Substrate
    MAAAALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,CC(=O)O.O
    MEMLEEHRCFEGWQQRWRHDSSTLNCPMTFSIFLPPPRDHTPPPVLYWLSGLTCNDENFTTKAGAQRVAAELGIVLVMPDTSPRGEKVANDDGYDLGQGAGFYLNATQPPWATHYRMYDYLRDELPALVQSQFNVSDRCAISGHSMGGHGALIMALKNPGKYTSVSAFAPIVNPCSVPWGIKAFSSYLGEDKNAWLEWDSCALMYASNAQDAIPTLIAQGDNDQFLADQLQPAVLAEAARQKAWPMTLRIQPGYDHSYYFIASFIEDHLRFHAQYLLK,C1CCCCC1.O
    MCTAITLNGNSNYFGRNLDLDFSYGEEVIITPAEYEFKFRKEKAIKNHKSLIGVGIVANDYPLYFDAINEDGLGMAGLNFPGNAYYSDALENDKDNITPFEFIPWILGQCSDVNEARNLVEKINLINLSFSEQLPLAGLHWLIADREKSIVVEVTKSGVHIYDNPIGILTNNPEFNYQMYNLNKYRNLSISTPQNTFSDSVDLKVDGTGFGGIGLPGDVSPESRFVRATFSKLNSSKGMTVEEDITQFFHILGTVEQIKGVNKTESGKEEYTVYSNCYDLDNKTLYYTTYENRQIVAVTLNKDKDGNRLVTYPFERKQIINKLN,OCC(O)CO.CCO
""")

# Full-reaction CSV (TurNup uses it natively; pair methods expand Substrates)
FULL_REACTION_CSV = textwrap.dedent("""\
    Protein Sequence,Substrates,Products
    MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,CC(=O)O;O,CC(O)=O;[H+]
    MEMLEEHRCFEGWQQRWRHDSSTLNCPMTFSIFLPPPRDHTPPPVLYWLSGLTCNDENFTTKAGAQRVAAELGIVLVMPDTSPRGEKVANDDGYDLGQGAGFYLNATQPPWATHYRMYDYLRDELPALVQSQFNVSDRCAISGHSMGGHGALIMALKNPGKYTSVSAFAPIVNPCSVPWGIKAFSSYLGEDKNAWLEWDSCALMYASNAQDAIPTLIAQGDNDQFLADQLQPAVLAEAARQKAWPMTLRIQPGYDHSYYFIASFIEDHLRFHAQYLLK,C1CCCCC1;O,OC1CCCCC1;[H+]
""")

SUBSTRATE_LIST_CSV = textwrap.dedent("""\
    Protein Sequence,Substrates
    MAAAALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,CC(=O)O;O;C1CCCCC1
""")

# CSV missing the required "Substrate" column — to test validation errors
MISSING_COLUMN_CSV = textwrap.dedent("""\
    Protein Sequence,Extra Column
    MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,foo
""")

# CSV with one valid row and one row with an invalid substrate + invalid protein.
# Used to verify that /validate/ correctly detects invalid content.
INVALID_CONTENT_CSV = textwrap.dedent("""\
    Protein Sequence,Substrate
    MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAELIEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKSGHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYFYPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIERWPGMPE,CC(=O)O
    !!!NOT_A_VALID_AMINO_ACID_SEQUENCE_123!!!,NOT_A_VALID_SMILES_OR_INCHI_STRING
""")

# A plain text file that is NOT a CSV — for extension validation
NOT_A_CSV_BYTES = b"this is not a csv file"

# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []  # (name, passed, detail)


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Record and print a single assertion."""
    status = "PASS" if condition else "FAIL"
    colour = "\033[32m" if condition else "\033[31m"
    reset = "\033[0m"
    pad = "." * max(0, 65 - len(name))
    print(f"  {colour}{status}{reset}  {name}{pad} {detail}")
    _results.append((name, condition, detail))
    return condition


def section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def csv_file(content: str, filename: str = "input.csv"):
    """Return a (name, file-object, mime-type) tuple for requests.post files=."""
    return (filename, io.BytesIO(content.encode("utf-8")), "text/csv")


def submit(
    base: str,
    headers: dict,
    csv_content: str,
    prediction_type: str,
    kcat_method: str = None,
    km_method: str = None,
    kcat_km_method: str = None,
    handle_long: str = "truncate",
    use_experimental: str = "false",
    include_similarity_columns: bool | None = None,
) -> requests.Response:
    """Helper: POST to /submit/ and return the response."""
    targets = []
    methods = {}
    if prediction_type == "kcat":
        targets = ["kcat"]
        if kcat_method:
            methods["kcat"] = kcat_method
    elif prediction_type == "Km":
        targets = ["Km"]
        if km_method:
            methods["Km"] = km_method
    elif prediction_type == "kcat/Km":
        targets = ["kcat/Km"]
        if kcat_km_method:
            methods["kcat/Km"] = kcat_km_method
    return submit_targets(
        base,
        headers,
        csv_content,
        targets,
        methods,
        handle_long=handle_long,
        use_experimental=use_experimental,
        include_similarity_columns=include_similarity_columns,
    )


def submit_targets(
    base: str,
    headers: dict,
    csv_content: str,
    targets: list[str],
    methods: dict[str, str],
    handle_long: str = "truncate",
    use_experimental: str = "false",
    include_similarity_columns: bool | None = None,
    recon_xkg: bool | None = None,
) -> requests.Response:
    """Submit an arbitrary supported target combination."""
    data = {
        "targets": json.dumps(targets),
        "methods": json.dumps(methods),
        "handleLongSequences": handle_long,
        "useExperimental": use_experimental,
    }
    if include_similarity_columns is not None:
        data["includeSimilarityColumns"] = "true" if include_similarity_columns else "false"
    if recon_xkg is not None:
        data["recon_xkg"] = "true" if recon_xkg else "false"

    return requests.post(
        f"{base}/submit/",
        headers=headers,
        files={"file": csv_file(csv_content)},
        data=data,
    )


def choose_submit_csv(
    prediction_type: str,
    kcat_method: str | None = None,
    km_method: str | None = None,
) -> str:
    """
    Choose the CSV fixture that matches method-specific input expectations.

    - TurNup (kcat) uses full-reaction CSV.
    - CatPred kcat compatibility tests use dot-joined values in "Substrate".
    - Other methods use the standard single-substrate fixture.
    """
    if prediction_type == "kcat":
        if kcat_method == "TurNup":
            return FULL_REACTION_CSV
        if kcat_method == "CatPred":
            return CATPRED_DOTJOIN_SUBSTRATE_CSV
    return SINGLE_SUBSTRATE_CSV


def _unique_protein_sequence(seed: str, length: int = 240) -> str:
    """
    Build a deterministic amino-acid sequence from a seed string.
    """
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    state = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    out: list[str] = []
    idx = 0
    while len(out) < length:
        state = hashlib.sha256(f"{state}:{idx}".encode("utf-8")).hexdigest()
        idx += 1
        for ch in state:
            out.append(alphabet[int(ch, 16) % len(alphabet)])
            if len(out) >= length:
                break
    return "".join(out)


def build_gpu_uncached_csv(label: str) -> str:
    """
    Build a small CSV fixture with unique proteins so GPU tests exercise
    uncached embedding work.
    """
    nonce = f"{label}:{time.time_ns()}"
    seq_a = _unique_protein_sequence(f"{nonce}:a")
    seq_b = _unique_protein_sequence(f"{nonce}:b")
    return textwrap.dedent(
        f"""\
        Protein Sequence,Substrates,Products
        {seq_a},CC(=O)O;O,CC(O)=O;[H+]
        {seq_b},C1CCCCC1;O,OC1CCCCC1;[H+]
    """
    )


def expected_kcat_similarity_columns(submitted: dict) -> tuple[str, str] | None:
    """
    Return expected similarity column names for kcat-target jobs, else None.
    """
    if submitted.get("prediction_type") != "kcat":
        return None
    if not submitted.get("include_similarity_columns", True):
        return None
    method_key = submitted.get("kcat_method")
    if not method_key:
        return None
    return (
        f"mean similarity to {method_key} training data",
        f"max similarity to {method_key} training data",
    )


def is_valid_similarity_cell(value) -> bool:
    """
    Similarity values are valid when blank/NaN or numeric in [0, 100].
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if math.isnan(number):
        return True
    return 0.0 <= number <= 100.0


# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------


def test_health(base: str) -> None:
    section("GET /health/ — no auth required")
    r = requests.get(f"{base}/health/")
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    j = r.json()
    check("has status=ok", j.get("status") == "ok")
    check("has service key", "service" in j)
    check("has version key", "version" in j)
    check("has timestamp", "timestamp" in j)


def test_methods(base: str, methods: set) -> None:
    section("GET /methods/ — no auth required")
    r = requests.get(f"{base}/methods/")
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    j = r.json()
    check("has predictionTypes", "predictionTypes" in j)
    check("kcat in predictionTypes", "kcat" in j.get("predictionTypes", []))
    check("Km in predictionTypes", "Km" in j.get("predictionTypes", []))
    check(
        "kcat/Km in predictionTypes",
        "kcat/Km" in j.get("predictionTypes", []),
    )
    check("has methods.kcat", isinstance(j.get("methods", {}).get("kcat"), list))
    check("has methods.Km", isinstance(j.get("methods", {}).get("Km"), list))
    check("has methods.kcat/Km", isinstance(j.get("methods", {}).get("kcat/Km"), list))
    kcat_ids = {m["id"] for m in j.get("methods", {}).get("kcat", [])}
    all_method_entries = [
        method
        for target_methods in j.get("methods", {}).values()
        for method in target_methods
    ]
    pair_entries = [
        method for method in all_method_entries if method.get("inputFormat") == "single"
    ]
    check(
        "pair methods accept substrate_list",
        bool(pair_entries)
        and all("substrate_list" in method.get("acceptedCsvTypes", []) for method in pair_entries),
    )
    check(
        "methods publish target-specific input metadata",
        bool(all_method_entries)
        and all(
            isinstance(method.get("acceptedCsvTypesByTarget"), dict)
            for method in all_method_entries
        )
        and all(
            isinstance(method.get("inputBehaviorByTarget"), dict)
            for method in all_method_entries
        ),
    )
    catpred_kcat = next(
        (
            method
            for method in j.get("methods", {}).get("kcat", [])
            if method.get("id") == "CatPred"
        ),
        None,
    )
    catpred_km = next(
        (method for method in j.get("methods", {}).get("Km", []) if method.get("id") == "CatPred"),
        None,
    )
    if catpred_kcat and catpred_km:
        check(
            "CatPred kcat is native multi-substrate",
            catpred_kcat.get("inputBehaviorByTarget", {}).get("kcat") == "native_multi",
        )
        check(
            "CatPred Km is expanded per substrate",
            catpred_km.get("inputBehaviorByTarget", {}).get("Km") == "expanded_pair",
        )
        check(
            "legacy dot CSV is CatPred-kcat-only",
            "multi" in catpred_kcat.get("acceptedCsvTypesByTarget", {}).get("kcat", [])
            and "multi" not in catpred_km.get("acceptedCsvTypesByTarget", {}).get("Km", []),
        )
    turnup_entries = [method for method in all_method_entries if method.get("id") == "TurNup"]
    if turnup_entries:
        check(
            "TurNup only accepts full_reaction",
            all(method.get("acceptedCsvTypes") == ["full_reaction"] for method in turnup_entries),
        )
    if sel(methods, "DLKcat"):
        check("DLKcat in kcat methods", "DLKcat" in kcat_ids)
    if sel(methods, "TurNup"):
        check("TurNup in kcat methods", "TurNup" in kcat_ids)
    if sel(methods, "EITLEM"):
        check("EITLEM in kcat methods", "EITLEM" in kcat_ids)
    if sel(methods, "UniKP"):
        check("UniKP in kcat methods", "UniKP" in kcat_ids)
    if sel(methods, "KinForm-H"):
        check("KinForm-H in kcat", "KinForm-H" in kcat_ids)
    if sel(methods, "KinForm-L"):
        check("KinForm-L in kcat", "KinForm-L" in kcat_ids)
    if sel(methods, "CatPred"):
        check("CatPred in kcat methods", "CatPred" in kcat_ids)
    if sel(methods, "RealKcat"):
        check("RealKcat in kcat methods", "RealKcat" in kcat_ids)
    km_ids = {m["id"] for m in j.get("methods", {}).get("Km", [])}
    if sel(methods, "EITLEM"):
        check("EITLEM in Km methods", "EITLEM" in km_ids)
    if sel(methods, "UniKP"):
        check("UniKP in Km methods", "UniKP" in km_ids)
    if sel(methods, "KinForm-H"):
        check("KinForm-H in Km methods", "KinForm-H" in km_ids)
    if sel(methods, "CatPred"):
        check("CatPred in Km methods", "CatPred" in km_ids)
    if sel(methods, "RealKcat"):
        check("RealKcat in Km methods", "RealKcat" in km_ids)
    ratio_ids = {m["id"] for m in j.get("methods", {}).get("kcat/Km", [])}
    if sel(methods, "CataPro"):
        check("CataPro in kcat/Km methods", "CataPro" in ratio_ids)
    if sel(methods, "IECata"):
        check("IECata in kcat/Km methods", "IECata" in ratio_ids)
    check("has longSequenceOptions", "longSequenceOptions" in j)


def test_auth(base: str, key: str) -> None:
    section("Authentication — invalid / missing keys")

    # No Authorization header at all
    r = requests.get(f"{base}/quota/")
    check("no header → 401", r.status_code == 401, f"got {r.status_code}")
    check("error key present", "error" in r.json())

    # Wrong Authorization format (not "Bearer ...")
    r = requests.get(f"{base}/quota/", headers={"Authorization": "Basic abc123"})
    check("non-Bearer scheme → 401", r.status_code == 401, f"got {r.status_code}")

    # Bearer prefix present but key does not exist in DB
    r = requests.get(f"{base}/quota/", headers={"Authorization": "Bearer ak_doesnotexist"})
    check("fake key → 401", r.status_code == 401, f"got {r.status_code}")

    # Valid key
    r = requests.get(f"{base}/quota/", headers={"Authorization": f"Bearer {key}"})
    check("valid key → 200", r.status_code == 200, f"got {r.status_code}")


def test_quota(base: str, headers: dict) -> None:
    section("GET /quota/")
    r = requests.get(f"{base}/quota/", headers=headers)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    j = r.json()
    check("has limit", "limit" in j)
    check("has used", "used" in j)
    check("has remaining", "remaining" in j)
    check("has resetsInSeconds", "resetsInSeconds" in j)
    check("limit > 0", j.get("limit", 0) > 0)
    check("remaining = limit−used", j.get("remaining") == j.get("limit", 0) - j.get("used", 0))


def build_selected_method_jobs(methods: set) -> list[dict]:
    """
    Build one submit job per selected method/target:
      - all selected kcat-capable methods as prediction_type=kcat
      - all selected Km-capable methods as prediction_type=Km
      - all selected kcat/Km-capable methods as prediction_type=kcat/Km
    """
    jobs: list[dict] = []
    for kcat_method in selected_kcat_methods(methods):
        jobs.append(
            {
                "prediction_type": "kcat",
                "kcat_method": kcat_method,
                "km_method": None,
                "kcat_km_method": None,
                "csv_content": choose_submit_csv(
                    prediction_type="kcat",
                    kcat_method=kcat_method,
                ),
                "label": f"kcat/{kcat_method}",
            }
        )
    for km_method in selected_km_methods(methods):
        jobs.append(
            {
                "prediction_type": "Km",
                "kcat_method": None,
                "km_method": km_method,
                "kcat_km_method": None,
                "csv_content": choose_submit_csv(
                    prediction_type="Km",
                    km_method=km_method,
                ),
                "label": f"Km/{km_method}",
            }
        )
    for ratio_method in selected_kcat_km_methods(methods):
        jobs.append(
            {
                "prediction_type": "kcat/Km",
                "kcat_method": None,
                "km_method": None,
                "kcat_km_method": ratio_method,
                "csv_content": SINGLE_SUBSTRATE_CSV,
                "label": f"kcat/Km/{ratio_method}",
            }
        )
    return jobs


def test_submit_selected_methods(base: str, headers: dict, methods: set) -> list[dict]:
    """
    Submit one job for every selected method/prediction type combination.
    Returns a list of tracked submitted jobs for later polling/result checks.
    """
    specs = build_selected_method_jobs(methods)
    if not specs:
        print("\n  (skipping method submits — no supported methods selected)")
        return []

    section("POST /submit/ — valid CSV upload for every selected method")
    submitted_jobs: list[dict] = []
    for spec in specs:
        label = spec["label"]
        r = submit(
            base,
            headers,
            spec["csv_content"],
            spec["prediction_type"],
            kcat_method=spec["kcat_method"],
            km_method=spec["km_method"],
            kcat_km_method=spec.get("kcat_km_method"),
            include_similarity_columns=spec.get("include_similarity_columns"),
        )
        ok = check(f"[{label}] status 201", r.status_code == 201, f"got {r.status_code}")
        if not ok:
            continue
        j = r.json()
        check(f"[{label}] has jobId", "jobId" in j)
        check(f"[{label}] status=Pending", j.get("status") == "Pending")
        check(f"[{label}] has statusUrl", "statusUrl" in j)
        check(f"[{label}] has resultUrl", "resultUrl" in j)
        check(f"[{label}] has quota", "quota" in j)
        q = j.get("quota", {})
        check(f"[{label}] quota has remaining", "remaining" in q)
        check(f"[{label}] quota has limit", "limit" in q)
        if "jobId" not in j:
            continue
        submitted = {**spec, "job": j, "job_id": j["jobId"]}
        submitted_jobs.append(submitted)
        print(f"         → Submitted {label}: jobId={j['jobId']}")
    return submitted_jobs


def test_submit_kcat_similarity_toggle_off(base: str, headers: dict, methods: set) -> dict | None:
    """
    Submit one explicit kcat job with includeSimilarityColumns=false and
    validate in result checks that similarity columns are absent.
    """
    kcat_method = next((m for m in selected_kcat_methods(methods)), None)
    if kcat_method is None:
        print("\n  (skipping similarity toggle-off submit test — no kcat method selected)")
        return None

    csv_content = choose_submit_csv(
        prediction_type="kcat",
        kcat_method=kcat_method,
    )
    label = f"kcat/{kcat_method}/simoff"
    section(f"POST /submit/ — kcat submit with includeSimilarityColumns=false [{kcat_method}]")

    r = submit(
        base,
        headers,
        csv_content,
        "kcat",
        kcat_method=kcat_method,
        include_similarity_columns=False,
    )
    if not check(f"[{label}] status 201", r.status_code == 201, f"got {r.status_code}"):
        return None

    j = r.json()
    check(f"[{label}] has jobId", "jobId" in j)
    check(f"[{label}] status=Pending", j.get("status") == "Pending")
    if "jobId" not in j:
        return None

    print(f"         → Submitted {label}: jobId={j['jobId']}")
    return {
        "prediction_type": "kcat",
        "kcat_method": kcat_method,
        "km_method": None,
        "kcat_km_method": None,
        "include_similarity_columns": False,
        "label": label,
        "job": j,
        "job_id": j["jobId"],
    }


def test_submit_json_body(base: str, headers: dict, methods: set) -> dict | None:
    kcat_method = next(
        (m for m in KCAT_METHOD_IDS if m.lower() in methods and m != "TurNup"), None
    )  # TurNup needs full-reaction format; pick any other kcat method
    if kcat_method is None:
        print("\n  (skipping JSON body submit test — no non-TurNup kcat method selected)")
        return None
    label = f"json/kcat/{kcat_method}"
    section(f"POST /submit/ — JSON body (inline data, no CSV file) [{kcat_method}]")
    json_headers = {**headers, "Content-Type": "application/json"}
    payload = {
        "targets": ["kcat"],
        "methods": {"kcat": kcat_method},
        "handleLongSequences": "truncate",
        "useExperimental": False,
        "includeSimilarityColumns": False,
        "data": [
            {
                "Protein Sequence": (
                    "MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAEL"
                    "IEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKS"
                    "GHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYF"
                    "YPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIER"
                    "WPGMPE"
                ),
                "Substrate": "CC(=O)O.O" if kcat_method == "CatPred" else "CC(=O)O",
            },
            {
                "Protein Sequence": (
                    "MEMLEEHRCFEGWQQRWRHDSSTLNCPMTFSIFLPPPRDHTPPPVLYWLSGLTCNDENFTTK"
                    "AGAQRVAAELGIVLVMPDTSPRGEKVANDDGYDLGQGAGFYLNATQPPWATHYRMYDYLRDEL"
                    "PALVQSQFNVSDRCAISGHSMGGHGALIMALKNPGKYTSVSAFAPIVNPCSVPWGIKAFSSYL"
                    "GEDKNAWLEWDSCALMYASNAQDAIPTLIAQGDNDQFLADQLQPAVLAEAARQKAWPMTLRIQ"
                    "PGYDHSYYFIASFIEDHLRFHAQYLLK"
                ),
                "Substrate": "C1CCCCC1.O" if kcat_method == "CatPred" else "C1CCCCC1",
            },
        ],
    }
    r = requests.post(f"{base}/submit/", headers=json_headers, json=payload)
    if not check(f"[{label}] status 201", r.status_code == 201, f"got {r.status_code}"):
        return None
    j = r.json()
    check(f"[{label}] has jobId", "jobId" in j)
    check(f"[{label}] status=Pending", j.get("status") == "Pending")
    if "jobId" not in j:
        return None
    print(f"         → Submitted {label}: jobId={j['jobId']}")
    return {
        "prediction_type": "kcat",
        "kcat_method": kcat_method,
        "km_method": None,
        "kcat_km_method": None,
        "include_similarity_columns": False,
        "label": label,
        "job": j,
        "job_id": j["jobId"],
    }


def test_submit_errors(base: str, headers: dict) -> None:
    section("POST /submit/ — validation error cases (all should fail cleanly)")

    # No file attached
    r = requests.post(
        f"{base}/submit/",
        headers=headers,
        data={
            "targets": '["kcat"]',
            "methods": '{"kcat":"DLKcat"}',
            "handleLongSequences": "truncate",
        },
    )
    check("no file → 400", r.status_code == 400, f"got {r.status_code}")
    check("error key present", "error" in r.json())

    # File with wrong extension (.txt instead of .csv)
    r = requests.post(
        f"{base}/submit/",
        headers=headers,
        files={"file": ("input.txt", io.BytesIO(NOT_A_CSV_BYTES), "text/plain")},
        data={
            "targets": '["kcat"]',
            "methods": '{"kcat":"DLKcat"}',
            "handleLongSequences": "truncate",
        },
    )
    check("non-.csv extension → 400", r.status_code == 400, f"got {r.status_code}")

    # Invalid target selection
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "invalid_type")
    check("bad target list → 400", r.status_code == 400, f"got {r.status_code}")

    # Invalid kcat method for target
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "kcat", kcat_method="NOTAMETHOD")
    check("bad kcat method → 400", r.status_code == 400, f"got {r.status_code}")

    # Valid predictionType but wrong method for it (KinForm-L is not a Km method)
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "Km", km_method="KinForm-L")
    check("KinForm-L for Km → 400", r.status_code == 400, f"got {r.status_code}")

    # handleLongSequences with invalid value
    r = submit(
        base,
        headers,
        SINGLE_SUBSTRATE_CSV,
        "kcat",
        kcat_method="DLKcat",
        handle_long="invalid_value",
    )
    check("bad handleLongSeq → 400", r.status_code == 400, f"got {r.status_code}")

    # CSV missing required "Substrate" column
    r = submit(base, headers, MISSING_COLUMN_CSV, "kcat", kcat_method="DLKcat")
    check("missing column → 400", r.status_code == 400, f"got {r.status_code}")

    # TurNup with single-substrate CSV (missing Substrates + Products columns)
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "kcat", kcat_method="TurNup")
    check("TurNup+wrong CSV → 400", r.status_code == 400, f"got {r.status_code}")

    # CatPred kcat now uses the semicolon Substrates schema; scalar Substrate
    # is not advertised, while legacy dot-joined Substrate remains supported.
    r = submit(base, headers, SINGLE_SUBSTRATE_CSV, "kcat", kcat_method="CatPred")
    check("CatPred kcat+scalar Substrate → 400", r.status_code == 400, f"got {r.status_code}")

    # Dot-joined input is compatibility-only for CatPred kcat. Expanded pair
    # methods must use semicolon-separated Substrates to get max/array behavior.
    r = submit(base, headers, CATPRED_DOTJOIN_SUBSTRATE_CSV, "kcat", kcat_method="DLKcat")
    check("pair method+legacy dot CSV → 400", r.status_code == 400, f"got {r.status_code}")

    invalid_product_csv = (
        "Protein Sequence,Substrates,Products\n"
        f"{_multi_substrate_test_sequence()},CCO;O,CC=O;not-a-molecule\n"
    )
    r = submit(base, headers, invalid_product_csv, "kcat", kcat_method="DLKcat")
    check("pair method+invalid Products → 400", r.status_code == 400, f"got {r.status_code}")

    # JSON body with empty data array
    r = requests.post(
        f"{base}/submit/",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "targets": ["kcat"],
            "methods": {"kcat": "DLKcat"},
            "handleLongSequences": "truncate",
            "data": [],
        },
    )
    check("empty JSON data → 400", r.status_code == 400, f"got {r.status_code}")

    # JSON body exceeding 10,000 row limit
    big_data = [{"Protein Sequence": "M" * 10, "Substrate": "C"} for _ in range(10_001)]
    r = requests.post(
        f"{base}/submit/",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "targets": ["kcat"],
            "methods": {"kcat": "DLKcat"},
            "handleLongSequences": "truncate",
            "data": big_data,
        },
    )
    check("10001-row JSON body → 400", r.status_code == 400, f"got {r.status_code}")


def test_status(base: str, headers: dict, submitted_jobs: list[dict]) -> None:
    if not submitted_jobs:
        return
    section("GET /status/<jobId>/ — job status polling")
    sample = submitted_jobs[0]
    job_id = sample["job_id"]
    label = sample["label"]

    # Valid status request
    r = requests.get(f"{base}/status/{job_id}/", headers=headers)
    check(f"[{label}] status 200", r.status_code == 200, f"got {r.status_code}")
    j = r.json()
    check(f"[{label}] jobId matches", j.get("jobId") == job_id)
    check(f"[{label}] status field present", "status" in j)
    check(
        f"[{label}] status is known value",
        j.get("status") in {"Pending", "Processing", "Completed", "Failed"},
    )
    check(f"[{label}] submittedAt present", "submittedAt" in j)
    check(f"[{label}] elapsedSeconds ≥ 0", j.get("elapsedSeconds", -1) >= 0)
    check(f"[{label}] progress present", "progress" in j)
    prog = j.get("progress", {})
    check(f"[{label}] progress.moleculesTotal", "moleculesTotal" in prog)
    check(f"[{label}] progress.predictionsTotal", "predictionsTotal" in prog)

    # Non-existent job
    r = requests.get(f"{base}/status/NOTAREALIDXXX/", headers=headers)
    check("fake jobId → 404", r.status_code == 404, f"got {r.status_code}")
    check("error key present", "error" in r.json())


def test_result_not_ready(base: str, headers: dict, submitted_jobs: list[dict]) -> None:
    if not submitted_jobs:
        return
    """
    Unless the job completed instantly (unlikely in tests), result should
    return 409 Conflict because the job is still Pending/Processing.
    """
    section("GET /result/<jobId>/ — result before job completes")
    sample = submitted_jobs[0]
    job_id = sample["job_id"]
    label = sample["label"]

    r = requests.get(f"{base}/result/{job_id}/", headers=headers)
    # The job might have completed if a worker picked it up; handle both cases.
    if r.status_code == 200:
        check(f"[{label}] result available (job already done)", True)
    else:
        check(f"[{label}] pending job → 409", r.status_code == 409, f"got {r.status_code}")
        check(f"[{label}] error key present", "error" in r.json())

    # Non-existent job
    r = requests.get(f"{base}/result/NOTAREALIDXXX/", headers=headers)
    check("fake jobId → 404", r.status_code == 404, f"got {r.status_code}")


def wait_for_terminal_status(
    base: str,
    headers: dict,
    job_id: str,
    label: str,
    poll_timeout: int,
) -> str | None:
    """Poll /status/<jobId>/ until Completed/Failed, or return None on error/timeout."""
    print(f"         → Polling [{label}] job {job_id} until completion (timeout {poll_timeout}s)…")
    deadline = time.time() + poll_timeout
    interval = 5.0
    last_status = "Unknown"
    while time.time() < deadline:
        r = requests.get(f"{base}/status/{job_id}/", headers=headers)
        if r.status_code != 200:
            check(f"[{label}] status polling HTTP 200", False, f"got {r.status_code}")
            return None
        j = r.json()
        last_status = str(j.get("status", "Unknown"))
        if last_status in ("Completed", "Failed"):
            return last_status
        time.sleep(interval)
        interval = min(interval * 1.5, 30.0)  # back off up to 30 s

    check(
        f"[{label}] completed within timeout",
        False,
        f"timed out after {poll_timeout}s (last status={last_status})",
    )
    return None


def fetch_job_status(base: str, headers: dict, job_id: str, label: str) -> dict | None:
    """
    Fetch the latest /status payload for a job and validate HTTP status.
    """
    r = requests.get(f"{base}/status/{job_id}/", headers=headers)
    if not check(f"[{label}] status fetch HTTP 200", r.status_code == 200, f"got {r.status_code}"):
        return None
    return r.json()


def validate_completed_result(
    base: str,
    headers: dict,
    job_id: str,
    label: str,
    submitted: dict,
) -> None:
    """Validate CSV and JSON result downloads for a completed job."""
    # CSV download
    r = requests.get(f"{base}/result/{job_id}/", headers=headers)
    check(f"[{label}] CSV status 200", r.status_code == 200, f"got {r.status_code}")
    check(
        f"[{label}] content-type is text/csv",
        "text/csv" in r.headers.get("Content-Type", ""),
        r.headers.get("Content-Type"),
    )
    check(f"[{label}] non-empty body", len(r.content) > 0)

    # JSON format
    r = requests.get(f"{base}/result/{job_id}/?format=json", headers=headers)
    check(f"[{label}] JSON status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code != 200:
        return
    j = r.json()
    check(f"[{label}] JSON has jobId", j.get("jobId") == job_id)
    check(f"[{label}] JSON has columns list", isinstance(j.get("columns"), list))
    check(f"[{label}] JSON has rowCount", isinstance(j.get("rowCount"), int))
    check(f"[{label}] JSON has data list", isinstance(j.get("data"), list))
    check(f"[{label}] rowCount matches data", j.get("rowCount") == len(j.get("data", [])))

    expected_similarity_cols = expected_kcat_similarity_columns(submitted)
    json_columns = j.get("columns", [])
    if expected_similarity_cols:
        mean_col, max_col = expected_similarity_cols
        check(f"[{label}] has mean similarity column", mean_col in json_columns)
        check(f"[{label}] has max similarity column", max_col in json_columns)
        for idx, row in enumerate(j.get("data", []), start=1):
            mean_value = row.get(mean_col)
            max_value = row.get(max_col)
            check(
                f"[{label}] row {idx} mean similarity valid",
                is_valid_similarity_cell(mean_value),
                f"value={mean_value!r}",
            )
            check(
                f"[{label}] row {idx} max similarity valid",
                is_valid_similarity_cell(max_value),
                f"value={max_value!r}",
            )
    else:
        check(
            f"[{label}] no mean similarity column",
            all(not str(col).startswith("mean similarity to ") for col in json_columns),
        )
        check(
            f"[{label}] no max similarity column",
            all(not str(col).startswith("max similarity to ") for col in json_columns),
        )

    if j.get("data"):
        first_row = j["data"][0]
        check(f"[{label}] row has Protein Sequence", "Protein Sequence" in first_row)
        submitted_header = str(submitted.get("csv_content", "")).lstrip().splitlines()[0]
        if "Substrates" in submitted_header:
            check(f"[{label}] row has Substrates", "Substrates" in first_row)
            if "Products" in submitted_header:
                check(f"[{label}] row has Products", "Products" in first_row)
        else:
            check(f"[{label}] row has Substrate", "Substrate" in first_row)


def test_result_completed(
    base: str, headers: dict, submitted_jobs: list[dict], poll_timeout: int = 1000
) -> None:
    """
    Poll every submitted method job until it reaches a terminal state, then
    validate its result in CSV and JSON formats.

    poll_timeout — seconds to wait per job (default 1000 s / 16.67 min).
    """
    if not submitted_jobs:
        return

    section("GET /result/<jobId>/ — downloading completed results for all selected methods")
    for submitted in submitted_jobs:
        job_id = submitted["job_id"]
        label = submitted["label"]
        final_status = wait_for_terminal_status(base, headers, job_id, label, poll_timeout)
        if final_status is None:
            continue
        check(
            f"[{label}] final status is Completed",
            final_status == "Completed",
            f"got {final_status}",
        )
        if final_status != "Completed":
            continue
        validate_completed_result(base, headers, job_id, label, submitted)


def test_validate(base: str, headers: dict) -> None:
    """
    POST /validate/ — input validation without running predictions.

    Covers:
      - Valid CSV → all-clear response
      - Invalid content CSV → invalid substrates + proteins detected
      - JSON body submission
      - Auth errors
      - Input errors (no file, bad extension, missing columns)
    """
    section("POST /validate/ — input validation (no similarity)")

    # ── Valid CSV — expect no issues ─────────────────────────────────────────
    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": csv_file(SINGLE_SUBSTRATE_CSV)},
        data={"runSimilarity": "false"},
    )
    if check("valid CSV → 200", r.status_code == 200, f"got {r.status_code}"):
        j = r.json()
        check("has rowCount", "rowCount" in j)
        check("rowCount = 3", j.get("rowCount") == 3)
        check("has invalidSubstrates", "invalidSubstrates" in j)
        check("has invalidProteins", "invalidProteins" in j)
        check("has lengthViolations", "lengthViolations" in j)
        check("has similarity key", "similarity" in j)
        check("similarity is null", j.get("similarity") is None)
        check("no invalid substrates", len(j.get("invalidSubstrates", [1])) == 0)
        check("no invalid proteins", len(j.get("invalidProteins", [1])) == 0)

    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": csv_file(SUBSTRATE_LIST_CSV)},
        data={"runSimilarity": "false"},
    )
    if check("Substrates list CSV → 200", r.status_code == 200, f"got {r.status_code}"):
        check(
            "Substrates list is chemically valid",
            len(r.json().get("invalidSubstrates", [1])) == 0,
        )

    invalid_product_csv = (
        "Protein Sequence,Substrates,Products\n"
        f"{_multi_substrate_test_sequence()},CCO;O,CC=O;not-a-molecule\n"
    )
    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": csv_file(invalid_product_csv)},
        data={"runSimilarity": "false"},
    )
    if check("full reaction with invalid product → 200", r.status_code == 200):
        invalid = r.json().get("invalidSubstrates", [])
        check(
            "invalid product is validated and reported",
            bool(invalid)
            and invalid[0].get("invalid_products", [{}])[0].get("position") == 2,
        )

    # ── Invalid content — one valid row + one invalid row ────────────────────
    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": csv_file(INVALID_CONTENT_CSV)},
        data={"runSimilarity": "false"},
    )
    if check("invalid CSV → 200", r.status_code == 200, f"got {r.status_code}"):
        j = r.json()
        check("rowCount = 2", j.get("rowCount") == 2)
        check("detects invalid substrate", len(j.get("invalidSubstrates", [])) > 0)
        check("detects invalid protein", len(j.get("invalidProteins", [])) > 0)

    # ── JSON body ────────────────────────────────────────────────────────────
    r = requests.post(
        f"{base}/validate/",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "data": [
                {
                    "Protein Sequence": (
                        "MSTAIVTNVKHFGGMGSALRLSEAGHTVACHDESFKQKDELEAFAETYPQLKPMSEQEPAEL"
                        "IEAVTSAYGQVDVLVSNDIFAPEFQPIDKYAVEDYRGAVEALQIRPFALVNAVASQMKKRKS"
                        "GHIIFITSATPFGPWKELSTYTSARAGACTLANALSKELGEYNIPVFAIGPNYLHSEDSPYF"
                        "YPTEPWKTNPEHVAHVKKVTALQRLGTQKELGELVAFLASGSCDYLTGQVFWLAGGFPMIER"
                        "WPGMPE"
                    ),
                    "Substrate": "CC(=O)O",
                },
            ],
            "runSimilarity": False,
        },
    )
    if check("JSON body → 200", r.status_code == 200, f"got {r.status_code}"):
        j = r.json()
        check("JSON rowCount = 1", j.get("rowCount") == 1)
        check("JSON similarity = null", j.get("similarity") is None)

    # ── Auth errors ──────────────────────────────────────────────────────────
    r = requests.post(f"{base}/validate/", files={"file": csv_file(SINGLE_SUBSTRATE_CSV)})
    check("no auth → 401", r.status_code == 401, f"got {r.status_code}")

    # ── Input errors ─────────────────────────────────────────────────────────
    r = requests.post(f"{base}/validate/", headers=headers)
    check("no file → 400", r.status_code == 400, f"got {r.status_code}")

    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": ("input.txt", io.BytesIO(NOT_A_CSV_BYTES), "text/plain")},
    )
    check("non-.csv extension → 400", r.status_code == 400, f"got {r.status_code}")

    r = requests.post(
        f"{base}/validate/",
        headers=headers,
        files={"file": csv_file(MISSING_COLUMN_CSV)},
    )
    check("missing columns → 400", r.status_code == 400, f"got {r.status_code}")

    r = requests.post(
        f"{base}/validate/",
        headers={**headers, "Content-Type": "application/json"},
        json={"data": [], "runSimilarity": False},
    )
    check("empty JSON data → 400", r.status_code == 400, f"got {r.status_code}")


def test_validate_similarity(base: str, headers: dict) -> None:
    """
    POST /validate/ with runSimilarity=true.

    This is a blocking synchronous call — the HTTP response is only returned
    once MMseqs2 has finished analysing all sequences.  Allow up to 10 minutes.
    Skipped gracefully if mmseqs2 is unavailable on this server.
    """
    section("POST /validate/ — with runSimilarity=true (blocking, may be slow)")
    print("         → sending request and waiting for MMseqs2 to complete…")

    try:
        r = requests.post(
            f"{base}/validate/",
            headers=headers,
            files={"file": csv_file(SINGLE_SUBSTRATE_CSV)},
            data={"runSimilarity": "true"},
            timeout=600,
        )
    except requests.exceptions.Timeout:
        print("         (skipped — request timed out after 10 minutes)")
        return

    if not check("status 200", r.status_code == 200, f"got {r.status_code}"):
        return

    j = r.json()
    sim = j.get("similarity")
    check("similarity is not null", sim is not None)
    if sim is None:
        return

    # The similarity service may return {"error": "..."} if mmseqs2 is unavailable
    if "error" in sim:
        check("similarity succeeded (mmseqs2 available)", False, sim["error"])
        return

    check("similarity is a dict", isinstance(sim, dict))
    check("similarity has method keys", len(sim) > 0)

    # Verify structure of at least one method's result
    for method_name, method_data in sim.items():
        if not isinstance(method_data, dict):
            continue
        check(f"{method_name} has histogram_max", "histogram_max" in method_data)
        check(f"{method_name} has histogram_mean", "histogram_mean" in method_data)
        check(f"{method_name} has average_max_similarity", "average_max_similarity" in method_data)
        check(f"{method_name} has count_max", "count_max" in method_data)
        hist = method_data.get("histogram_max", [])
        check(f"{method_name} histogram has 101 bins", len(hist) == 101)
        break


def test_gpu_status(base: str) -> None:
    """GET /api/v1/gpu/status/ — GPU embed-service health snapshot."""
    section("GET /gpu/status/ — GPU embed-service reachability")
    r = requests.get(f"{base}/gpu/status/")
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code != 200:
        return
    j = r.json()
    check("has configured", "configured" in j)
    check("has online", "online" in j)
    check("has mode", "mode" in j)
    check("mode is gpu or cpu", j.get("mode") in {"gpu", "cpu"})
    if j.get("online"):
        check("mode=gpu when online", j.get("mode") == "gpu")
        print(f"         → GPU online: {j.get('gpu_name') or 'unknown'}"
              f"  free={j.get('free_vram_gb')}GB / {j.get('total_vram_gb')}GB")
    else:
        print(f"         → GPU offline (reason: {j.get('reason', 'unknown')})")


def test_gpu_methods(base: str, headers: dict, methods: set, poll_timeout: int) -> None:
    """
    If GPU is online, submit prediction jobs for all selected GPU-capable methods
    and verify they complete successfully, confirming the GPU pipeline is working.
    Skipped gracefully when GPU is offline or not configured.
    """
    section("GPU Pipeline — submit uncached GPU-capable methods and verify GPU execution")

    r = requests.get(f"{base}/gpu/status/")
    if r.status_code != 200:
        print(f"  (skipping — GET /gpu/status/ returned {r.status_code})")
        return
    gpu_status = r.json()
    if not gpu_status.get("configured"):
        print("  (skipping — GPU service not configured on this server)")
        return
    if not gpu_status.get("online"):
        print(f"  (skipping — GPU is offline: {gpu_status.get('reason', 'unknown')})")
        return

    print(f"  GPU online: {gpu_status.get('gpu_name') or 'unknown'}")

    specs: list[dict] = []
    for m in GPU_SUPPORTED_KCAT_METHOD_IDS:
        if m.lower() not in methods:
            continue
        specs.append({
            "prediction_type": "kcat",
            "kcat_method": m,
            "km_method": None,
            "kcat_km_method": None,
            "csv_content": build_gpu_uncached_csv(f"gpu/kcat/{m}"),
            "label": f"gpu/kcat/{m}",
        })
    for m in GPU_SUPPORTED_KM_METHOD_IDS:
        if m.lower() not in methods:
            continue
        specs.append({
            "prediction_type": "Km",
            "kcat_method": None,
            "km_method": m,
            "kcat_km_method": None,
            "csv_content": build_gpu_uncached_csv(f"gpu/Km/{m}"),
            "label": f"gpu/Km/{m}",
        })
    for m in GPU_SUPPORTED_KCAT_KM_METHOD_IDS:
        if m.lower() not in methods:
            continue
        specs.append({
            "prediction_type": "kcat/Km",
            "kcat_method": None,
            "km_method": None,
            "kcat_km_method": m,
            "csv_content": build_gpu_uncached_csv(f"gpu/kcat_Km/{m}"),
            "label": f"gpu/kcat_Km/{m}",
        })

    if not specs:
        print("  (no GPU-supported methods are selected — nothing to submit)")
        return

    submitted_gpu_jobs: list[dict] = []
    for spec in specs:
        label = spec["label"]
        r = submit(
            base, headers, spec["csv_content"], spec["prediction_type"],
            kcat_method=spec["kcat_method"],
            km_method=spec["km_method"],
            kcat_km_method=spec.get("kcat_km_method"),
        )
        ok = check(f"[{label}] status 201", r.status_code == 201, f"got {r.status_code}")
        if not ok:
            continue
        j = r.json()
        check(f"[{label}] has jobId", "jobId" in j)
        if "jobId" not in j:
            continue
        submitted_gpu_jobs.append({**spec, "job": j, "job_id": j["jobId"]})
        print(f"         → Submitted {label}: jobId={j['jobId']}")

    for submitted in submitted_gpu_jobs:
        job_id = submitted["job_id"]
        label = submitted["label"]
        final_status = wait_for_terminal_status(base, headers, job_id, label, poll_timeout)
        if final_status is None:
            continue
        check(
            f"[{label}] final status is Completed",
            final_status == "Completed",
            f"got {final_status}",
        )
        if final_status == "Completed":
            status_payload = fetch_job_status(base, headers, job_id, label)
            if status_payload is not None:
                gpu_precompute = status_payload.get("gpuPrecompute")
                check(
                    f"[{label}] status exposes gpuPrecompute",
                    isinstance(gpu_precompute, dict),
                )
                if isinstance(gpu_precompute, dict):
                    used_gpu = bool(
                        gpu_precompute.get("usedGpu", gpu_precompute.get("used_gpu", False))
                    )
                    check(
                        f"[{label}] gpuPrecompute.attempted=true",
                        bool(gpu_precompute.get("attempted")),
                    )
                    check(f"[{label}] gpuPrecompute.usedGpu=true", used_gpu)
                    check(
                        f"[{label}] gpuPrecompute.completed=true",
                        bool(gpu_precompute.get("completed")),
                    )
                    check(
                        f"[{label}] gpuPrecompute.failed=false",
                        not bool(gpu_precompute.get("failed")),
                    )
                    check(
                        f"[{label}] gpuPrecompute.reason=done (uncached GPU work executed)",
                        str(gpu_precompute.get("reason", "")).strip() == "done",
                        f"got {gpu_precompute.get('reason')!r}",
                    )
            validate_completed_result(base, headers, job_id, label, submitted)


# ---------------------------------------------------------------------------
# Opt-in multi-substrate value verification
# ---------------------------------------------------------------------------


def _multi_substrate_test_sequence() -> str:
    return SINGLE_SUBSTRATE_CSV.strip().splitlines()[1].rsplit(",", 1)[0]


def _multi_substrate_test_csv(kind: str) -> str:
    sequence = _multi_substrate_test_sequence()
    if kind == "single":
        return (
            "Protein Sequence,Substrate\n"
            f"{sequence},CCO\n"
            f"{sequence},C1CCCCC1\n"
        )
    if kind == "multi":
        return "Protein Sequence,Substrates\n" f"{sequence},CCO;C1CCCCC1\n"
    if kind == "full":
        return (
            "Protein Sequence,Substrates,Products\n"
            f"{sequence},CCO;C1CCCCC1,CC=O;O\n"
        )
    if kind == "legacy_catpred":
        return "Protein Sequence,Substrate\n" f"{sequence},CCO.C1CCCCC1\n"
    raise ValueError(f"Unknown comparison CSV kind: {kind}")


def _fetch_method_metadata(base: str) -> dict[str, dict]:
    response = requests.get(f"{base}/methods/")
    if not check("[multi-sub] methods metadata HTTP 200", response.status_code == 200):
        return {}
    flat: dict[str, dict] = {}
    for entries in response.json().get("methods", {}).values():
        for entry in entries:
            flat[entry["id"]] = entry
    return flat


def _submit_wait_json_result(
    base: str,
    headers: dict,
    *,
    label: str,
    csv_content: str,
    targets: list[str],
    methods: dict[str, str],
    poll_timeout: int,
) -> dict | None:
    response = submit_targets(
        base,
        headers,
        csv_content,
        targets,
        methods,
        include_similarity_columns=False,
    )
    if not check(
        f"[{label}] submit 201",
        response.status_code == 201,
        f"got {response.status_code}",
    ):
        return None
    job_id = response.json().get("jobId")
    if not check(f"[{label}] has jobId", bool(job_id)):
        return None
    final_status = wait_for_terminal_status(base, headers, job_id, label, poll_timeout)
    if not check(f"[{label}] completed", final_status == "Completed", f"got {final_status}"):
        return None
    result_response = requests.get(f"{base}/result/{job_id}/?format=json", headers=headers)
    if not check(
        f"[{label}] result JSON 200",
        result_response.status_code == 200,
        f"got {result_response.status_code}",
    ):
        return None
    return result_response.json()


def _prediction_column(result: dict, target: str) -> str | None:
    columns = result.get("columns", [])
    source_column = f"Source {target}"
    try:
        source_position = columns.index(source_column)
    except ValueError:
        return None
    return columns[source_position - 1] if source_position > 0 else None


def _finite_api_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_cell_array(value) -> list | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _numbers_match(left, right, *, rel_tol: float = 1e-7, abs_tol: float = 1e-12) -> bool:
    left_number = _finite_api_number(left)
    right_number = _finite_api_number(right)
    if left_number is None or right_number is None:
        return left_number is right_number
    return math.isclose(left_number, right_number, rel_tol=rel_tol, abs_tol=abs_tol)


def _assert_preserved_reaction_input(label: str, result: dict, *, full: bool) -> None:
    data = result.get("data", [])
    check(f"[{label}] one output row", result.get("rowCount") == 1 and len(data) == 1)
    if not data:
        return
    row = data[0]
    check(f"[{label}] Substrates preserved", row.get("Substrates") == "CCO;C1CCCCC1")
    if full:
        check(f"[{label}] Products preserved", row.get("Products") == "CC=O;O")


def _compare_expanded_target(
    *,
    label: str,
    target: str,
    control: dict,
    multi: dict,
    full: dict,
) -> None:
    control_column = _prediction_column(control, target)
    aggregate_column = _prediction_column(multi, target)
    full_column = _prediction_column(full, target)
    if not check(
        f"[{label}] {target} prediction columns found",
        bool(control_column and aggregate_column and full_column),
    ):
        return

    control_rows = control.get("data", [])
    multi_rows = multi.get("data", [])
    full_rows = full.get("data", [])
    if not check(
        f"[{label}] comparison rows present",
        len(control_rows) == 2 and len(multi_rows) == 1 and len(full_rows) == 1,
    ):
        return

    controls = [_finite_api_number(row.get(control_column)) for row in control_rows]
    aggregate_value = multi_rows[0].get(aggregate_column)
    full_value = full_rows[0].get(full_column)

    if target == "kcat":
        successful = [(index, value) for index, value in enumerate(controls) if value is not None]
        if not check(f"[{label}] kcat control has a successful prediction", bool(successful)):
            return
        winner_index, expected = max(successful, key=lambda item: item[1])
        check(f"[{label}] kcat equals control maximum", _numbers_match(aggregate_value, expected))
        check(f"[{label}] full kcat equals multi kcat", _numbers_match(full_value, aggregate_value))

        source_column = f"Source {target}"
        check(
            f"[{label}] winning source propagated",
            multi_rows[0].get(source_column) == control_rows[winner_index].get(source_column),
        )
        extra_items = _json_cell_array(multi_rows[0].get(f"Extra Info {target}"))
        if check(f"[{label}] kcat Extra Info is an ordered array", isinstance(extra_items, list)):
            check(f"[{label}] kcat Extra Info has both substrates", len(extra_items) == 2)
            selected = [index for index, item in enumerate(extra_items) if item.get("selected")]
            check(f"[{label}] kcat marks the earliest maximum", selected == [winner_index])
            check(
                f"[{label}] kcat Extra Info preserves substrate order",
                [item.get("substrate") for item in extra_items] == ["CCO", "C1CCCCC1"],
            )
            check(
                f"[{label}] kcat Extra Info values match scalar controls",
                len(extra_items) == len(controls)
                and all(
                    _numbers_match(item.get("prediction"), controls[index])
                    for index, item in enumerate(extra_items)
                ),
            )
        return

    aggregate_array = _json_cell_array(aggregate_value)
    full_array = _json_cell_array(full_value)
    if not check(f"[{label}] {target} is a JSON array", isinstance(aggregate_array, list)):
        return
    check(f"[{label}] {target} array length/order", len(aggregate_array) == 2)
    check(
        f"[{label}] {target} values match scalar controls",
        len(aggregate_array) == len(controls)
        and all(
            _numbers_match(value, controls[index])
            for index, value in enumerate(aggregate_array)
        ),
    )
    check(
        f"[{label}] full {target} equals multi {target}",
        isinstance(full_array, list)
        and len(full_array) == len(aggregate_array)
        and all(
            _numbers_match(value, aggregate_array[index])
            for index, value in enumerate(full_array)
        ),
    )
    extra_items = _json_cell_array(multi_rows[0].get(f"Extra Info {target}"))
    check(
        f"[{label}] {target} Extra Info preserves substrate order",
        isinstance(extra_items, list)
        and [item.get("substrate") for item in extra_items] == ["CCO", "C1CCCCC1"],
    )


def _test_catpred_native_multi(
    base: str,
    headers: dict,
    metadata: dict[str, dict],
    methods: set,
    poll_timeout: int,
) -> None:
    if "catpred" not in methods or "CatPred" not in metadata:
        return

    label = "multi-sub/CatPred-native"
    km_control = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/Km-control",
        csv_content=_multi_substrate_test_csv("single"),
        targets=["Km"],
        methods={"Km": "CatPred"},
        poll_timeout=poll_timeout,
    )
    legacy = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/legacy-dot",
        csv_content=_multi_substrate_test_csv("legacy_catpred"),
        targets=["kcat"],
        methods={"kcat": "CatPred"},
        poll_timeout=poll_timeout,
    )
    combined_methods = {"kcat": "CatPred", "Km": "CatPred"}
    multi = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/semicolon",
        csv_content=_multi_substrate_test_csv("multi"),
        targets=["kcat", "Km"],
        methods=combined_methods,
        poll_timeout=poll_timeout,
    )
    full = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/full",
        csv_content=_multi_substrate_test_csv("full"),
        targets=["kcat", "Km"],
        methods=combined_methods,
        poll_timeout=poll_timeout,
    )
    if not all((km_control, legacy, multi, full)):
        return

    _assert_preserved_reaction_input(f"{label}/semicolon", multi, full=False)
    _assert_preserved_reaction_input(f"{label}/full", full, full=True)

    legacy_column = _prediction_column(legacy, "kcat")
    multi_column = _prediction_column(multi, "kcat")
    full_column = _prediction_column(full, "kcat")
    legacy_value = legacy["data"][0].get(legacy_column) if legacy_column else None
    multi_value = multi["data"][0].get(multi_column) if multi_column else None
    full_value = full["data"][0].get(full_column) if full_column else None
    check(
        f"[{label}] semicolon kcat equals legacy dot kcat",
        _numbers_match(multi_value, legacy_value),
    )
    check(f"[{label}] full kcat equals semicolon kcat", _numbers_match(full_value, multi_value))
    check(
        f"[{label}] native kcat has no child-reduction Extra Info",
        multi["data"][0].get("Extra Info kcat") in (None, ""),
    )

    _compare_expanded_target(
        label=f"{label}/Km",
        target="Km",
        control=km_control,
        multi=multi,
        full=full,
    )


def test_multi_sub_for_single_sub_methods(
    base: str,
    headers: dict,
    methods: set,
    poll_timeout: int,
) -> None:
    """Compare expanded multi/full outputs against real scalar controls."""
    section("Multi-Substrate expansion — value-level verification")
    metadata = _fetch_method_metadata(base)
    if not metadata:
        return

    target_order = ["kcat", "Km", "kcat/Km"]
    for method_key, method_meta in sorted(metadata.items()):
        if method_key.lower() not in methods or method_key in {"CatPred", "TurNup"}:
            continue
        behaviors = method_meta.get("inputBehaviorByTarget", {})
        targets = [
            target
            for target in target_order
            if target in method_meta.get("supports", [])
            and behaviors.get(target, "expanded_pair") == "expanded_pair"
        ]
        if not targets:
            continue

        selected_methods = {target: method_key for target in targets}
        label = f"multi-sub/{method_key}/{'-'.join(targets)}"
        control = _submit_wait_json_result(
            base,
            headers,
            label=f"{label}/control",
            csv_content=_multi_substrate_test_csv("single"),
            targets=targets,
            methods=selected_methods,
            poll_timeout=poll_timeout,
        )
        multi = _submit_wait_json_result(
            base,
            headers,
            label=f"{label}/multi",
            csv_content=_multi_substrate_test_csv("multi"),
            targets=targets,
            methods=selected_methods,
            poll_timeout=poll_timeout,
        )
        full = _submit_wait_json_result(
            base,
            headers,
            label=f"{label}/full",
            csv_content=_multi_substrate_test_csv("full"),
            targets=targets,
            methods=selected_methods,
            poll_timeout=poll_timeout,
        )
        if not all((control, multi, full)):
            continue
        _assert_preserved_reaction_input(f"{label}/multi", multi, full=False)
        _assert_preserved_reaction_input(f"{label}/full", full, full=True)
        for target in targets:
            _compare_expanded_target(
                label=label,
                target=target,
                control=control,
                multi=multi,
                full=full,
            )

    _test_catpred_native_multi(base, headers, metadata, methods, poll_timeout)


# ---------------------------------------------------------------------------
# Opt-in multi-sequence value verification
# ---------------------------------------------------------------------------


def _multimer_test_sequences() -> list[str]:
    lines = SINGLE_SUBSTRATE_CSV.strip().splitlines()[1:3]
    return [line.rsplit(",", 1)[0] for line in lines]


def _multimer_sequence_value() -> str:
    return ";".join(_multimer_test_sequences())


def _multimer_csv(kind: str) -> str:
    seq_a, seq_b = _multimer_test_sequences()
    seqs = f"{seq_a};{seq_b}"
    if kind == "scalar_control":
        return (
            "Protein Sequence,Substrate\n"
            f"{seq_a},CCO\n"
            f"{seq_b},CCO\n"
        )
    if kind == "scalar_multimer":
        return "Protein Sequence,Substrate\n" f"{seqs},CCO\n"
    if kind == "multi_sub_control":
        return (
            "Protein Sequence,Substrate\n"
            f"{seq_a},CCO\n"
            f"{seq_a},C1CCCCC1\n"
            f"{seq_b},CCO\n"
            f"{seq_b},C1CCCCC1\n"
        )
    if kind == "multi_sub_multimer":
        return "Protein Sequence,Substrates\n" f"{seqs},CCO;C1CCCCC1\n"
    if kind == "native_multi_control":
        return (
            "Protein Sequence,Substrates\n"
            f"{seq_a},CCO;C1CCCCC1\n"
            f"{seq_b},CCO;C1CCCCC1\n"
        )
    if kind == "native_multi_multimer":
        return "Protein Sequence,Substrates\n" f"{seqs},CCO;C1CCCCC1\n"
    if kind == "full_control":
        return (
            "Protein Sequence,Substrates,Products\n"
            f"{seq_a},CCO;C1CCCCC1,CC=O;O\n"
            f"{seq_b},CCO;C1CCCCC1,CC=O;O\n"
        )
    if kind == "full_multimer":
        return (
            "Protein Sequence,Substrates,Products\n"
            f"{seqs},CCO;C1CCCCC1,CC=O;O\n"
        )
    raise ValueError(f"Unknown multimer CSV kind: {kind}")


def _aggregation_direction(target: str) -> str:
    return "min" if target == "Km" else "max"


def _select_expected_child(values: list[float | None], target: str) -> tuple[int, float] | None:
    successful = [(index, value) for index, value in enumerate(values) if value is not None]
    if not successful:
        return None
    if _aggregation_direction(target) == "min":
        return min(successful, key=lambda item: item[1])
    return max(successful, key=lambda item: item[1])


def _sequence_extra_items(row: dict, target: str) -> list | None:
    return _json_cell_array(row.get(f"Extra Info {target}"))


def _extra_selected_indices(extra_items: list | None) -> list[int]:
    if not isinstance(extra_items, list):
        return []
    return [
        index
        for index, item in enumerate(extra_items)
        if isinstance(item, dict) and item.get("selected")
    ]


def _extra_selected_substrate_pairs(extra_items: list | None) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    if not isinstance(extra_items, list):
        return selected
    for sequence_index, sequence_item in enumerate(extra_items):
        if not isinstance(sequence_item, dict):
            continue
        substrates = sequence_item.get("substrates")
        if not isinstance(substrates, list):
            continue
        for substrate_index, substrate_item in enumerate(substrates):
            if isinstance(substrate_item, dict) and substrate_item.get("selected"):
                selected.append((sequence_index, substrate_index))
    return selected


def _compare_multimer_scalar_target(
    *,
    label: str,
    target: str,
    control: dict,
    multimer: dict,
) -> None:
    control_column = _prediction_column(control, target)
    multimer_column = _prediction_column(multimer, target)
    if not check(
        f"[{label}] {target} prediction columns found",
        bool(control_column and multimer_column),
    ):
        return

    control_rows = control.get("data", [])
    multimer_rows = multimer.get("data", [])
    if not check(
        f"[{label}] scalar comparison rows present",
        len(control_rows) == 2 and len(multimer_rows) == 1,
    ):
        return

    controls = [_finite_api_number(row.get(control_column)) for row in control_rows]
    expected_child = _select_expected_child(controls, target)
    aggregate = multimer_rows[0].get(multimer_column)
    if expected_child is None:
        check(f"[{label}] {target} aggregate blank when all controls fail", aggregate in (None, ""))
        return

    winner_index, expected = expected_child
    check(
        f"[{label}] {target} aggregate matches {_aggregation_direction(target)} control",
        _numbers_match(aggregate, expected),
        f"controls={controls}, aggregate={aggregate}",
    )
    check(
        f"[{label}] {target} winning source propagated",
        multimer_rows[0].get(f"Source {target}")
        == control_rows[winner_index].get(f"Source {target}"),
    )
    extra_items = _sequence_extra_items(multimer_rows[0], target)
    if check(f"[{label}] {target} Extra Info is sequence array", isinstance(extra_items, list)):
        seqs = _multimer_test_sequences()
        check(f"[{label}] {target} Extra Info has both sequences", len(extra_items) == 2)
        check(
            f"[{label}] {target} Extra Info preserves sequence order",
            [item.get("sequence") for item in extra_items] == seqs,
        )
        check(
            f"[{label}] {target} Extra Info predictions match scalar controls",
            len(extra_items) == len(controls)
            and all(
                _numbers_match(item.get("prediction"), controls[index])
                for index, item in enumerate(extra_items)
            ),
        )
        check(
            f"[{label}] {target} Extra Info marks earliest winner",
            _extra_selected_indices(extra_items) == [winner_index],
        )


def _compare_multimer_multi_sub_target(
    *,
    label: str,
    target: str,
    control: dict,
    multimer: dict,
) -> None:
    control_column = _prediction_column(control, target)
    multimer_column = _prediction_column(multimer, target)
    if not check(
        f"[{label}] {target} multi-sub columns found",
        bool(control_column and multimer_column),
    ):
        return

    control_rows = control.get("data", [])
    multimer_rows = multimer.get("data", [])
    if not check(
        f"[{label}] multi-sub comparison rows present",
        len(control_rows) == 4 and len(multimer_rows) == 1,
    ):
        return

    controls = [_finite_api_number(row.get(control_column)) for row in control_rows]
    aggregate = multimer_rows[0].get(multimer_column)
    extra_items = _sequence_extra_items(multimer_rows[0], target)

    if target == "kcat":
        expected_child = _select_expected_child(controls, target)
        if expected_child is None:
            check(f"[{label}] kcat aggregate blank when all controls fail", aggregate in (None, ""))
            return
        winner_index, expected = expected_child
        winner_sequence = winner_index // 2
        winner_substrate = winner_index % 2
        check(
            f"[{label}] kcat max over sequence×substrate controls",
            _numbers_match(aggregate, expected),
            f"controls={controls}, aggregate={aggregate}",
        )
        check(
            f"[{label}] kcat winning source propagated",
            multimer_rows[0].get("Source kcat") == control_rows[winner_index].get("Source kcat"),
        )
        if check(
            f"[{label}] kcat Extra Info is sequence/substrate array",
            isinstance(extra_items, list),
        ):
            check(
                f"[{label}] kcat Extra Info marks winning sequence/substrate",
                _extra_selected_substrate_pairs(extra_items)
                == [(winner_sequence, winner_substrate)],
            )
        _assert_multimer_multi_sub_extra(label, target, extra_items, controls)
        return

    expected_values: list[float | None] = []
    expected_pairs: list[tuple[int, int]] = []
    for substrate_index in range(2):
        candidates = [
            controls[substrate_index],
            controls[2 + substrate_index],
        ]
        selected = _select_expected_child(candidates, target)
        if selected is None:
            expected_values.append(None)
            continue
        selected_sequence, expected_value = selected
        expected_values.append(expected_value)
        expected_pairs.append((selected_sequence, substrate_index))

    aggregate_array = _json_cell_array(aggregate)
    if not check(
        f"[{label}] {target} aggregate is substrate JSON array",
        isinstance(aggregate_array, list),
    ):
        return
    check(
        f"[{label}] {target} array matches per-substrate sequence reduction",
        len(aggregate_array) == len(expected_values)
        and all(
            _numbers_match(aggregate_array[index], expected_values[index])
            for index in range(len(expected_values))
        ),
        f"expected={expected_values}, aggregate={aggregate_array}",
    )
    _assert_multimer_multi_sub_extra(label, target, extra_items, controls)
    check(
        f"[{label}] {target} Extra Info marks per-substrate winners",
        sorted(_extra_selected_substrate_pairs(extra_items)) == sorted(expected_pairs),
    )


def _assert_multimer_multi_sub_extra(
    label: str,
    target: str,
    extra_items: list | None,
    controls: list[float | None],
) -> None:
    if not check(f"[{label}] {target} Extra Info is sequence array", isinstance(extra_items, list)):
        return
    seqs = _multimer_test_sequences()
    check(f"[{label}] {target} Extra Info has both sequences", len(extra_items) == 2)
    check(
        f"[{label}] {target} Extra Info preserves sequence order",
        [item.get("sequence") for item in extra_items] == seqs,
    )
    expected_substrates = ["CCO", "C1CCCCC1"]
    shape_ok = True
    values_ok = True
    for sequence_index, item in enumerate(extra_items):
        substrates = item.get("substrates") if isinstance(item, dict) else None
        if not isinstance(substrates, list) or len(substrates) != 2:
            shape_ok = False
            values_ok = False
            continue
        if [sub.get("substrate") for sub in substrates] != expected_substrates:
            shape_ok = False
        for substrate_index, substrate_item in enumerate(substrates):
            control_index = sequence_index * 2 + substrate_index
            if not _numbers_match(substrate_item.get("prediction"), controls[control_index]):
                values_ok = False
    check(f"[{label}] {target} Extra Info preserves substrate order", shape_ok)
    check(f"[{label}] {target} Extra Info values match controls", values_ok)


def _multimer_expanded_targets(method_meta: dict) -> list[str]:
    target_order = ["kcat", "Km", "kcat/Km"]
    behaviors = method_meta.get("inputBehaviorByTarget", {})
    return [
        target
        for target in target_order
        if target in method_meta.get("supports", [])
        and behaviors.get(target, "expanded_pair") == "expanded_pair"
    ]


def _test_multimer_for_expanded_method(
    base: str,
    headers: dict,
    method_key: str,
    method_meta: dict,
    poll_timeout: int,
) -> None:
    targets = _multimer_expanded_targets(method_meta)
    if not targets:
        return
    selected_methods = {target: method_key for target in targets}

    label = f"multimer/{method_key}/scalar"
    scalar_control = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/control",
        csv_content=_multimer_csv("scalar_control"),
        targets=targets,
        methods=selected_methods,
        poll_timeout=poll_timeout,
    )
    scalar_multimer = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/combined",
        csv_content=_multimer_csv("scalar_multimer"),
        targets=targets,
        methods=selected_methods,
        poll_timeout=poll_timeout,
    )
    if scalar_control and scalar_multimer:
        row = scalar_multimer.get("data", [{}])[0]
        check(
            f"[{label}] input Protein Sequence preserved",
            row.get("Protein Sequence") == _multimer_sequence_value(),
        )
        for target in targets:
            _compare_multimer_scalar_target(
                label=label,
                target=target,
                control=scalar_control,
                multimer=scalar_multimer,
            )

    edge_targets = targets
    edge_label = f"multimer/{method_key}/multi-sub"
    multi_control = _submit_wait_json_result(
        base,
        headers,
        label=f"{edge_label}/control-grid",
        csv_content=_multimer_csv("multi_sub_control"),
        targets=edge_targets,
        methods={target: method_key for target in edge_targets},
        poll_timeout=poll_timeout,
    )
    multi_multimer = _submit_wait_json_result(
        base,
        headers,
        label=f"{edge_label}/combined",
        csv_content=_multimer_csv("multi_sub_multimer"),
        targets=edge_targets,
        methods={target: method_key for target in edge_targets},
        poll_timeout=poll_timeout,
    )
    if multi_control and multi_multimer:
        row = multi_multimer.get("data", [{}])[0]
        check(
            f"[{edge_label}] input Protein Sequence preserved",
            row.get("Protein Sequence") == _multimer_sequence_value(),
        )
        check(f"[{edge_label}] Substrates preserved", row.get("Substrates") == "CCO;C1CCCCC1")
        for target in edge_targets:
            _compare_multimer_multi_sub_target(
                label=edge_label,
                target=target,
                control=multi_control,
                multimer=multi_multimer,
            )


def _test_multimer_catpred_native_kcat(
    base: str,
    headers: dict,
    method_meta: dict,
    poll_timeout: int,
) -> None:
    if "kcat" not in method_meta.get("supports", []):
        return
    if method_meta.get("inputBehaviorByTarget", {}).get("kcat") != "native_multi":
        return

    label = "multimer/CatPred/native-kcat"
    control = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/control",
        csv_content=_multimer_csv("native_multi_control"),
        targets=["kcat"],
        methods={"kcat": "CatPred"},
        poll_timeout=poll_timeout,
    )
    multimer = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/combined",
        csv_content=_multimer_csv("native_multi_multimer"),
        targets=["kcat"],
        methods={"kcat": "CatPred"},
        poll_timeout=poll_timeout,
    )
    if not (control and multimer):
        return
    _compare_multimer_scalar_target(
        label=label,
        target="kcat",
        control=control,
        multimer=multimer,
    )
    check(
        f"[{label}] native combined Substrates preserved",
        multimer["data"][0].get("Substrates") == "CCO;C1CCCCC1",
    )


def _test_multimer_full_reaction_method(
    base: str,
    headers: dict,
    method_key: str,
    method_meta: dict,
    poll_timeout: int,
) -> None:
    if method_meta.get("inputBehaviorByTarget", {}).get("kcat") != "native_full_reaction":
        return
    label = f"multimer/{method_key}/full-reaction"
    control = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/control",
        csv_content=_multimer_csv("full_control"),
        targets=["kcat"],
        methods={"kcat": method_key},
        poll_timeout=poll_timeout,
    )
    multimer = _submit_wait_json_result(
        base,
        headers,
        label=f"{label}/combined",
        csv_content=_multimer_csv("full_multimer"),
        targets=["kcat"],
        methods={"kcat": method_key},
        poll_timeout=poll_timeout,
    )
    if not (control and multimer):
        return
    _compare_multimer_scalar_target(
        label=label,
        target="kcat",
        control=control,
        multimer=multimer,
    )
    row = multimer.get("data", [{}])[0]
    check(f"[{label}] Substrates preserved", row.get("Substrates") == "CCO;C1CCCCC1")
    check(f"[{label}] Products preserved", row.get("Products") == "CC=O;O")


def test_multimer_for_methods(
    base: str,
    headers: dict,
    methods: set,
    poll_timeout: int,
) -> None:
    """Compare scalar controls with semicolon-separated protein sequence inputs."""
    section("Multi-Sequence protein input — value-level verification")
    metadata = _fetch_method_metadata(base)
    if not metadata:
        return

    for method_key, method_meta in sorted(metadata.items()):
        if method_key.lower() not in methods:
            continue
        if method_key == "CatPred":
            _test_multimer_catpred_native_kcat(base, headers, method_meta, poll_timeout)
            _test_multimer_for_expanded_method(base, headers, method_key, method_meta, poll_timeout)
            continue
        if any(
            behavior == "native_full_reaction"
            for behavior in method_meta.get("inputBehaviorByTarget", {}).values()
        ):
            _test_multimer_full_reaction_method(
                base,
                headers,
                method_key,
                method_meta,
                poll_timeout,
            )
            continue
        _test_multimer_for_expanded_method(base, headers, method_key, method_meta, poll_timeout)


def test_recon_xkg_transparency(
    base: str, headers: dict, methods: set, poll_timeout: int
) -> None:
    """
    The undocumented recon_xkg flag must be transparent to clients.

    Submitting the same input twice — once normally, once with recon_xkg=true —
    must complete and produce a result with an identical schema and row count.
    This holds whether or not the key is allowlisted: a non-allowlisted key is
    silently downgraded to a normal job, and an allowlisted key serves from the
    cache while keeping the output byte-compatible. When allowlisted, a second
    recon_xkg run should be served entirely from cache.
    """
    section("ReconXKG — recon_xkg=true is transparent (schema-identical)")

    kcat_method = next((m for m in selected_kcat_methods(methods)), None)
    if kcat_method is None:
        print("  (skipping — no kcat method selected)")
        return

    csv_content = choose_submit_csv(prediction_type="kcat", kcat_method=kcat_method)
    targets = ["kcat"]
    method_map = {"kcat": kcat_method}

    control = _submit_wait_json_result(
        base,
        headers,
        label=f"recon_xkg/{kcat_method}/control",
        csv_content=csv_content,
        targets=targets,
        methods=method_map,
        poll_timeout=poll_timeout,
    )

    recon = None
    response = submit_targets(
        base, headers, csv_content, targets, method_map,
        include_similarity_columns=False, recon_xkg=True,
    )
    label = f"recon_xkg/{kcat_method}/recon"
    if check(f"[{label}] submit 201", response.status_code == 201, f"got {response.status_code}"):
        submitted = response.json()
        job_id = submitted.get("jobId")
        check(f"[{label}] submit status is Pending", submitted.get("status") == "Pending")
        check(f"[{label}] submit omits inline result", "result" not in submitted)
        final_status = wait_for_terminal_status(base, headers, job_id, label, poll_timeout)
        if check(f"[{label}] completed", final_status == "Completed", f"got {final_status}"):
            result_response = requests.get(f"{base}/result/{job_id}/?format=json", headers=headers)
            if check(f"[{label}] result JSON 200", result_response.status_code == 200):
                recon = result_response.json()

    if not control or not recon:
        return

    check(
        f"[recon_xkg/{kcat_method}] identical columns",
        control.get("columns") == recon.get("columns"),
        f"{control.get('columns')} vs {recon.get('columns')}",
    )
    check(
        f"[recon_xkg/{kcat_method}] identical rowCount",
        control.get("rowCount") == recon.get("rowCount"),
    )


def test_wrong_methods(base: str, headers: dict) -> None:
    """Make sure method-not-allowed cases are handled (submit must be POST)."""
    section("HTTP method errors")
    # GET on submit/ should not work the same way as POST
    r = requests.get(f"{base}/submit/", headers=headers)
    # Django returns 405 for wrong method on csrf_exempt views, or our own 405.
    check("GET /submit/ → 4xx", r.status_code >= 400, f"got {r.status_code}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Open Kinetics Predictor API test suite")
    parser.add_argument(
        "--base", default=DEFAULT_BASE, help="Base URL including /api/v1 (default: %(default)s)"
    )
    parser.add_argument(
        "--key", default=DEFAULT_KEY, help="API Bearer token (default: hardcoded test key)"
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip the slow runSimilarity=true test (requires MMseqs2)",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=10000,
        metavar="SECONDS",
        help="Seconds to wait per submitted method job before marking "
        "that job as failed (default: 1000)",
    )
    parser.add_argument(
        "--extra-submit-variants",
        action="store_true",
        help=(
            "Also submit an additional success-path variant (JSON-body submit). "
            "This job is fully waited/validated too. "
            "Default is off for faster runs."
        ),
    )
    parser.add_argument(
        "--test-multi-sub-for-single-sub-methods",
        action="store_true",
        help=(
            "Run slow value-level multi-substrate tests for every selected pair method. "
            "Compares scalar controls with semicolon-list and full-reaction outputs."
        ),
    )
    parser.add_argument(
        "--only-multi-sub",
        action="store_true",
        help=(
            "Focused run: ONLY execute the multi-substrate value-level checks "
            "(same as --test-multi-sub-for-single-sub-methods) against the fixed/"
            "cached sequences, and skip every other test group. Implies that flag."
        ),
    )
    parser.add_argument(
        "--test-multimer",
        action="store_true",
        help=(
            "Run slow value-level multi-sequence protein tests for every selected method. "
            "Compares scalar single-sequence controls with semicolon-combined sequence rows."
        ),
    )
    parser.add_argument(
        "--only-multimer",
        action="store_true",
        help=(
            "Focused run: ONLY execute the multi-sequence protein value-level checks "
            "and skip every other test group. Implies --test-multimer."
        ),
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help=(
            "Skip the GPU pipeline tests (GET /gpu/status/ + GPU-capable method submission). "
            "These tests are skipped automatically when GPU is offline."
        ),
    )
    parser.add_argument(
        "--test-recon-xkg",
        action="store_true",
        help=(
            "Run the recon_xkg transparency test: submit the same input normally "
            "and with recon_xkg=true and confirm identical result schema/rowCount."
        ),
    )
    parser.add_argument(
        "--methods",
        default="all",
        metavar="METHOD[,METHOD…]",
        help=(
            "Comma-separated list of prediction methods to test. "
            f"Recognised values (case-insensitive): {', '.join(ALL_METHOD_IDS)}. "
            "Use 'all' (default) to test every method."
        ),
    )
    args = parser.parse_args()

    base = args.base.rstrip("/")
    key = args.key
    headers = {"Authorization": f"Bearer {key}"}

    # Build the normalised set of selected methods.
    if args.methods.strip().lower() == "all":
        methods: set = {m.lower() for m in ALL_METHOD_IDS}
    else:
        methods = {m.strip().lower() for m in args.methods.split(",") if m.strip()}
        unknown = methods - {m.lower() for m in ALL_METHOD_IDS}
        if unknown:
            parser.error(
                f"Unknown method(s): {', '.join(sorted(unknown))}. "
                f"Valid choices: {', '.join(ALL_METHOD_IDS)}"
            )

    print("=" * 70)
    print("  Open Kinetics Predictor API Test Suite")
    print(f"  Base URL : {base}")
    print(f"  API Key  : {key[:15]}…")
    print(f"  Methods  : {', '.join(sorted(methods))}")

    # Focused mode: only run the cached multi-substrate value checks, skip the
    # rest of the suite. --only-multi-sub implies the value-verification test.
    if args.only_multi_sub:
        print("  Mode     : ONLY multi-substrate value verification (cached sequences)")
        print("=" * 70)
        test_multi_sub_for_single_sub_methods(
            base,
            headers,
            methods,
            poll_timeout=args.poll_timeout,
        )
        return _print_summary()

    if args.only_multimer:
        print("  Mode     : ONLY multi-sequence protein value verification")
        print("=" * 70)
        test_multimer_for_methods(
            base,
            headers,
            methods,
            poll_timeout=args.poll_timeout,
        )
        return _print_summary()

    print(f"  Variants : {'extra' if args.extra_submit_variants else 'minimal'}")
    print(
        "  Multi-sub: "
        + ("value verification enabled" if args.test_multi_sub_for_single_sub_methods else "skip")
    )
    print("  Multimer : " + ("value verification enabled" if args.test_multimer else "skip"))
    print(f"  GPU tests: {'skip' if args.skip_gpu else 'auto (runs if GPU online)'}")
    print("=" * 70)

    # Run all test groups
    test_health(base)
    test_methods(base, methods)
    test_auth(base, key)
    test_quota(base, headers)

    # Submit one job for each selected method/prediction type.
    submitted_jobs = test_submit_selected_methods(base, headers, methods)
    simoff_job = test_submit_kcat_similarity_toggle_off(base, headers, methods)
    if simoff_job:
        submitted_jobs.append(simoff_job)

    # Optional success-path submission variants. If enabled, their jobs are
    # included in full end-to-end status/result validation.
    if args.extra_submit_variants:
        json_job = test_submit_json_body(base, headers, methods)
        if json_job:
            submitted_jobs.append(json_job)
    else:
        print("\n  (skipping extra submit variants — pass --extra-submit-variants to run them)")

    # All the ways submission can fail
    test_submit_errors(base, headers)

    # Status polling
    test_status(base, headers, submitted_jobs)

    # Result endpoint (job not done yet)
    test_result_not_ready(base, headers, submitted_jobs)

    # Result endpoint (poll every submitted method job until done, then download)
    test_result_completed(base, headers, submitted_jobs, poll_timeout=args.poll_timeout)

    if args.test_multi_sub_for_single_sub_methods:
        test_multi_sub_for_single_sub_methods(
            base,
            headers,
            methods,
            poll_timeout=args.poll_timeout,
        )
    else:
        print(
            "\n  (skipping multi-substrate value tests — pass "
            "--test-multi-sub-for-single-sub-methods to run them)"
        )

    if args.test_multimer:
        test_multimer_for_methods(
            base,
            headers,
            methods,
            poll_timeout=args.poll_timeout,
        )
    else:
        print("\n  (skipping multi-sequence protein tests — pass --test-multimer to run them)")

    # Validate endpoint — fast checks (no similarity)
    test_validate(base, headers)

    # Validate endpoint — with MMseqs2 similarity (slow, blocking)
    if args.skip_similarity:
        print("\n  (skipping similarity test — pass without --skip-similarity to run it)")
    else:
        test_validate_similarity(base, headers)

    # GPU embed-service status and GPU pipeline tests
    if args.skip_gpu:
        print("\n  (skipping GPU tests — pass without --skip-gpu to run them)")
    else:
        test_gpu_status(base)
        test_gpu_methods(base, headers, methods, poll_timeout=args.poll_timeout)

    # ReconXKG transparency (opt-in — submits two extra jobs)
    if args.test_recon_xkg:
        test_recon_xkg_transparency(base, headers, methods, poll_timeout=args.poll_timeout)
    else:
        print("\n  (skipping recon_xkg transparency test — pass --test-recon-xkg to run it)")

    # Method-not-allowed sanity check
    test_wrong_methods(base, headers)

    _print_summary()


def _print_summary() -> None:
    """Print the pass/fail summary and exit non-zero if anything failed."""
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
    else:
        print("  — all tests passed!")
    print("=" * 70)

    if failed:
        print("\nFailed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  ✗  {name}  {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
