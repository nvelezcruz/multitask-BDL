#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import os
import json
import re
import hashlib
from datetime import datetime, timezone
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedShuffleSplit

# PathUDP+MutTMB 10RS preprocessing: non-overwriting outputs for 10 repeated splits.
# =============================================================================
# PATHS / OUTPUTS
# =============================================================================

CONFIG_NAME = "PathUDP_MutTMB"
TCGA_ROOT = Path(os.environ["TCGA_ROOT"]).expanduser().resolve()

OUTDIR = Path(
    os.getenv(
        "OUTDIR",
        str(TCGA_ROOT / "NPJ_August_2026_FULLRERUN_PathUDP_MutTMB_pathway_UDP_clinical_treatment_mutation_TMB_MT2" / "preprocessing"),
    )
)
SEARCH_ROOTS = [TCGA_ROOT]
OUTPUT_SUFFIX = "_PathUDP_MutTMB"

# =============================================================================
# MAIN TOGGLES
# =============================================================================

# Main model feature families. Gene-expression PCA is intentionally disabled.
USE_PATHWAY_ACTIVITY = True
USE_PATHWAY_CONSISTENCY_FILTER = True
USE_UDP = True
USE_MUTATION = True
USE_TMB = True
USE_MUTATION_PROFILE = True

# Keep old RNA PC pipeline off for the main pathway/network resistance model.
USE_GENE_EXPRESSION_PCS = False

# =============================================================================
# SPLITS / SCREENING
# =============================================================================

TEST_FRAC = 0.40
SPLIT_SEED = int(os.getenv("SPLIT_SEED", "42"))
RUN_MULTI_SEEDS = os.getenv("RUN_MULTI_SEEDS", "0").strip().lower() in {"1", "true", "yes"}
MULTI_SEEDS = [int(x) for x in os.getenv("MULTI_SEEDS", "42,43,44,45,46,47,48,49,50,51").split(",") if x.strip()]

COX_PENALIZER = 0.05
DROP_FIRST_OHE = True

# Feature screening sizes.
SHARED_TOP_K_PATHWAY = 50
SPECIFIC_TOP_K_PATHWAY = 35
SHARED_TOP_K_UDP = 20
SPECIFIC_TOP_K_UDP = 10
SHARED_TOP_K_MUTATION = 20
SPECIFIC_TOP_K_MUTATION = 15

# UDP can be large; variance prefilter is fitted inside each training split.
UDP_VAR_TOP_N = 3000

# Mutation profile sparsity filters, fitted inside each training split.
MUT_MIN_PREVALENCE = 0.03
MUT_MIN_COUNT = 5

# Publication-grade mutation-profile handling.
# The raw mutationID matrix is too sparse and event-level for direct modeling.
# Collapse protein-altering mutation events to patient-by-gene mutation indicators first.
# Unparseable mutation IDs are dropped by default rather than silently treated as biology.
MUTATION_PROFILE_COLLAPSE_TO_GENE = True
MUTATION_DROP_UNPARSEABLE_EVENTS = True
MUTATION_MIN_PARSED_FRACTION_WARN = 0.25

# Consistency filter. Applied after Cox ranking for pathway/network features.
CONSISTENCY_MIN_QUANTILE = 0.50
CONSISTENCY_USE_ABS = True
SHARED_DIRECTION_MIN_CANCERS = 3

# Missing policy: do not fill missing continuous values with train mean/median.
DROP_SAMPLES_MISSING_REQUIRED_MODALITIES = True

# Treatment is central to the resistance analysis. Do not treat an unmatched
# treatment record as confirmed untreated. Keep only patients with an auditable
# treatment record and interpretable treatment status. Confirmed no-treatment
# patients are retained only if the source file explicitly encodes no treatment.
REQUIRE_TREATMENT_RECORD = True
REQUIRE_KNOWN_TREATMENT_STATUS = True

# Parser-based mutation-event-to-gene collapse. No external mutation-event map is
# assumed. The script saves a parse report and prints retained/dropped examples.
# It fails only if no gene-level mutation features can be recovered.
MUTATION_PARSE_EXAMPLE_N = 12
MUTATION_FAIL_IF_NO_PARSED_GENES = True

# =============================================================================
# COHORT CONFIG
# =============================================================================

CANCER_CONFIGS = {
    "BRCA": {
        "cancer_label": "Breast",
        "survival_candidates": [
            TCGA_ROOT
            / "TCGA_BRCA2"
            / "RNAseq_Counts_GDC"
            / "survival_BRCA_survival.txt"
        ],
        "clinical_candidates": [
            TCGA_ROOT / "TCGA_BRCA2" / "RNAseq_Counts_GDC" / "TCGA_BRCA_clinical.txt",
            TCGA_ROOT
            / "TCGA_BRCA2"
            / "RNAseq_Counts_GDC"
            / "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix.txt",
        ],
        "treatment_path": TCGA_ROOT / "TCGA_BRCA2" / "Clincal_information" / "TCGA_BRCA_GDC_clinical.tsv",
        "gene_state_dir": TCGA_ROOT / "TCGA_BRCA2" / "TCGA_GeneState_BRCA",
        "udp_file": "TCGA_BRCA-DESEQ2_normalized_counts_with_annotations_UDP.txt",
        "network_state_dir": TCGA_ROOT / "TCGA_BRCA2" / "TCGA_NetworkState_BRCA",
        "pathway_activity_file": "pathway_activity_ALL_BRCA.tsv",
        "pathway_consistency_file": "pathway_consistency_ALL_BRCA.tsv",
        "tmb_dir": TCGA_ROOT / "TCGA_BRCA2" / "TCGA_TumorMutationalBurden_BRCA",
        "tmb_file": "TCGA-BRCA_TMB_from_GDC.txt",
        "mutation_profile_dir": TCGA_ROOT
        / "TCGA_BRCA2"
        / "TCGA_TumorMutationalProfile_BRCA",
        "mutation_profile_file": "TCGA_BRCA_protein_coding_nonsyn_mutationID_matrix.tsv",
    },
    "LUAD": {
        "cancer_label": "Lung",
        "survival_candidates": [
            TCGA_ROOT / "LUAD" / "RNAseq_GDC_counts" / "TCGA_LUAD_survival.txt"
        ],
        "clinical_candidates": [
            TCGA_ROOT / "LUAD" / "RNAseq_GDC_counts" / "TCGA_LUAD_clinical.txt"
        ],
        "treatment_path": TCGA_ROOT / "LUAD" / "Clincal_information" / "TCGA_LUAD_clinical.tsv",
        "gene_state_dir": TCGA_ROOT / "LUAD" / "TCGA_GeneState_LUAD",
        "udp_file": "TCGA_LUAD_DESeq2_normalized_counts_UDP.txt",
        "network_state_dir": TCGA_ROOT / "LUAD" / "TCGA_NetworkState_LUAD",
        "pathway_activity_file": "pathway_activity_ALL_LUAD.tsv",
        "pathway_consistency_file": "pathway_consistency_ALL_LUAD.tsv",
        "tmb_dir": TCGA_ROOT / "LUAD" / "TCGA_TumorMutationalBurden_LUAD",
        "tmb_file": "TCGA-LUAD_TMB_from_GDC.txt",
        "mutation_profile_dir": TCGA_ROOT / "LUAD" / "TCGA_TumorMutationalProfile_LUAD",
        "mutation_profile_file": "TCGA_LUAD_protein_coding_nonsyn_mutationID_matrix.tsv",
    },
    "COAD": {
        "cancer_label": "Colon",
        "survival_candidates": [
            TCGA_ROOT / "COAD" / "RNAseq_GDC_counts" / "TCGA_COAD_survival.txt"
        ],
        "clinical_candidates": [
            TCGA_ROOT / "COAD" / "RNAseq_GDC_counts" / "TCGA_COAD_clinical.txt"
        ],
        "treatment_path": TCGA_ROOT / "COAD" / "RNAseq_GDC_counts" / "TCGA_COAD_GDC_clinical.tsv",
        "gene_state_dir": TCGA_ROOT / "COAD" / "TCGA_GeneState_COAD",
        "udp_file": "TCGA_COAD_DESeq2_normalized_counts_UDP.txt",
        "network_state_dir": TCGA_ROOT / "COAD" / "TCGA_NetworkState_COAD",
        "pathway_activity_file": "pathway_activity_ALL_COAD.tsv",
        "pathway_consistency_file": "pathway_consistency_ALL_COAD.tsv",
        "tmb_dir": TCGA_ROOT / "COAD" / "TCGA_TumorMutationalBurden_COAD",
        "tmb_file": "TCGA-COAD_TMB_from_GDC.txt",
        "mutation_profile_dir": TCGA_ROOT / "COAD" / "TCGA_TumorMutationalProfile_COAD",
        "mutation_profile_file": "TCGA_COAD_protein_coding_nonsyn_mutationID_matrix.tsv",
    },
}

PATHWAY_ACTIVITY_PATTERNS = [
    "pathway_activity_ALL_{code}.tsv",
    "*pathway*activity*{code}*.tsv",
]
PATHWAY_CONSISTENCY_PATTERNS = [
    "pathway_consistency_ALL_{code}.tsv",
    "*pathway*consistency*{code}*.tsv",
]
UDP_PATTERNS = [
    "*DESeq2*normalized*counts*UDP*.txt",
    "*DESEQ2*normalized*counts*UDP*.txt",
    "*UDP*.txt",
]
TMB_PATTERNS = [
    "TCGA-{code}_TMB_from_GDC.txt",
    "TCGA_{code}_TMB_from_GDC.txt",
    "*{code}*TMB*GDC*.txt",
]
MUTATION_PROFILE_PATTERNS = [
    "*protein_coding*nonsyn*mutation*matrix*.tsv",
    "*protein_coding*nonsyn*mut*.tsv",
    "*mutationID*matrix*.tsv",
    "*.tsv",
]

YES_TOKENS = {"yes", "true", "1"}
MISSING_TOKENS = {
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "not reported",
    "unknown",
    ".",
    "--",
    "'--",
}
IGNORE_TREATMENT_TYPES = {
    "Surgery, NOS",
    "Ancillary Treatment",
    "Bisphosphonate Therapy",
    "Unknown",
    "'--",
}
TOKEN = re.compile(r"[A-Z0-9-]+")

# =============================================================================
# SMALL HELPERS
# =============================================================================


def ph(msg):
    print("\n" + "=" * 88)
    print(msg)
    print("=" * 88)


def pc(name, s):
    print(f"\n[{name}]")
    vc = s.value_counts(dropna=False)
    print(vc.to_string() if len(vc) else "(empty)")


def load_table_guess(path: Path, index_col=None):
    path = Path(path)
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path, index_col=index_col, dtype=str)
        df = pd.read_table(
            path,
            sep="\t",
            comment="#",
            index_col=index_col,
            encoding="utf-8-sig",
            dtype=str,
            low_memory=False,
        )
        if df.shape[1] == 1:
            header = str(df.columns[0])
            # Some GDC-derived small tables, especially TMB files, are whitespace-delimited
            # even though they have .txt extensions. If tab parsing leaves the whole header
            # in one column, retry with the correct delimiter before giving up.
            if "," in header:
                return pd.read_csv(
                    path,
                    index_col=index_col,
                    dtype=str,
                    encoding="utf-8-sig",
                    low_memory=False,
                )
            if re.search(r"\S+\s+\S+", header):
                return pd.read_csv(
                    path,
                    sep=r"\s+",
                    engine="python",
                    comment="#",
                    index_col=index_col,
                    dtype=str,
                    encoding="utf-8-sig",
                )
        return df
    except Exception:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                index_col=index_col,
                dtype=str,
                encoding="utf-8-sig",
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                index_col=index_col,
                dtype=str,
                encoding="latin1",
            )


def resolve_first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError("\n".join(map(str, paths)))


def validate_august_outdir(path):
    """Abort before any write unless this is the approved August/config root."""
    p = Path(path).expanduser().resolve()
    allowed_roots = [
        TCGA_ROOT,
        Path(os.getenv("NPJ_SCRATCH_ROOT", str(TCGA_ROOT))).expanduser().resolve(),
    ]
    if not any(p == root or root in p.parents for root in allowed_roots):
        raise ValueError(f"Unsafe OUTDIR outside approved roots {allowed_roots}: {p}")
    if "august" not in str(p).lower() or CONFIG_NAME.lower() not in str(p).lower():
        raise ValueError(f"Unsafe OUTDIR must contain August and {CONFIG_NAME}: {p}")
    return p


def file_provenance(path, role, cancer=None):
    p = Path(path).resolve()
    h = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
    st = p.stat()
    try:
        tab = pd.read_csv(p, sep=None, engine="python", nrows=0, encoding="utf-8-sig")
        rows = max(int(newline_count) - 1 + (1 if last_byte not in {b"", b"\n"} else 0), 0)
        cols = int(tab.shape[1])
        schema = json.dumps([str(c) for c in tab.columns])
    except Exception as exc:
        rows, cols, schema = None, None, json.dumps({"read_error": str(exc)})
    return {"configuration": CONFIG_NAME, "cancer": cancer, "role": role,
            "exact_path": str(p), "sha256": h.hexdigest(), "file_size_bytes": int(st.st_size),
            "modification_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "row_count": rows, "column_count": cols, "selected_columns_schema": schema}


def resolve_by_filenames(filenames):
    for root in SEARCH_ROOTS:
        if root.exists():
            for name in filenames:
                if (root / name).exists():
                    return root / name
                hits = list(root.rglob(name))
                if hits:
                    return hits[0]
    raise FileNotFoundError("\n".join(filenames))


def first_glob(base, patterns, code=None):
    base = Path(base)
    hits = []
    for pat in patterns:
        p = pat.format(code=code) if code is not None else pat
        hits.extend(sorted(base.glob(p)))
    if not hits:
        raise FileNotFoundError(f"No matching file under {base}: {patterns}")
    return hits[0]


def patient_key(x):
    toks = str(x).strip().replace(".", "-").split("-")
    if len(toks) >= 3 and toks[0].upper() == "TCGA":
        return "-".join(toks[:3]).upper()
    return str(x).strip().upper()


def sample_key(x):
    s = str(x).strip().split(":")[0].strip().replace(".", "-")
    toks = s.split("-")
    if len(toks) >= 4 and toks[0].upper() == "TCGA":
        return "-".join([toks[0], toks[1], toks[2], toks[3][:2]]).upper()
    return s.upper()


def is_primary_tumor_sample(x):
    toks = sample_key(x).split("-")
    return len(toks) >= 4 and toks[3] == "01"


def clean_token(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return None if s == "" or s.lower() in MISSING_TOKENS else s


def split_values(x):
    s = clean_token(x)
    if s is None:
        return []
    return [
        p
        for p in [p.strip() for p in re.split(r"[;|]", s) if p.strip()]
        if p.lower() not in MISSING_TOKENS
    ]


def prep_stage_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.upper() in {
        "NA",
        "N/A",
        "NULL",
        "NONE",
        "NAN",
        "UNKNOWN",
        ".",
        "?",
        "--",
    }:
        return np.nan
    u = re.sub(r"^STAGE\s+", "", s.upper()).replace(" ", "")
    return np.nan if u in {"X", "STAGEX", "NX", "MX", "TX"} else u


def prep_generic_cat(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return (
        np.nan
        if s == ""
        or s.upper()
        in {
            "NA",
            "N/A",
            "NULL",
            "NONE",
            "NAN",
            "UNKNOWN",
            ".",
            "?",
            "--",
            "NOT REPORTED",
        }
        else s
    )


def best_patient_id_series(df, target_patients):
    candidates = [
        "cases.submitter_id",
        "bcr_patient_barcode",
        "submitter_id",
        "_PATIENT",
        "patient_id",
        "PATIENT_ID",
        "patient_barcode",
    ]
    best_col = None
    best_series = None
    best_overlap = -1
    for c in candidates:
        if c not in df.columns:
            continue
        s = df[c].map(patient_key)
        overlap = int(s.isin(target_patients).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = c
            best_series = s
    if best_series is None:
        raise ValueError("No usable patient ID col")
    return best_series, best_col, best_overlap


def _cat_levels_train(series):
    s = series.dropna().astype(str)
    levels = sorted(pd.Index(s).unique().tolist())
    if "MISSING" not in levels:
        levels.append("MISSING")
    if "OTHER" not in levels:
        levels.append("OTHER")
    return levels


def _ohe_no_mean_median_fill(series, levels, prefix):
    s = series.copy()
    s = s.where(s.notna(), "MISSING").astype(str)
    s = s.where(s.isin(levels), "OTHER")
    out = pd.get_dummies(
        pd.Categorical(s, categories=levels), prefix=prefix, dtype=float
    )
    if DROP_FIRST_OHE and out.shape[1] >= 2:
        out = out.iloc[:, 1:].copy()
    return out


def _suffix_cols(df, prefix):
    out = df.copy()
    out.attrs = {}
    out.columns = [f"{prefix}::{c}" for c in out.columns]
    return out


def _copy_without_attrs(df):
    out = df.copy()
    out.attrs = {}
    return out


def print_missing_audit(name, df, max_cols=15):
    n_rows, n_cols = df.shape
    total_missing = int(df.isna().sum().sum())
    print(f"[{name}] shape={n_rows} x {n_cols} | total_missing_cells={total_missing}")
    if total_missing:
        miss = df.isna().sum(axis=0).sort_values(ascending=False)
        miss = miss[miss > 0].head(max_cols)
        print(f"[{name}] columns with missing values, top {len(miss)}:")
        print(miss.to_string())


def assert_no_missing_values(name, df):
    if df.isna().any().any():
        bad = df.columns[df.isna().any(axis=0)].tolist()[:25]
        raise ValueError(
            f"{name} still contains missing values. This nonmissingness workflow should not pass NaNs downstream. "
            f"Example columns: {bad}"
        )


def print_design_audit(name, Xtr, Xte):
    print(f"[{name}] train shape={Xtr.shape} | test shape={Xte.shape}")
    print(
        f"[{name}] train missing cells={int(Xtr.isna().sum().sum())} | test missing cells={int(Xte.isna().sum().sum())}"
    )
    duplicated = int(pd.Index(Xtr.columns).duplicated().sum())
    print(f"[{name}] duplicated train column names={duplicated}")
    if duplicated:
        dup = pd.Index(Xtr.columns)[pd.Index(Xtr.columns).duplicated()].tolist()[:20]
        raise ValueError(
            f"{name} has duplicated feature columns after concatenation: {dup}"
        )


def _is_explicit_no_treatment_token(x):
    s = clean_token(x)
    if s is None:
        return False
    u = s.strip().upper()
    return u in {"NO", "FALSE", "0", "NO TREATMENT", "NOT TREATED", "UNTREATED"}


def _is_explicit_yes_treatment_token(x):
    s = clean_token(x)
    if s is None:
        return False
    u = s.strip().upper()
    return u in {"YES", "TRUE", "1", "TREATED"}


# =============================================================================
# SURVIVAL / CLINICAL / TREATMENT
# =============================================================================


def load_survival(path, cancer_code):
    df = load_table_guess(path)
    need = ["sample", "_PATIENT", "PFI", "PFI.time"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{cancer_code}: survival missing {miss}")
    out = pd.DataFrame(
        {
            "CANCER_CODE": cancer_code,
            "SAMPLE_KEY": df["sample"].map(sample_key),
            "PATIENT_ID": df["_PATIENT"].map(patient_key),
            "PFI_EVENT": pd.to_numeric(df["PFI"], errors="coerce"),
            "PFI_TIME_DAYS": pd.to_numeric(df["PFI.time"], errors="coerce"),
        }
    )
    out = out[df["sample"].map(is_primary_tumor_sample)].copy()
    out["PFI_TIME_MONTHS"] = out["PFI_TIME_DAYS"] / 30.4375
    return out.drop_duplicates(subset=["SAMPLE_KEY", "PATIENT_ID"])


def load_clinical(path, cancer_code, target_patients):
    df = load_table_guess(path)
    pid_series, pid_col, pid_overlap = best_patient_id_series(df, target_patients)
    age = pd.Series([pd.NA] * len(df))
    for c in [
        "age_at_diagnosis",
        "Age_at_Initial_Pathologic_Diagnosis_nature2012",
        "age_at_initial_pathologic_diagnosis",
    ]:
        if c in df.columns:
            age = pd.to_numeric(df[c], errors="coerce")
            if c == "age_at_diagnosis":
                age = age / 365.25
            break
    stage = pd.Series([pd.NA] * len(df))
    for c in [
        "ajcc_pathologic_stage",
        "ajcc_pathologic_stage.diagnoses",
        "AJCC_Stage_nature2012",
        "Converted_Stage_nature2012",
        "pathologic_stage",
    ]:
        if c in df.columns:
            stage = df[c]
            break
    grade = pd.Series([pd.NA] * len(df))
    for c in ["tumor_grade", "grade", "GRADE"]:
        if c in df.columns:
            grade = df[c]
            break
    tstage = pd.Series([pd.NA] * len(df))
    for c in ["ajcc_pathologic_t", "tumor_t", "pathologic_t"]:
        if c in df.columns:
            tstage = df[c]
            break
    out = pd.DataFrame(
        {
            "CANCER_CODE": cancer_code,
            "PATIENT_ID": pid_series,
            "AGE": age,
            "STAGE_CAT": stage.map(prep_stage_value),
            "GRADE_CAT": grade.map(prep_generic_cat),
            "AJCC_T_CAT": tstage.map(prep_generic_cat),
        }
    )
    out = out[out["PATIENT_ID"].notna()].drop_duplicates(subset=["PATIENT_ID"])
    ph(f"{cancer_code} / CLINICAL FIELD SELECTION")
    print(f"clinical_patient_col     : {pid_col}")
    print(f"clinical_patient_overlap : {pid_overlap}")
    return out


def summarize_treatments_as_flags(path, cancer_code, target_patients):
    """Summarize GDC treatment records into patient-level treatment flags.

    Critical rule for the resistance workflow:
    a treatment type is counted as positive only when the corresponding
    treatment_or_therapy field is explicitly yes.  Rows where the type says
    Radiation Therapy, NOS or Pharmaceutical Therapy, NOS but treatment_or_therapy
    says no are treated as documented no-treatment rows, not treatment-positive rows.

    The function supports both common GDC exports:
      1. long format: treatments.treatment_type + treatments.treatment_or_therapy
      2. wide format: treatments_pharmaceutical_* and treatments_radiation_*
    """
    df = load_table_guess(path)
    pid_series, patient_col, overlap = best_patient_id_series(df, target_patients)
    target_patients = set(target_patients)

    ignored_upper = {x.upper() for x in IGNORE_TREATMENT_TYPES}

    def _pick_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _therapy_status(x):
        vals = split_values(x)
        if any(_is_explicit_yes_treatment_token(v) for v in vals):
            return "yes"
        if any(_is_explicit_no_treatment_token(v) for v in vals):
            return "no"
        return "unknown"

    def _classify_treatment_type(raw):
        s = clean_token(raw)
        if s is None:
            return None
        u = s.upper().strip()
        if u in ignored_upper or u in {
            "UNKNOWN",
            "NOT REPORTED",
            "NA",
            "N/A",
            "--",
            "'--",
        }:
            return None
        if "RADIATION" in u or "BRACHY" in u:
            return "TX_RADIATION"
        if u == "CHEMOTHERAPY":
            return "TX_CHEMO"
        if u == "HORMONE THERAPY":
            return "TX_HORMONE"
        if u in {
            "PHARMACEUTICAL THERAPY, NOS",
            "TARGETED MOLECULAR THERAPY",
            "IMMUNOTHERAPY (INCLUDING VACCINES)",
        }:
            return "TX_PHARMACEUTICAL"
        return "UNKNOWN_TYPE"

    # Build a normalized long treatment table with one row per treatment record.
    long_parts = []

    type_col = _pick_col(
        ["treatments.treatment_type", "treatments_treatment_type", "treatment_type"]
    )
    therapy_col = _pick_col(
        [
            "treatments.treatment_or_therapy",
            "treatments_treatment_or_therapy",
            "treatment_or_therapy",
        ]
    )
    if type_col is not None:
        ph(f"{cancer_code} / UNIQUE VALUES / {type_col}")
        raw_types = (
            df[type_col].fillna("<NA>").astype(str).str.strip().replace("", "<EMPTY>")
        )
        for v in sorted(pd.Index(raw_types.unique()).tolist()):
            print(v)
        if therapy_col is not None:
            ph(f"{cancer_code} / UNIQUE VALUES / {therapy_col}")
            raw_therapy = (
                df[therapy_col]
                .fillna("<NA>")
                .astype(str)
                .str.strip()
                .replace("", "<EMPTY>")
            )
            for v in sorted(pd.Index(raw_therapy.unique()).tolist()):
                print(v)
        else:
            print(
                f"[{cancer_code} / TREATMENT] No treatment_or_therapy yes/no column detected for long-format treatment table."
            )
        part = pd.DataFrame(
            {
                "PATIENT_ID": pid_series,
                "TYPE": df[type_col],
                "THERAPY_YN": df[therapy_col] if therapy_col is not None else pd.NA,
                "SOURCE": "long",
            }
        )
        long_parts.append(part)

    wide_specs = [
        (
            "pharmaceutical",
            _pick_col(
                [
                    "treatments_pharmaceutical_treatment_type",
                    "treatments.pharmaceutical.treatment_type",
                    "treatments_pharmaceutical_treatment_type.diagnoses",
                ]
            ),
            _pick_col(
                [
                    "treatments_pharmaceutical_treatment_or_therapy",
                    "treatments.pharmaceutical.treatment_or_therapy",
                    "treatments_pharmaceutical_treatment_or_therapy.diagnoses",
                ]
            ),
        ),
        (
            "radiation",
            _pick_col(
                [
                    "treatments_radiation_treatment_type",
                    "treatments.radiation.treatment_type",
                    "treatments_radiation_treatment_type.diagnoses",
                ]
            ),
            _pick_col(
                [
                    "treatments_radiation_treatment_or_therapy",
                    "treatments.radiation.treatment_or_therapy",
                    "treatments_radiation_treatment_or_therapy.diagnoses",
                ]
            ),
        ),
    ]
    for source, t_col, yn_col in wide_specs:
        if t_col is None:
            continue
        ph(f"{cancer_code} / UNIQUE VALUES / {t_col}")
        raw_types = (
            df[t_col].fillna("<NA>").astype(str).str.strip().replace("", "<EMPTY>")
        )
        for v in sorted(pd.Index(raw_types.unique()).tolist()):
            print(v)
        if yn_col is not None:
            ph(f"{cancer_code} / UNIQUE VALUES / {yn_col}")
            raw_therapy = (
                df[yn_col].fillna("<NA>").astype(str).str.strip().replace("", "<EMPTY>")
            )
            for v in sorted(pd.Index(raw_therapy.unique()).tolist()):
                print(v)
        else:
            print(
                f"[{cancer_code} / TREATMENT] No yes/no treatment_or_therapy column detected for {source} treatment block."
            )
        part = pd.DataFrame(
            {
                "PATIENT_ID": pid_series,
                "TYPE": df[t_col],
                "THERAPY_YN": df[yn_col] if yn_col is not None else pd.NA,
                "SOURCE": source,
            }
        )
        long_parts.append(part)

    if not long_parts:
        raise ValueError(
            f"{cancer_code}: no usable treatment type columns found in {path}. "
            f"Columns were: {list(df.columns)}"
        )

    tmp = pd.concat(long_parts, axis=0, ignore_index=True)
    tmp["PATIENT_ID"] = tmp["PATIENT_ID"].map(patient_key)
    tmp["THERAPY_STATUS"] = tmp["THERAPY_YN"].map(_therapy_status)
    tmp["TYPE_CLASS"] = tmp["TYPE"].map(_classify_treatment_type)

    ph(f"{cancer_code} / TREATMENT TYPE BY YES-NO AUDIT")
    audit = (
        tmp[tmp["PATIENT_ID"].isin(target_patients)]
        .assign(TYPE_CLEAN=lambda z: z["TYPE"].fillna("<NA>").astype(str).str.strip())
        .groupby(["SOURCE", "TYPE_CLEAN", "THERAPY_STATUS"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["SOURCE", "TYPE_CLEAN", "THERAPY_STATUS"])
    )
    if audit.empty:
        print("(no treatment rows matched survival target patients)")
    else:
        print(audit.to_string(index=False))

    rows = []
    unknown_type_counter = {}
    for pid, g in tmp.groupby("PATIENT_ID"):
        tx_chemo = tx_hormone = tx_radiation = tx_pharma = 0
        explicit_yes_treatment = False
        explicit_no_treatment = False
        positive_type_tokens = []
        no_type_tokens = []
        unknown_status_tokens = []
        unknown_type_tokens = []

        for _, r in g.iterrows():
            status = str(r.get("THERAPY_STATUS", "unknown"))
            raw_type = r.get("TYPE", pd.NA)
            cls = r.get("TYPE_CLASS", None)

            if status == "yes":
                explicit_yes_treatment = True
                if cls == "TX_CHEMO":
                    tx_chemo = 1
                    positive_type_tokens.append(str(raw_type))
                elif cls == "TX_HORMONE":
                    tx_hormone = 1
                    positive_type_tokens.append(str(raw_type))
                elif cls == "TX_RADIATION":
                    tx_radiation = 1
                    positive_type_tokens.append(str(raw_type))
                elif cls == "TX_PHARMACEUTICAL":
                    tx_pharma = 1
                    positive_type_tokens.append(str(raw_type))
                elif cls == "UNKNOWN_TYPE":
                    unknown_type_tokens.append(str(raw_type))
                    unknown_type_counter[str(raw_type)] = (
                        unknown_type_counter.get(str(raw_type), 0) + 1
                    )
            elif status == "no":
                explicit_no_treatment = True
                if cls is not None:
                    no_type_tokens.append(str(raw_type))
            else:
                if cls is not None:
                    unknown_status_tokens.append(str(raw_type))

        recognized_positive = bool(tx_chemo or tx_hormone or tx_radiation or tx_pharma)
        confirmed_untreated = bool(explicit_no_treatment and not explicit_yes_treatment
                                   and not unknown_status_tokens and not unknown_type_tokens)
        known_status = bool(recognized_positive or confirmed_untreated)
        rows.append(
            {
                "CANCER_CODE": cancer_code,
                "PATIENT_ID": pid,
                "TX_RECORD_PRESENT": 1.0,
                "TX_KNOWN_TREATMENT_STATUS": float(known_status),
                "TX_EXPLICIT_NO_TREATMENT": float(
                    confirmed_untreated
                ),
                "TX_EXPLICIT_YES_TREATMENT": float(explicit_yes_treatment),
                "TX_CHEMO": float(tx_chemo),
                "TX_HORMONE": float(tx_hormone),
                "TX_RADIATION": float(tx_radiation),
                "TX_PHARMACEUTICAL": float(tx_pharma),
                "TX_N_RECOGNIZED_TYPES": float(len(set(positive_type_tokens))),
                "TX_N_NO_TYPES": float(len(set(no_type_tokens))),
                "TX_N_UNKNOWN_STATUS_TYPES": float(len(set(unknown_status_tokens))),
                "TX_N_UNKNOWN_TYPES": float(len(set(unknown_type_tokens))),
                "TX_TIMING_CLASS": ("timing_unknown" if recognized_positive else
                                    ("confirmed_untreated" if confirmed_untreated else "status_or_type_unknown")),
                "TX_BASELINE_CONTEXT_ELIGIBLE": float(confirmed_untreated),
            }
        )

    out = pd.DataFrame(rows)
    matched = int(out["PATIENT_ID"].isin(target_patients).sum()) if len(out) else 0
    known = (
        int(
            out.loc[
                out["PATIENT_ID"].isin(target_patients), "TX_KNOWN_TREATMENT_STATUS"
            ].sum()
        )
        if len(out)
        else 0
    )

    ph(f"{cancer_code} / TREATMENT RECORD AUDIT")
    print(f"treatment_patient_col        : {patient_col}")
    print(f"treatment_patient_overlap    : {overlap}")
    print(f"survival_target_patients     : {len(target_patients)}")
    print(f"patients_with_treatment_rows : {matched}")
    print(f"patients_with_known_tx_status: {known}")
    if len(out):
        flag_cols = [
            "TX_CHEMO",
            "TX_HORMONE",
            "TX_RADIATION",
            "TX_PHARMACEUTICAL",
            "TX_EXPLICIT_NO_TREATMENT",
        ]
        print("treatment_flag_positive_counts_among_target_patients:")
        print(
            out.loc[out["PATIENT_ID"].isin(target_patients), flag_cols]
            .sum(axis=0)
            .astype(int)
            .to_string()
        )
    if unknown_type_counter:
        print("unrecognized_treatment_type_tokens_with_yes_status, top 20:")
        for k, v in sorted(
            unknown_type_counter.items(), key=lambda kv: (-kv[1], kv[0])
        )[:20]:
            print(f"  {k}: {v}")

    return out, {
        "treatment_patient_col": patient_col,
        "treatment_patient_overlap": int(overlap),
        "treatment_type_col": type_col,
        "treatment_or_therapy_col": therapy_col,
        "patients_with_treatment_rows": int(matched),
        "patients_with_known_treatment_status": int(known),
        "unrecognized_treatment_type_tokens_top20": sorted(
            unknown_type_counter.items(), key=lambda kv: (-kv[1], kv[0])
        )[:20],
    }


def load_one_cancer_metadata(code, cfg):
    p_surv = resolve_first_existing([cfg["survival_candidates"][0]])
    p_clin = resolve_first_existing([cfg["clinical_candidates"][0]])
    p_treat = resolve_first_existing([cfg["treatment_path"]])
    ph(f"{code} / {cfg['cancer_label']}")
    print(f"SURVIVAL  = {p_surv}")
    print(f"CLINICAL  = {p_clin}")
    print(f"TREATMENT = {p_treat}")
    surv = load_survival(p_surv, code)
    target_patients = set(surv["PATIENT_ID"].astype(str))
    clin = load_clinical(p_clin, code, target_patients)
    tx, tx_debug = summarize_treatments_as_flags(p_treat, code, target_patients)
    meta = surv.merge(clin, on=["CANCER_CODE", "PATIENT_ID"], how="left").merge(
        tx, on=["CANCER_CODE", "PATIENT_ID"], how="left"
    )

    ph(f"{code} / METADATA MERGE AUDIT BEFORE TREATMENT FILTER")
    print(f"survival_rows                    : {len(surv)}")
    print(f"after_clinical_treatment_merge   : {len(meta)}")
    print(
        f"rows_with_treatment_record       : {int(meta['TX_RECORD_PRESENT'].eq(1.0).sum()) if 'TX_RECORD_PRESENT' in meta.columns else 0}"
    )
    print(
        f"rows_with_known_treatment_status : {int(meta['TX_KNOWN_TREATMENT_STATUS'].eq(1.0).sum()) if 'TX_KNOWN_TREATMENT_STATUS' in meta.columns else 0}"
    )

    before_tx_filter = len(meta)
    if REQUIRE_TREATMENT_RECORD:
        meta = meta[meta["TX_RECORD_PRESENT"].eq(1.0)].copy()
    if REQUIRE_KNOWN_TREATMENT_STATUS:
        meta = meta[meta["TX_KNOWN_TREATMENT_STATUS"].eq(1.0)].copy()
    after_tx_filter = len(meta)

    ph(f"{code} / STRICT TREATMENT FILTER")
    print(f"REQUIRE_TREATMENT_RECORD         : {REQUIRE_TREATMENT_RECORD}")
    print(f"REQUIRE_KNOWN_TREATMENT_STATUS   : {REQUIRE_KNOWN_TREATMENT_STATUS}")
    print(f"rows_before_treatment_filter     : {before_tx_filter}")
    print(f"rows_after_treatment_filter      : {after_tx_filter}")
    print(f"rows_removed_by_treatment_filter : {before_tx_filter - after_tx_filter}")

    tx_cols = ["TX_CHEMO", "TX_HORMONE", "TX_RADIATION", "TX_PHARMACEUTICAL"]
    required_tx_cols = [
        "TX_RECORD_PRESENT",
        "TX_KNOWN_TREATMENT_STATUS",
        "TX_EXPLICIT_NO_TREATMENT",
    ] + tx_cols
    missing_tx_cols = [c for c in required_tx_cols if c not in meta.columns]
    if missing_tx_cols:
        raise ValueError(
            f"{code}: treatment filtering removed/failed required treatment columns: {missing_tx_cols}"
        )
    if meta[required_tx_cols].isna().any().any():
        bad = (
            meta[required_tx_cols]
            .columns[meta[required_tx_cols].isna().any(axis=0)]
            .tolist()
        )
        raise ValueError(
            f"{code}: treatment columns still contain missing values after strict filter: {bad}"
        )
    for c in tx_cols:
        meta[c] = meta[c].astype(float)

    print("treatment_flags_after_filter:")
    print(
        meta[["TX_EXPLICIT_NO_TREATMENT"] + tx_cols].sum(axis=0).astype(int).to_string()
    )

    # Multi-hot treatment audit: these treatment columns are independent binary
    # covariates, not mutually exclusive one-hot categories. A patient can have
    # TX_CHEMO=1 and TX_RADIATION=1 if both paired treatment_or_therapy fields
    # are yes in the GDC treatment records.
    combo = (
        meta[tx_cols]
        .astype(int)
        .astype(str)
        .agg(
            lambda r: "+".join(
                [c.replace("TX_", "") for c, v in zip(tx_cols, r) if v == "1"]
            )
            or "NO_RECOGNIZED_TREATMENT",
            axis=1,
        )
        .value_counts()
    )
    print("treatment_multi_hot_combinations_after_filter:")
    print(combo.to_string())

    meta["GROUP_LABEL"] = cfg["cancer_label"]
    info = {
        "cancer_code": code,
        "cancer_label": cfg["cancer_label"],
        "survival_file": str(p_surv),
        "clinical_file": str(p_clin),
        "treatment_file": str(p_treat),
        "treatment_debug": tx_debug,
        "rows_before_treatment_filter": int(before_tx_filter),
        "rows_after_treatment_filter": int(after_tx_filter),
        "rows_removed_by_treatment_filter": int(before_tx_filter - after_tx_filter),
    }
    return meta, info


# =============================================================================
# FEATURE MATRIX LOADERS
# =============================================================================


def _looks_like_tcga_series(s):
    vals = s.dropna().astype(str).head(50)
    if len(vals) == 0:
        return False
    return vals.str.upper().str.startswith("TCGA").mean() > 0.5


def _numeric_matrix(df):
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _read_gene_by_sample_matrix_with_missing_corner_header(path):
    """Read matrices whose header omits the leading gene-ID corner cell.

    The TCGA UDP files have N TCGA sample IDs on the first line but N+1 fields
    on every data line (gene ID followed by N values).  Letting pandas infer the
    header silently shifts the matrix and creates numeric tuple feature labels.
    Return None for ordinary rectangular files so the general loader handles
    them unchanged.
    """
    path = Path(path)
    with path.open("rt", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        first_data = handle.readline().rstrip("\r\n").split("\t")
    if not header or not first_data:
        return None
    tcga_header_fraction = float(np.mean([
        str(value).strip().upper().replace(".", "-").startswith("TCGA-")
        for value in header
    ]))
    if len(first_data) != len(header) + 1 or tcga_header_fraction <= 0.5:
        return None
    matrix = pd.read_csv(
        path,
        sep="\t",
        header=None,
        skiprows=1,
        index_col=0,
        low_memory=False,
    )
    if matrix.shape[1] != len(header):
        raise ValueError(
            f"Malformed gene-by-sample matrix {path}: parsed {matrix.shape[1]} sample columns "
            f"but header contains {len(header)} sample IDs"
        )
    matrix.columns = header
    matrix.index = matrix.index.astype(str)
    return matrix


def load_sample_feature_matrix(path, cohort_samples=None, binary=False, name="feature"):
    path = Path(path)
    missing_corner = _read_gene_by_sample_matrix_with_missing_corner_header(path)
    if missing_corner is not None:
        X = _numeric_matrix(missing_corner).T
        X.index = [sample_key(i) for i in X.index]
        X = X.groupby(X.index).max() if binary else X.groupby(X.index).median()
        X.columns = [str(c) for c in X.columns]
    else:
        df = load_table_guess(path)
        tcga_cols = [c for c in df.columns if str(c).upper().startswith("TCGA")]
        if len(tcga_cols) > 0:
            feature_cols = [c for c in df.columns if c not in tcga_cols]
            feature_col = feature_cols[0] if feature_cols else df.columns[0]
            M = df[[feature_col] + tcga_cols].copy().set_index(feature_col)
            X = _numeric_matrix(M).T
            X.index = [sample_key(i) for i in X.index]
            X = X.groupby(X.index).max() if binary else X.groupby(X.index).median()
            X.columns = [str(c) for c in X.columns]
        else:
            first = df.columns[0]
            if _looks_like_tcga_series(df[first]):
                M = df.set_index(first)
                X = _numeric_matrix(M)
                X.index = [sample_key(i) for i in X.index]
                X = X.groupby(X.index).max() if binary else X.groupby(X.index).median()
            else:
                raise ValueError(
                    f"{name}: cannot detect TCGA sample orientation for {path}"
                )

    # BRCA UDP uses versioned Ensembl IDs (e.g. ENSG...15.4), while LUAD and
    # COAD use unversioned IDs.  Normalize only UDP feature identifiers so the
    # same biological gene can enter the prespecified all-cancer intersection.
    if "UDP" in str(name).upper():
        X.columns = [re.sub(r"\.\d+$", "", str(c)) for c in X.columns]
        if pd.Index(X.columns).duplicated().any():
            X = X.T.groupby(level=0, sort=False).median().T

    if binary:
        n_binary_missing = int(X.isna().sum().sum())
        if n_binary_missing:
            print(
                f"[{name}] binary matrix contains {n_binary_missing} missing cells before thresholding; "
                "for mutation event matrices, blank/non-numeric cells are interpreted as event absence."
            )
        X = (X.fillna(0.0) > 0).astype(float)

    X = X.loc[:, ~pd.Index(X.columns).duplicated()].copy()
    if cohort_samples is not None:
        keep = [s for s in cohort_samples if s in X.index]
        X = X.loc[keep].copy()
    print(f"[{name}] {path.name}: {X.shape[0]} samples x {X.shape[1]} features")
    return X


def _norm_colname(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())


def _find_first_col(df, exact_names=None, contains_all=None, contains_any=None):
    exact_names = {_norm_colname(x) for x in (exact_names or [])}
    contains_all = [str(x).lower() for x in (contains_all or [])]
    contains_any = [str(x).lower() for x in (contains_any or [])]
    for c in df.columns:
        nc = _norm_colname(c)
        lc = str(c).strip().lower()
        if nc in exact_names:
            return c
        if contains_all and all(tok in lc for tok in contains_all):
            return c
        if contains_any and any(tok in lc for tok in contains_any):
            return c
    return None


def load_tmb_table(path, cohort_patients=None, cohort_samples=None):
    df = load_table_guess(path)
    print(f"[TMB] columns in {Path(path).name}: {list(df.columns)}")

    tmb_col = _find_first_col(
        df,
        exact_names=[
            "TMB_mut_per_mb",
            "tmb_mut_per_mb",
            "TMB",
            "tmb",
            "tumor_mutational_burden",
            "tumor_mutation_burden",
            "mut_per_mb",
            "mutations_per_mb",
            "mutation_per_mb",
            "nonsyn_mut_per_mb",
            "nonsynonymous_mut_per_mb",
        ],
    )
    if tmb_col is None:
        tmb_col = _find_first_col(df, contains_all=["tmb"])
    if tmb_col is None:
        tmb_col = _find_first_col(df, contains_all=["per", "mb"])

    count_col = _find_first_col(
        df,
        exact_names=[
            "nonsyn_mutations",
            "nonsynonymous_mutations",
            "nonsynonymous",
            "nonsyn_count",
            "nonsynonymous_count",
            "mutation_count",
            "mutations",
        ],
    )
    if count_col is None:
        count_col = _find_first_col(df, contains_all=["nonsyn"])
    if count_col is None:
        count_col = _find_first_col(df, contains_all=["mutation"])

    pid_col = _find_first_col(
        df,
        exact_names=[
            "patient_barcode",
            "patient",
            "patient_id",
            "PATIENT_ID",
            "bcr_patient_barcode",
            "cases.submitter_id",
            "submitter_id",
            "_PATIENT",
        ],
    )
    sample_col = _find_first_col(
        df,
        exact_names=[
            "sample_barcode",
            "Tumor_Sample_Barcode",
            "tumor_sample_barcode",
            "sample",
            "sample_id",
            "aliquot_barcode",
            "Tumor_Sample_UUID",
        ],
    )
    if sample_col is None:
        sample_col = _find_first_col(df, contains_all=["sample"])

    if pid_col is None and sample_col is not None:
        pid_values = df[sample_col].map(patient_key)
        pid_col_used = f"derived_from::{sample_col}"
    elif pid_col is not None:
        pid_values = df[pid_col].map(patient_key)
        pid_col_used = pid_col
    else:
        raise ValueError(
            f"No patient/sample barcode column in TMB file: {path}. Columns were: {list(df.columns)}"
        )

    if tmb_col is not None:
        tmb_values = pd.to_numeric(df[tmb_col], errors="coerce")
        tmb_source = tmb_col
    elif count_col is not None:
        # The TCGA files used here have nonsyn_mutations / TMB_mut_per_mb ratios
        # consistent with a 38 Mb coding-region denominator. This is only used
        # when the file provides counts but not an explicit TMB-per-Mb column.
        EXOME_MB_FOR_TMB_FALLBACK = 38.0
        print(
            f"[TMB] WARNING: no explicit TMB-per-Mb column found in {Path(path).name}; "
            f"using {count_col} / {EXOME_MB_FOR_TMB_FALLBACK:g}."
        )
        tmb_values = (
            pd.to_numeric(df[count_col], errors="coerce") / EXOME_MB_FOR_TMB_FALLBACK
        )
        tmb_source = f"{count_col}/38Mb"
    else:
        raise ValueError(
            f"No usable TMB or mutation-count column in TMB file: {path}. Columns were: {list(df.columns)}"
        )

    tmp = pd.DataFrame({"PATIENT_ID": pid_values, "TMB_mut_per_mb": tmb_values})
    tmp["SOURCE_ROW"] = np.arange(len(tmp), dtype=int)
    tmp["SAMPLE_BARCODE"] = df[sample_col].astype(str).values if sample_col is not None else ""
    tmp["SAMPLE_KEY"] = tmp["SAMPLE_BARCODE"].map(sample_key) if sample_col is not None else ""
    tmp["PRIMARY_TUMOR"] = tmp["SAMPLE_BARCODE"].map(is_primary_tumor_sample).astype(bool) if sample_col is not None else False
    rna_by_patient = {}
    for sk in (cohort_samples or []):
        rna_by_patient.setdefault(patient_key(sk), sample_key(sk))
    tmp["RNA_SAMPLE_KEY"] = tmp["PATIENT_ID"].map(rna_by_patient).fillna("")
    tmp["MATCHES_RNA_SPECIMEN"] = tmp["SAMPLE_KEY"].eq(tmp["RNA_SAMPLE_KEY"]) & tmp["RNA_SAMPLE_KEY"].ne("")
    tmp = tmp[tmp["PATIENT_ID"].notna() & tmp["TMB_mut_per_mb"].notna()].copy()
    tmp["SELECTION_PRIORITY"] = np.where(tmp["MATCHES_RNA_SPECIMEN"], 0, np.where(tmp["PRIMARY_TUMOR"], 1, 2))
    tmp = tmp.sort_values(["PATIENT_ID", "SELECTION_PRIORITY", "SAMPLE_BARCODE", "SOURCE_ROW"], kind="mergesort")
    chosen = tmp.groupby("PATIENT_ID", sort=True, as_index=False).head(1).copy()
    chosen["SELECTION_REASON"] = np.select(
        [chosen["MATCHES_RNA_SPECIMEN"], chosen["PRIMARY_TUMOR"]],
        ["matched_RNA_specimen", "deterministic_primary_tumor_barcode"],
        default="deterministic_lexicographic_barcode",
    )
    out = chosen.set_index("PATIENT_ID")[["TMB_mut_per_mb"]].copy()
    out["LOG1P_TMB"] = np.log1p(out["TMB_mut_per_mb"].astype(float))
    if cohort_patients is not None:
        out = out.loc[[p for p in cohort_patients if p in out.index]].copy()
    dup = tmp[tmp.duplicated("PATIENT_ID", keep=False)].copy()
    med = tmp.groupby("PATIENT_ID")["TMB_mut_per_mb"].median().rename("TMB_patient_median_sensitivity")
    disagreement = tmp.groupby("PATIENT_ID")["TMB_mut_per_mb"].agg(["count", "min", "max", "std"]).reset_index()
    disagreement = disagreement.merge(med.reset_index(), on="PATIENT_ID", how="left")
    out.attrs["duplicate_aliquot_tmb_audit"] = dup
    out.attrs["tmb_disagreement_by_patient"] = disagreement
    out.attrs["tmb_sample_selection_rule"] = chosen
    print(
        f"[TMB] {Path(path).name}: {out.shape[0]} patients | "
        f"patient_col={pid_col_used} | sample_col={sample_col} | tmb_source={tmb_source}"
    )
    return out


def _parse_gene_symbol_from_mutation_id(feature_name):
    """Best-effort parser for mutation-event identifiers.

    Expected mutation matrices are often event-level. For modeling, we collapse
    event indicators to gene-level indicators when a gene symbol can be recovered
    from the event name. If the file uses opaque mutation IDs with no gene symbol,
    those events are dropped under MUTATION_DROP_UNPARSEABLE_EVENTS=True.
    """
    s = str(feature_name).strip()
    if not s or s.lower() in MISSING_TOKENS:
        return None
    if re.match(r"^TCGA[-.]", s, flags=re.I):
        return None

    # Actual TCGA matrices used here encode events as GENE_chr... .  Parse that
    # structure first so valid HGNC symbols such as CHRD, NME1, NPM1, and RSPH1
    # are never rejected merely because of their first letters.
    m = re.match(r"^([A-Za-z][A-Za-z0-9.-]{1,30})[_|:-]+chr(?:chr)?(?:[0-9XYM]+)\b", s, flags=re.I)
    if m:
        return m.group(1).upper().replace(".", "-")

    # Common cases: TP53:p.R175H, TP53|chr17:..., TP53_Missense_Mutation, TP53-...
    toks = re.findall(r"[A-Za-z][A-Za-z0-9-]{1,30}", s)
    if not toks:
        return None

    reject_exact = {
        "TCGA",
        "MUT",
        "MUTATION",
        "MUTATIONS",
        "MUTATIONID",
        "ID",
        "NONSYN",
        "NONSYNONYMOUS",
        "PROTEIN",
        "CODING",
        "SAMPLE",
        "MISSENSE",
        "NONSENSE",
        "SILENT",
        "SPLICE",
        "FRAME",
        "FRAMESHIFT",
        "DEL",
        "INS",
        "DUP",
        "AMP",
        "HET",
        "HOM",
        "VARIANT",
        "SNV",
        "SNP",
        "INDEL",
        "STOPGAIN",
        "STOPLOSS",
        "UNKNOWN",
        "NA",
    }
    accession_patterns = (
        r"^(?:ENST|ENSG)\d+(?:\.\d+)?$",
        r"^(?:NM|NR|NP|XM|XR|XP)_?\d+(?:\.\d+)?$",
        r"^RS\d+$",
        r"^(?:CHR|CHROM)(?:[0-9XYM]+)$",
        r"^EXON\d+$",
    )

    # Prefer known-looking HGNC symbols near the start of the event label.
    for tok in toks[:8]:
        u = tok.upper().strip("-_")
        if len(u) < 2 or len(u) > 20:
            continue
        if u in reject_exact:
            continue
        if u in {"NM", "NR", "NP", "XM", "XR", "XP", "RS", "CHR", "CHROM", "EXON"}:
            continue
        if any(re.match(pat, u) for pat in accession_patterns):
            continue
        if re.match(r"^[A-Z][A-Z0-9-]*[A-Z0-9]$", u):
            return u
    return None


def _collapse_mutation_events_to_genes(Xs):
    """Collapse sample-by-event mutation matrix to sample-by-gene indicators."""
    gene_by_col = {}
    dropped = []
    for c in Xs.columns:
        g = _parse_gene_symbol_from_mutation_id(c)
        if g is None:
            dropped.append(c)
        else:
            gene_by_col[c] = g

    n_total = int(Xs.shape[1])
    n_parsed = int(len(gene_by_col))
    parsed_fraction = float(n_parsed / max(n_total, 1))
    ph("MUTATION PROFILE / EVENT-TO-GENE PARSE AUDIT")
    print(
        f"[MUTATION_PROFILE_GENE_PARSE] parsed {n_parsed}/{n_total} "
        f"mutation-event columns to gene symbols ({parsed_fraction:.3f})"
    )
    if gene_by_col:
        print("[MUTATION_PROFILE_GENE_PARSE] parsed examples:")
        for raw, gene in list(gene_by_col.items())[:MUTATION_PARSE_EXAMPLE_N]:
            print(f"  {raw} -> {gene}")
    if dropped:
        print("[MUTATION_PROFILE_GENE_PARSE] dropped/unparseable examples:")
        for raw in dropped[:MUTATION_PARSE_EXAMPLE_N]:
            print(f"  {raw}")
    if parsed_fraction < MUTATION_MIN_PARSED_FRACTION_WARN:
        print(
            "[MUTATION_PROFILE_GENE_PARSE] WARNING: low parsed fraction. "
            "This run is relying on parser-based mutation-event labels because no external map was provided. "
            "Inspect the saved parse report before making mutation-feature claims."
        )

    if len(gene_by_col) == 0:
        parse_report = pd.DataFrame(
            {
                "raw_feature": dropped,
                "parsed_gene": [None] * len(dropped),
                "kept": False,
            }
        )
        if MUTATION_FAIL_IF_NO_PARSED_GENES:
            raise ValueError(
                "No gene-level mutation features could be recovered from the mutation-profile matrix. "
                "Because this workflow relies on parser-based mutation labels, inspect the raw mutation column names "
                "or provide mutation columns that contain gene symbols."
            )
        if MUTATION_DROP_UNPARSEABLE_EVENTS:
            return pd.DataFrame(index=Xs.index), parse_report
        out = Xs.copy()
        out.columns = [f"RAW_EVENT::{c}" for c in out.columns]
        return out, pd.DataFrame(
            {
                "raw_feature": list(Xs.columns),
                "parsed_gene": list(out.columns),
                "kept": True,
            }
        )

    kept_cols = list(gene_by_col.keys())
    genes = pd.Index([gene_by_col[c] for c in kept_cols], name="GENE")
    X_kept = Xs.loc[:, kept_cols].astype(float)
    # Max over events belonging to the same gene: any protein-altering mutation -> gene mutated.
    X_gene = X_kept.T.groupby(genes).max().T
    X_gene = (X_gene > 0).astype(float)

    parse_report = pd.DataFrame(
        {
            "raw_feature": kept_cols + dropped,
            "parsed_gene": [gene_by_col[c] for c in kept_cols] + [None] * len(dropped),
            "kept": [True] * len(kept_cols) + [False] * len(dropped),
        }
    )
    print(
        f"[MUTATION_PROFILE_GENE_COLLAPSE] {Xs.shape[1]} event columns -> "
        f"{X_gene.shape[1]} gene-level mutation indicators"
    )
    gene_prev = X_gene.mean(axis=0).sort_values(ascending=False)
    print(
        "[MUTATION_PROFILE_GENE_COLLAPSE] top mutated parsed genes by sample prevalence:"
    )
    print(gene_prev.head(15).to_string())
    return X_gene, parse_report


def load_mutation_profile_matrix(path, cohort_patients=None):
    Xs = load_sample_feature_matrix(
        path, cohort_samples=None, binary=True, name="MUTATION_PROFILE_SAMPLE"
    )

    primary_mask = pd.Index(Xs.index).map(is_primary_tumor_sample)
    if int(np.sum(primary_mask)) > 0:
        Xs = Xs.loc[primary_mask].copy()
        print(
            f"[MUTATION_PROFILE_PRIMARY_ONLY] {Path(path).name}: kept {Xs.shape[0]} primary-tumor samples"
        )
    else:
        print(
            f"[MUTATION_PROFILE_PRIMARY_ONLY] WARNING: primary-tumor filter found zero rows in {Path(path).name}; keeping all rows."
        )

    parse_report = pd.DataFrame()
    if MUTATION_PROFILE_COLLAPSE_TO_GENE:
        Xs, parse_report = _collapse_mutation_events_to_genes(Xs)
        if Xs.shape[1] == 0:
            raise ValueError(
                "No gene-level mutation features could be recovered; this nonmissingness workflow should not continue with an empty mutation profile."
            )

    Xs["PATIENT_ID"] = [patient_key(i) for i in Xs.index]
    Xp = Xs.groupby("PATIENT_ID").max()
    if cohort_patients is not None:
        Xp = Xp.loc[[p for p in cohort_patients if p in Xp.index]].copy()
    Xp.attrs["mutation_parse_report"] = parse_report
    print(
        f"[MUTATION_PROFILE_PATIENT] {Path(path).name}: {Xp.shape[0]} patients x {Xp.shape[1]} features"
    )
    return Xp


def resolve_exact_or_glob(base, exact_name=None, patterns=None, code=None):
    base = Path(base)
    if exact_name:
        p = base / exact_name
        if p.exists():
            return p
        raise FileNotFoundError(f"Expected exact file not found: {p}")
    return first_glob(base, patterns or [], code)


def resolve_feature_files():
    files = {}
    for code, cfg in CANCER_CONFIGS.items():
        files[code] = {}
        if USE_PATHWAY_ACTIVITY:
            files[code]["pathway_activity"] = resolve_exact_or_glob(
                cfg["network_state_dir"],
                cfg.get("pathway_activity_file"),
                PATHWAY_ACTIVITY_PATTERNS,
                code,
            )
        if USE_PATHWAY_CONSISTENCY_FILTER:
            files[code]["pathway_consistency"] = resolve_exact_or_glob(
                cfg["network_state_dir"],
                cfg.get("pathway_consistency_file"),
                PATHWAY_CONSISTENCY_PATTERNS,
                code,
            )
        if USE_UDP:
            files[code]["udp"] = resolve_exact_or_glob(
                cfg["gene_state_dir"], cfg.get("udp_file"), None, code
            )
        if USE_MUTATION and USE_TMB:
            files[code]["tmb"] = resolve_exact_or_glob(
                cfg["tmb_dir"], cfg.get("tmb_file"), TMB_PATTERNS, code
            )
        if USE_MUTATION and USE_MUTATION_PROFILE:
            files[code]["mutation_profile"] = resolve_exact_or_glob(
                cfg["mutation_profile_dir"], cfg.get("mutation_profile_file"), None, code
            )
    return files


# =============================================================================
# COX / SELECTION / SCALING
# =============================================================================


def _binary_separation_problem(
    df, x_col="x", event_col="event", min_events_per_level=1
):
    vals = sorted(pd.Series(df[x_col]).dropna().unique().tolist())
    if len(vals) != 2:
        return False
    for v in vals:
        g = df[df[x_col] == v]
        if len(g) == 0:
            return True
        n_event = int(g[event_col].sum())
        n_nonevent = int(len(g) - n_event)
        if n_event < int(min_events_per_level) or n_nonevent < int(
            min_events_per_level
        ):
            return True
    return False


def _fit_univariate_cox_with_retries(df, use_strata=False, base_penalizer=0.0):
    """Fit a one-feature Cox model and treat convergence warnings as failed fits.

    The screening stage should not select unstable features whose apparent
    association is driven by sparse events, separation, or non-converged Newton
    steps.  We therefore retry with stronger ridge penalization and only return
    a nonzero score from a fit that completed without a convergence warning.
    """
    base_penalizer = float(base_penalizer or 0.0)
    penalizers = []
    for pen in [base_penalizer, 0.10, 0.25, 0.50, 1.00]:
        if pen not in penalizers:
            penalizers.append(pen)

    last_msg = ""
    for pen in penalizers:
        try:
            cph = CoxPHFitter(penalizer=float(pen))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                if use_strata:
                    cph.fit(
                        df,
                        duration_col="time",
                        event_col="event",
                        strata=["STRATA"],
                        show_progress=False,
                    )
                else:
                    cph.fit(
                        df, duration_col="time", event_col="event", show_progress=False
                    )
            conv_msgs = [
                str(w.message)
                for w in caught
                if issubclass(w.category, ConvergenceWarning)
            ]
            if conv_msgs:
                last_msg = " | ".join(conv_msgs[:2])
                continue
            s = (
                cph.summary.loc["x"]
                if "x" in cph.summary.index
                else cph.summary.iloc[0]
            )
            return {
                "z": float(s["z"]),
                "p": float(s["p"]),
                "coef": float(s["coef"]),
                "penalizer_used": float(pen),
                "cox_status": "ok",
                "convergence_warning": "",
            }
        except Exception as exc:
            last_msg = str(exc)[:500]
            continue

    return {
        "z": 0.0,
        "p": 1.0,
        "coef": 0.0,
        "penalizer_used": np.nan,
        "cox_status": "failed_or_nonconverged",
        "convergence_warning": last_msg[:500],
    }


def cox_rank_univariate(X, t, e, strata=None, penalizer=0.0):
    rows = []
    X = X.copy()
    base = pd.DataFrame(
        {"time": np.asarray(t, dtype=float), "event": np.asarray(e, dtype=int)},
        index=X.index,
    )
    if strata is not None:
        base["STRATA"] = pd.Series(strata, index=X.index).astype(str)
    use_strata = strata is not None

    for feat in X.columns:
        x = pd.to_numeric(X[feat], errors="coerce")
        ok = base["time"].notna() & base["event"].notna() & x.notna()
        n_ok = int(ok.sum())
        if n_ok < 10 or x.loc[ok].nunique() < 2 or float(np.nanvar(x.loc[ok])) < 1e-12:
            rows.append(
                {
                    "feat": feat,
                    "z": 0.0,
                    "p": 1.0,
                    "coef": 0.0,
                    "n": n_ok,
                    "penalizer_used": np.nan,
                    "cox_status": "insufficient_variation",
                    "convergence_warning": "",
                }
            )
            continue

        df = base.loc[ok].copy()
        df["x"] = x.loc[ok].astype(float)

        # Sparse binary mutation indicators can create complete or quasi-complete
        # separation.  Those features are not reliable enough for supervised
        # screening, so skip them rather than letting an unstable hazard ratio rank highly.
        if _binary_separation_problem(
            df, x_col="x", event_col="event", min_events_per_level=1
        ):
            rows.append(
                {
                    "feat": feat,
                    "z": 0.0,
                    "p": 1.0,
                    "coef": 0.0,
                    "n": n_ok,
                    "penalizer_used": np.nan,
                    "cox_status": "binary_separation_or_empty_event_level",
                    "convergence_warning": "",
                }
            )
            continue

        fit = _fit_univariate_cox_with_retries(
            df, use_strata=use_strata, base_penalizer=penalizer
        )
        rows.append({"feat": feat, "n": n_ok, **fit})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by="z", key=np.abs, ascending=False).reset_index(drop=True)


def standardize_train_test(Xtr, Xte):
    """Train-only filtering and scaling for required continuous feature matrices.

    No feature filtering decision is made from test-set values. If a feature kept
    from the training set has missing test values, fail explicitly rather than
    silently imputing pathway/UDP activity.
    """
    cols = [c for c in Xtr.columns if c in Xte.columns]
    Xtr = Xtr.loc[:, cols].copy()
    Xte = Xte.loc[:, cols].copy()

    train_complete = ~Xtr.isna().any(axis=0)
    Xtr = Xtr.loc[:, train_complete].copy()
    Xte = Xte.loc[:, Xtr.columns].copy()

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=0)
    keep = sd[np.isfinite(sd) & (sd >= 1e-8)].index.tolist()
    Xtr = Xtr.loc[:, keep]
    Xte = Xte.loc[:, keep]

    if Xte.isna().any().any():
        bad = Xte.columns[Xte.isna().any(axis=0)].tolist()[:20]
        raise ValueError(
            "Held-out/test rows contain missing values in features selected from training. "
            "Because pathway/UDP activity is a required modality, this should be fixed upstream, "
            f"not mean/median-imputed. Example columns: {bad}"
        )

    mu = mu.loc[keep]
    sd = sd.loc[keep]
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    params = pd.DataFrame(
        {"feature": keep, "mean_train": mu.values, "sd_train": sd.values}
    )
    return Ztr, Zte, params


def filter_udp_by_train_variance(Xtr, Xte, top_n=UDP_VAR_TOP_N):
    common = [c for c in Xtr.columns if c in Xte.columns]
    Xtr = Xtr.loc[:, common].copy()
    Xte = Xte.loc[:, common].copy()
    train_complete = ~Xtr.isna().any(axis=0)
    Xtr = Xtr.loc[:, train_complete]
    Xte = Xte.loc[:, Xtr.columns]
    if Xte.isna().any().any():
        bad = Xte.columns[Xte.isna().any(axis=0)].tolist()[:20]
        raise ValueError(
            "Held-out/test UDP rows contain missing values in train-selected features. "
            f"Fix upstream or disable UDP for this run. Example columns: {bad}"
        )
    var = Xtr.var(axis=0, skipna=True).sort_values(ascending=False)
    keep = var.head(min(top_n, len(var))).index.tolist()
    return (
        Xtr.loc[:, keep],
        Xte.loc[:, keep],
        var.reset_index().rename(columns={"index": "feature", 0: "train_variance"}),
    )


def filter_mutations_by_train_prevalence(Xtr, Xte):
    common = [c for c in Xtr.columns if c in Xte.columns]
    Xtr = Xtr.loc[:, common].copy()
    Xte = Xte.loc[:, common].copy()
    measured_tr = ~Xtr.isna().all(axis=1)
    Xtr_measured = Xtr.loc[measured_tr].fillna(0.0).astype(float)
    if Xtr_measured.shape[0] == 0 or Xtr_measured.shape[1] == 0:
        report = pd.DataFrame(
            columns=["feature", "train_prevalence", "train_count", "n_train_measured"]
        )
        return Xtr.iloc[:, 0:0].copy(), Xte.iloc[:, 0:0].copy(), report
    prev = Xtr_measured.mean(axis=0)
    cnt = Xtr_measured.sum(axis=0)
    keep = prev[(prev >= MUT_MIN_PREVALENCE) & (cnt >= MUT_MIN_COUNT)].index.tolist()
    report = pd.DataFrame(
        {
            "feature": prev.index,
            "train_prevalence": prev.values,
            "train_count": cnt.values,
            "n_train_measured": int(Xtr_measured.shape[0]),
        }
    )
    return Xtr.loc[:, keep].copy(), Xte.loc[:, keep].copy(), report


def apply_consistency_filter(ranking, Ctr, top_k):
    if Ctr is None or Ctr.shape[1] == 0 or ranking.empty:
        return ranking.head(top_k).copy(), pd.DataFrame()
    C = Ctr.copy()
    if CONSISTENCY_USE_ABS:
        score = C.abs().mean(axis=0, skipna=True)
    else:
        score = C.mean(axis=0, skipna=True)
    q = float(score.quantile(CONSISTENCY_MIN_QUANTILE)) if len(score) else -np.inf
    ok = set(score[score >= q].index)
    out = ranking[ranking["feat"].isin(ok)].head(top_k).copy()
    score_df = (
        score.rename("consistency_score_train")
        .reset_index()
        .rename(columns={"index": "feat"})
    )
    return out, score_df


def direction_consistency_filter(
    shared_rank, per_cancer_coefs, min_cancers=SHARED_DIRECTION_MIN_CANCERS
):
    if shared_rank.empty:
        return shared_rank
    keep_rows = []
    for _, row in shared_rank.iterrows():
        feat = row["feat"]
        signs = []
        for code, coef_map in per_cancer_coefs.items():
            coef = coef_map.get(feat, np.nan)
            if np.isfinite(coef) and abs(coef) > 0:
                signs.append(np.sign(coef))
        pos = sum(s > 0 for s in signs)
        neg = sum(s < 0 for s in signs)
        if max(pos, neg) >= min_cancers:
            r = row.copy()
            r["direction_pos_cancers"] = int(pos)
            r["direction_neg_cancers"] = int(neg)
            keep_rows.append(r)
    if not keep_rows:
        return shared_rank.iloc[0:0].copy()
    return pd.DataFrame(keep_rows).reset_index(drop=True)


# =============================================================================
# CLINICAL / TREATMENT / MUTATION FEATURE BLOCKS
# =============================================================================


def build_covariate_block(
    meta_tr_g, meta_te_g, tmb_by_patient=None, mut_tr=None, mut_te=None,
    cancer_code=None,
):
    # AGE: patients with missing age were already dropped. No mean/median missing imputation.
    age_mu = float(meta_tr_g["AGE"].mean())
    age_sd = float(meta_tr_g["AGE"].std(ddof=0))
    if (not np.isfinite(age_sd)) or age_sd < 1e-8:
        age_sd = 1.0
    Xtr = pd.DataFrame(index=meta_tr_g["SAMPLE_KEY"].tolist())
    Xte = pd.DataFrame(index=meta_te_g["SAMPLE_KEY"].tolist())
    Xtr["clincont::AGE"] = (meta_tr_g["AGE"].astype(float).to_numpy() - age_mu) / age_sd
    Xte["clincont::AGE"] = (meta_te_g["AGE"].astype(float).to_numpy() - age_mu) / age_sd

    # Treatment flags are multi-hot independent binary covariates. They are not
    # mutually exclusive one-hot categories: a patient with both chemotherapy and
    # radiation recorded as treatment_or_therapy == yes receives both inputs.
    tx_cols = ["TX_CHEMO", "TX_HORMONE", "TX_RADIATION", "TX_PHARMACEUTICAL"]
    Xtr = pd.concat(
        [
            Xtr,
            _suffix_cols(
                meta_tr_g.set_index("SAMPLE_KEY")[tx_cols].astype(float), "clinbin"
            ),
        ],
        axis=1,
    )
    Xte = pd.concat(
        [
            Xte,
            _suffix_cols(
                meta_te_g.set_index("SAMPLE_KEY")[tx_cols].astype(float), "clinbin"
            ),
        ],
        axis=1,
    )

    for col, prefix in [
        ("STAGE_CAT", "STAGE"),
        ("GRADE_CAT", "GRADE"),
        ("AJCC_T_CAT", "AJCC_T"),
    ]:
        levels = _cat_levels_train(meta_tr_g[col])
        A = _ohe_no_mean_median_fill(meta_tr_g[col], levels, f"clinbin::{prefix}")
        B = _ohe_no_mean_median_fill(meta_te_g[col], levels, f"clinbin::{prefix}")
        A.index = meta_tr_g["SAMPLE_KEY"].tolist()
        B.index = meta_te_g["SAMPLE_KEY"].tolist()
        Xtr = pd.concat([Xtr, A], axis=1)
        Xte = pd.concat([Xte, B], axis=1)

    if USE_MUTATION and USE_TMB and tmb_by_patient is not None:
        tr_pid = meta_tr_g["PATIENT_ID"].tolist()
        te_pid = meta_te_g["PATIENT_ID"].tolist()
        tr_tmb = tmb_by_patient.loc[tr_pid, "LOG1P_TMB"].astype(float)
        te_tmb = tmb_by_patient.loc[te_pid, "LOG1P_TMB"].astype(float)
        mu = float(tr_tmb.mean())
        sd = float(tr_tmb.std(ddof=0))
        if (not np.isfinite(sd)) or sd < 1e-8:
            sd = 1.0
        Xtr["tmb::LOG1P_TMB"] = (tr_tmb.to_numpy() - mu) / sd
        Xte["tmb::LOG1P_TMB"] = (te_tmb.to_numpy() - mu) / sd

    if (
        USE_MUTATION
        and USE_MUTATION_PROFILE
        and mut_tr is not None
        and mut_te is not None
    ):
        if cancer_code not in CANCER_CONFIGS:
            raise ValueError("cancer_code is required when adding cancer-specific mutation covariates")
        mut_prefix = f"mut_specific_{cancer_code}"
        Xtr = pd.concat([Xtr, _suffix_cols(mut_tr, mut_prefix)], axis=1)
        Xte = pd.concat([Xte, _suffix_cols(mut_te, mut_prefix)], axis=1)

    return Xtr, Xte


# =============================================================================
# SPLIT WORKFLOW
# =============================================================================


def load_all_metadata_and_modalities(feature_files):
    parts = []
    source_manifest = []
    modalities = {code: {} for code in CANCER_CONFIGS}

    for code, cfg in CANCER_CONFIGS.items():
        meta, info = load_one_cancer_metadata(code, cfg)
        target_samples = meta["SAMPLE_KEY"].tolist()
        target_patients = meta["PATIENT_ID"].tolist()

        if USE_PATHWAY_ACTIVITY:
            p = feature_files[code]["pathway_activity"]
            info["pathway_activity_file"] = str(p)
            modalities[code]["pathway_activity"] = load_sample_feature_matrix(
                p, target_samples, binary=False, name=f"{code}_PATHWAY_ACTIVITY"
            )

        if USE_PATHWAY_CONSISTENCY_FILTER:
            p = feature_files[code]["pathway_consistency"]
            info["pathway_consistency_file"] = str(p)
            modalities[code]["pathway_consistency"] = load_sample_feature_matrix(
                p, target_samples, binary=False, name=f"{code}_PATHWAY_CONSISTENCY"
            )

        if USE_UDP:
            p = feature_files[code]["udp"]
            info["udp_file"] = str(p)
            modalities[code]["udp"] = load_sample_feature_matrix(
                p, target_samples, binary=False, name=f"{code}_UDP"
            )

        if USE_MUTATION and USE_TMB:
            p = feature_files[code]["tmb"]
            info["tmb_file"] = str(p)
            modalities[code]["tmb"] = load_tmb_table(p, target_patients, target_samples)

        if USE_MUTATION and USE_MUTATION_PROFILE:
            p = feature_files[code]["mutation_profile"]
            info["mutation_profile_file"] = str(p)
            modalities[code]["mutation_profile"] = load_mutation_profile_matrix(
                p, target_patients
            )

        # Required modality alignment. No missing modality imputation.
        keep = pd.Series(True, index=meta.index)
        if USE_PATHWAY_ACTIVITY:
            keep &= (
                meta["SAMPLE_KEY"]
                .isin(modalities[code]["pathway_activity"].index)
                .to_numpy()
            )
        if USE_PATHWAY_CONSISTENCY_FILTER:
            keep &= (
                meta["SAMPLE_KEY"]
                .isin(modalities[code]["pathway_consistency"].index)
                .to_numpy()
            )
        if USE_UDP:
            keep &= meta["SAMPLE_KEY"].isin(modalities[code]["udp"].index).to_numpy()
        if USE_MUTATION and USE_TMB:
            keep &= meta["PATIENT_ID"].isin(modalities[code]["tmb"].index).to_numpy()
        if USE_MUTATION and USE_MUTATION_PROFILE:
            keep &= (
                meta["PATIENT_ID"]
                .isin(modalities[code]["mutation_profile"].index)
                .to_numpy()
            )

        ph(f"{code} / REQUIRED MODALITY ALIGNMENT AUDIT")
        print(f"rows_before_required_modality_alignment : {len(meta)}")
        print(f"rows_after_required_modality_alignment  : {int(keep.sum())}")
        print(f"rows_removed_required_modality_alignment: {int((~keep).sum())}")
        if USE_PATHWAY_ACTIVITY:
            print(
                f"rows_with_pathway_activity             : {int(meta['SAMPLE_KEY'].isin(modalities[code]['pathway_activity'].index).sum())}"
            )
        if USE_PATHWAY_CONSISTENCY_FILTER:
            print(
                f"rows_with_pathway_consistency          : {int(meta['SAMPLE_KEY'].isin(modalities[code]['pathway_consistency'].index).sum())}"
            )
        if USE_MUTATION and USE_TMB:
            print(
                f"rows_with_tmb                          : {int(meta['PATIENT_ID'].isin(modalities[code]['tmb'].index).sum())}"
            )
        if USE_MUTATION and USE_MUTATION_PROFILE:
            print(
                f"rows_with_mutation_profile             : {int(meta['PATIENT_ID'].isin(modalities[code]['mutation_profile'].index).sum())}"
            )

        meta = meta.loc[keep].copy().reset_index(drop=True)
        info["n_rows_after_required_modality_alignment"] = int(len(meta))
        parts.append(meta)
        source_manifest.append(info)

    meta = pd.concat(parts, axis=0, ignore_index=True)
    before_outcome = len(meta)
    meta = meta[meta["PFI_EVENT"].notna() & meta["PFI_TIME_DAYS"].notna()].copy()
    before_age = len(meta)
    meta = meta[meta["AGE"].notna()].copy()
    meta = meta.sort_values(["CANCER_CODE", "PATIENT_ID", "SAMPLE_KEY"]).reset_index(
        drop=True
    )
    ph("ALL CANCERS / FINAL PRE-SPLIT COHORT AUDIT")
    print(f"rows_before_outcome_filter : {before_outcome}")
    print(f"rows_after_outcome_filter  : {before_age}")
    print(f"rows_after_age_filter      : {len(meta)}")
    pc("Final pre-split cancer groups", meta["GROUP_LABEL"])
    pc("Final pre-split PFI event", meta["PFI_EVENT"].astype(int))
    tx_cols = [
        "TX_EXPLICIT_NO_TREATMENT",
        "TX_CHEMO",
        "TX_HORMONE",
        "TX_RADIATION",
        "TX_PHARMACEUTICAL",
    ]
    available_tx_cols = [c for c in tx_cols if c in meta.columns]
    if available_tx_cols:
        print("Final pre-split treatment flag positive counts:")
        print(meta[available_tx_cols].sum(axis=0).astype(int).to_string())
    return meta, modalities, source_manifest


def get_modality_matrix(modalities, meta_subset, family):
    frames = []
    for code in CANCER_CONFIGS:
        m = meta_subset[meta_subset["CANCER_CODE"] == code]
        if len(m) == 0:
            continue
        if family in {"pathway_activity", "pathway_consistency", "udp"}:
            X = modalities[code][family]
            rows = m["SAMPLE_KEY"].tolist()
            Xi = X.loc[rows].copy()
            Xi.attrs = {}
            frames.append(Xi)
        elif family == "mutation_profile":
            Xp = modalities[code][family]
            rows = m["PATIENT_ID"].tolist()
            Xm = Xp.loc[rows].copy()
            Xm.index = m["SAMPLE_KEY"].tolist()
            # Important: the mutation matrix stores its parse report in attrs.
            # Pandas may try to compare attrs during concat and fail because the
            # reports are DataFrames with different labels.  Keep attrs only on
            # the original modality object, not on split matrices used for concat.
            Xm.attrs = {}
            frames.append(Xm)
        else:
            raise ValueError(f"Unknown family: {family}")
    if not frames:
        return pd.DataFrame(index=meta_subset["SAMPLE_KEY"].tolist())
    # Use common feature columns across cancers for pooled screens.
    common = (
        sorted(set.intersection(*[set(f.columns) for f in frames]))
        if len(frames) > 1
        else list(frames[0].columns)
    )
    pieces = []
    for f in frames:
        part = f.loc[:, common].copy()
        part.attrs = {}
        pieces.append(part)
    out = pd.concat(pieces, axis=0)
    out.attrs = {}
    out.index = meta_subset["SAMPLE_KEY"].tolist()
    print(
        f"[{family} pooled matrix] rows={out.shape[0]} features={out.shape[1]} common_across_cancers={len(common)}"
    )
    if family == "mutation_profile" and out.shape[1] == 0:
        raise ValueError(
            "No common mutation genes remained across cancers for shared mutation-profile screening. "
            "Keep cancer-specific mutation screening, or relax the pooled shared mutation screen if this is unexpected."
        )
    return out


def select_shared_features(meta_tr, meta_te, modalities, family, top_k):
    Xtr_raw = get_modality_matrix(modalities, meta_tr, family)
    Xte_raw = get_modality_matrix(modalities, meta_te, family)

    if family == "udp":
        Xtr_raw, Xte_raw, variance_report = filter_udp_by_train_variance(
            Xtr_raw, Xte_raw, UDP_VAR_TOP_N
        )
    elif family == "mutation_profile":
        Xtr_raw, Xte_raw, variance_report = filter_mutations_by_train_prevalence(
            Xtr_raw, Xte_raw
        )
    else:
        variance_report = pd.DataFrame()

    if family == "mutation_profile":
        # Mutation profile is sparse binary. In this nonmissingness workflow, patients without mutation-profile rows are removed before splitting.
        Xtr = Xtr_raw.copy()
        Xte = Xte_raw.copy()
        scaler = pd.DataFrame()
    else:
        Xtr, Xte, scaler = standardize_train_test(Xtr_raw, Xte_raw)

    rank = cox_rank_univariate(
        Xtr,
        meta_tr["PFI_TIME_MONTHS"].to_numpy(float),
        meta_tr["PFI_EVENT"].to_numpy(int),
        strata=meta_tr["CANCER_CODE"].tolist(),
        penalizer=COX_PENALIZER,
    )

    # Direction consistency from within-cancer Cox coefficients.
    per_cancer_coefs = {}
    for code in CANCER_CONFIGS:
        idx = meta_tr["CANCER_CODE"] == code
        Xc = Xtr.loc[meta_tr.loc[idx, "SAMPLE_KEY"].tolist()]
        rc = cox_rank_univariate(
            Xc,
            meta_tr.loc[idx, "PFI_TIME_MONTHS"].to_numpy(float),
            meta_tr.loc[idx, "PFI_EVENT"].to_numpy(int),
            strata=None,
            penalizer=COX_PENALIZER,
        )
        per_cancer_coefs[code] = dict(zip(rc["feat"], rc["coef"]))
    rank_dir = direction_consistency_filter(rank, per_cancer_coefs)

    if family == "pathway_activity" and USE_PATHWAY_CONSISTENCY_FILTER:
        Ctr = get_modality_matrix(modalities, meta_tr, "pathway_consistency")
        Cte = get_modality_matrix(modalities, meta_te, "pathway_consistency")
        Ctr = Ctr.loc[
            :, [c for c in Ctr.columns if c in Xtr.columns and c in Cte.columns]
        ]
        selected, consistency_scores = apply_consistency_filter(rank_dir, Ctr, top_k)
    else:
        selected = rank_dir.head(top_k).copy()
        consistency_scores = pd.DataFrame()

    feats = selected["feat"].tolist()
    ph(f"SHARED FEATURE SELECTION AUDIT / {family}")
    print(f"raw_train_shape_after_filtering : {Xtr.shape}")
    print(f"raw_test_shape_after_filtering  : {Xte.shape}")
    print(f"requested_top_k                 : {top_k}")
    print(f"selected_feature_count          : {len(feats)}")
    if not rank.empty and "cox_status" in rank.columns:
        print("cox_status_counts:")
        print(rank["cox_status"].value_counts(dropna=False).to_string())
    print(f"selected_features_preview       : {feats[:12]}")
    Xtr_sel = Xtr.loc[:, feats].copy()
    Xte_sel = Xte.loc[:, feats].copy()
    return Xtr_sel, Xte_sel, rank, selected, scaler, variance_report, consistency_scores


def select_specific_features(
    meta_tr_g, meta_te_g, modalities, code, family, top_k, shared_exclude=None
):
    shared_exclude = set(shared_exclude or [])
    if family in {"pathway_activity", "pathway_consistency", "udp"}:
        Xtr_raw = modalities[code][family].loc[meta_tr_g["SAMPLE_KEY"].tolist()].copy()
        Xte_raw = modalities[code][family].loc[meta_te_g["SAMPLE_KEY"].tolist()].copy()
        Xtr_raw.attrs = {}
        Xte_raw.attrs = {}
    elif family == "mutation_profile":
        Xtr_raw = (
            modalities[code][family].reindex(meta_tr_g["PATIENT_ID"].tolist()).copy()
        )
        Xtr_raw.index = meta_tr_g["SAMPLE_KEY"].tolist()
        Xte_raw = (
            modalities[code][family].reindex(meta_te_g["PATIENT_ID"].tolist()).copy()
        )
        Xte_raw.index = meta_te_g["SAMPLE_KEY"].tolist()
        Xtr_raw.attrs = {}
        Xte_raw.attrs = {}
    else:
        raise ValueError(f"Unknown family: {family}")

    if family == "udp":
        Xtr_raw, Xte_raw, variance_report = filter_udp_by_train_variance(
            Xtr_raw, Xte_raw, UDP_VAR_TOP_N
        )
    elif family == "mutation_profile":
        Xtr_raw, Xte_raw, variance_report = filter_mutations_by_train_prevalence(
            Xtr_raw, Xte_raw
        )
    else:
        variance_report = pd.DataFrame()

    if family == "mutation_profile":
        Xtr = Xtr_raw.copy()
        Xte = Xte_raw.copy()
        scaler = pd.DataFrame()
    else:
        Xtr, Xte, scaler = standardize_train_test(Xtr_raw, Xte_raw)

    rank = cox_rank_univariate(
        Xtr,
        meta_tr_g["PFI_TIME_MONTHS"].to_numpy(float),
        meta_tr_g["PFI_EVENT"].to_numpy(int),
        strata=None,
        penalizer=COX_PENALIZER,
    )
    rank = rank[~rank["feat"].isin(shared_exclude)].copy()

    if family == "pathway_activity" and USE_PATHWAY_CONSISTENCY_FILTER:
        Ctr = (
            modalities[code]["pathway_consistency"]
            .loc[meta_tr_g["SAMPLE_KEY"].tolist()]
            .copy()
        )
        Ctr = Ctr.loc[:, [c for c in Ctr.columns if c in Xtr.columns]]
        selected, consistency_scores = apply_consistency_filter(rank, Ctr, top_k)
    else:
        selected = rank.head(top_k).copy()
        consistency_scores = pd.DataFrame()

    feats = selected["feat"].tolist()
    ph(f"{code} / SPECIFIC FEATURE SELECTION AUDIT / {family}")
    print(f"raw_train_shape_after_filtering : {Xtr.shape}")
    print(f"raw_test_shape_after_filtering  : {Xte.shape}")
    print(f"requested_top_k                 : {top_k}")
    print(f"shared_exclude_count            : {len(shared_exclude)}")
    print(f"selected_feature_count          : {len(feats)}")
    if not rank.empty and "cox_status" in rank.columns:
        print("cox_status_counts:")
        print(rank["cox_status"].value_counts(dropna=False).to_string())
    print(f"selected_features_preview       : {feats[:12]}")
    Xtr_sel = Xtr.loc[:, feats].copy()
    Xte_sel = Xte.loc[:, feats].copy()
    return Xtr_sel, Xte_sel, rank, selected, scaler, variance_report, consistency_scores


def run_one_split(split_seed=SPLIT_SEED):
    split_seed = int(split_seed)
    validate_august_outdir(OUTDIR)
    split_outdir = OUTDIR / f"split_seed{split_seed}{OUTPUT_SUFFIX}"
    split_outdir.mkdir(parents=True, exist_ok=True)

    feature_files = resolve_feature_files()
    meta, modalities, source_manifest = load_all_metadata_and_modalities(feature_files)

    provenance_rows = []
    duplicate_clinical_rows = []
    for _code, _cfg in CANCER_CONFIGS.items():
        configured = {"survival": _cfg["survival_candidates"][0], "clinical": _cfg["clinical_candidates"][0],
                      "treatment": _cfg["treatment_path"], **feature_files[_code]}
        for _role, _path in configured.items():
            provenance_rows.append(file_provenance(_path, _role, _code))
        for _alt in _cfg.get("clinical_candidates", [])[1:]:
            if Path(_alt).exists():
                _row = file_provenance(_alt, "duplicate_clinical_export", _code)
                _row["selected_primary_path"] = str(_cfg["clinical_candidates"][0])
                duplicate_clinical_rows.append(_row)
    pd.DataFrame(provenance_rows).to_csv(split_outdir / "input_file_provenance.csv", index=False)
    pd.DataFrame([{"configuration": CONFIG_NAME, "cancer": c,
                   "sample_field": "sample", "patient_field": "_PATIENT",
                   "PFI_event_field": "PFI", "PFI_time_field": "PFI.time",
                   "PFI_time_input_unit": "days", "PFI_time_model_unit": "months",
                   "days_per_month": 30.4375,
                   "survival_file": str(cfg["survival_candidates"][0])}
                  for c, cfg in CANCER_CONFIGS.items()]).to_csv(
        split_outdir / "pfi_field_provenance.csv", index=False)
    duplicate_clinical_columns = [
        "configuration", "cancer", "role", "exact_path", "sha256",
        "file_size_bytes", "modification_utc", "row_count", "column_count",
        "selected_columns_schema", "selected_primary_path",
    ]
    pd.DataFrame(
        duplicate_clinical_rows, columns=duplicate_clinical_columns
    ).to_csv(split_outdir / "duplicate_clinical_export_audit.csv", index=False)
    meta[[c for c in ["CANCER_CODE", "PATIENT_ID", "SAMPLE_KEY", "PFI_TIME", "PFI_EVENT"] if c in meta]].to_csv(
        split_outdir / "sample_to_patient_mapping.csv", index=False)
    _txcols = [c for c in meta.columns if c.startswith("TX_")]
    meta[["CANCER_CODE", "PATIENT_ID", "SAMPLE_KEY"] + _txcols].to_csv(split_outdir / "treatment_timing_audit.csv", index=False)
    meta.groupby("CANCER_CODE", as_index=False).agg(final_patients=("PATIENT_ID", "nunique"), pfi_events=("PFI_EVENT", "sum")).assign(split_seed=split_seed).to_csv(
        split_outdir / "cohort_attrition_table.csv", index=False)
    pd.DataFrame(source_manifest).to_csv(split_outdir / "cross_modality_overlap_audit.csv", index=False)

    # Save mutation-event-to-gene parse reports for auditability.
    if USE_MUTATION and USE_MUTATION_PROFILE:
        for _code in CANCER_CONFIGS:
            _mpr = modalities.get(_code, {}).get("mutation_profile")
            _rep = (
                getattr(_mpr, "attrs", {}).get("mutation_parse_report", pd.DataFrame())
                if _mpr is not None
                else pd.DataFrame()
            )
            if isinstance(_rep, pd.DataFrame) and not _rep.empty:
                _rep.to_csv(
                    split_outdir
                    / f"tcga_{_code.lower()}_mutation_profile_gene_parse_report{OUTPUT_SUFFIX}.csv",
                    index=False,
                )
                _rep.assign(cancer=_code, split_seed=split_seed).to_csv(split_outdir / f"mutation_event_parser_audit_{_code}.csv", index=False)
                _rep[_rep["kept"].astype(bool)].head(100).assign(cancer=_code).to_csv(split_outdir / f"retained_mutation_events_examples_{_code}.csv", index=False)
                _rep[~_rep["kept"].astype(bool)].head(100).assign(cancer=_code).to_csv(split_outdir / f"dropped_mutation_events_examples_{_code}.csv", index=False)
        _parser_files = sorted(split_outdir.glob("mutation_event_parser_audit_*.csv"))
        if _parser_files:
            _parser_all = pd.concat([pd.read_csv(x) for x in _parser_files], ignore_index=True)
            _parser_all.to_csv(split_outdir / "mutation_event_parser_audit.csv", index=False)
            _parser_all.groupby(["cancer", "kept"], dropna=False).size().reset_index(name="mutation_event_count").to_csv(
                split_outdir / "mutation_event_parser_counts.csv", index=False)
            pd.concat([pd.read_csv(x) for x in split_outdir.glob("retained_mutation_events_examples_*.csv")], ignore_index=True).to_csv(split_outdir / "retained_mutation_events_examples.csv", index=False)
            pd.concat([pd.read_csv(x) for x in split_outdir.glob("dropped_mutation_events_examples_*.csv")], ignore_index=True).to_csv(split_outdir / "dropped_mutation_events_examples.csv", index=False)
        for _code in CANCER_CONFIGS:
            _tmb = modalities.get(_code, {}).get("tmb")
            if _tmb is not None:
                for _attr, _name in [("duplicate_aliquot_tmb_audit", "duplicate_aliquot_tmb_audit"), ("tmb_disagreement_by_patient", "tmb_disagreement_by_patient"), ("tmb_sample_selection_rule", "tmb_sample_selection_rule")]:
                    _df = getattr(_tmb, "attrs", {}).get(_attr, pd.DataFrame())
                    if isinstance(_df, pd.DataFrame):
                        _df.assign(cancer=_code).to_csv(split_outdir / f"{_name}_{_code}.csv", index=False)
        for _name in ["duplicate_aliquot_tmb_audit", "tmb_disagreement_by_patient", "tmb_sample_selection_rule"]:
            _files = sorted(split_outdir.glob(f"{_name}_*.csv"))
            if _files:
                pd.concat([pd.read_csv(x) for x in _files], ignore_index=True).to_csv(
                    split_outdir / f"{_name}.csv", index=False)

    ph("ALL CANCERS / REQUIRED DATA AFTER ALIGNMENT")
    pc("Cancer groups", meta["GROUP_LABEL"])
    pc("PFI event", meta["PFI_EVENT"].astype(int))

    strat = (
        meta["GROUP_LABEL"].astype(str)
        + "__"
        + meta["PFI_EVENT"].astype(int).astype(str)
    )
    tr_idx, te_idx = next(
        StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_FRAC, random_state=split_seed
        ).split(np.zeros((len(meta), 1)), strat)
    )
    meta["split"] = ""
    meta.loc[tr_idx, "split"] = "train"
    meta.loc[te_idx, "split"] = "test"
    meta_tr = meta.iloc[tr_idx].copy().reset_index(drop=True)
    meta_te = meta.iloc[te_idx].copy().reset_index(drop=True)

    ph("GLOBAL SPLIT COUNTS")
    print(f"[SPLIT] seed    = {split_seed}")
    print(f"[SPLIT] train n = {len(meta_tr)}")
    print(f"[SPLIT] test  n = {len(meta_te)}")
    pc("Train cancer groups", meta_tr["GROUP_LABEL"])
    pc("Test cancer groups", meta_te["GROUP_LABEL"])

    # -------------------------------------------------------------------------
    # Shared all-cancer latent GP inputs
    # -------------------------------------------------------------------------
    shared_blocks_tr = []
    shared_blocks_te = []
    shared_selected_by_family = {}
    reports = []

    if USE_PATHWAY_ACTIVITY:
        ph("ALL-CANCER LATENT GP / PATHWAY ACTIVITY SELECTION")
        Xtr, Xte, rank, selected, scaler, var_report, cons = select_shared_features(
            meta_tr, meta_te, modalities, "pathway_activity", SHARED_TOP_K_PATHWAY
        )
        shared_blocks_tr.append(_suffix_cols(Xtr, "act_shared"))
        shared_blocks_te.append(_suffix_cols(Xte, "act_shared"))
        shared_selected_by_family["pathway_activity"] = selected["feat"].tolist()
        rank.to_csv(
            split_outdir / f"shared_pathway_activity_cox_ranking{OUTPUT_SUFFIX}.csv",
            index=False,
        )
        selected.to_csv(
            split_outdir / f"shared_pathway_activity_selected{OUTPUT_SUFFIX}.csv",
            index=False,
        )
        scaler.to_csv(
            split_outdir / f"shared_pathway_activity_scaler{OUTPUT_SUFFIX}.csv",
            index=False,
        )
        if not cons.empty:
            cons.to_csv(
                split_outdir / f"shared_pathway_consistency_scores{OUTPUT_SUFFIX}.csv",
                index=False,
            )

    if USE_UDP:
        ph("ALL-CANCER LATENT GP / UDP SELECTION")
        Xtr, Xte, rank, selected, scaler, var_report, cons = select_shared_features(
            meta_tr, meta_te, modalities, "udp", SHARED_TOP_K_UDP
        )
        shared_blocks_tr.append(_suffix_cols(Xtr, "udp_shared"))
        shared_blocks_te.append(_suffix_cols(Xte, "udp_shared"))
        shared_selected_by_family["udp"] = selected["feat"].tolist()
        rank.to_csv(
            split_outdir / f"shared_udp_cox_ranking{OUTPUT_SUFFIX}.csv", index=False
        )
        selected.to_csv(
            split_outdir / f"shared_udp_selected{OUTPUT_SUFFIX}.csv", index=False
        )
        scaler.to_csv(
            split_outdir / f"shared_udp_scaler{OUTPUT_SUFFIX}.csv", index=False
        )
        var_report.to_csv(
            split_outdir / f"shared_udp_train_variance_filter{OUTPUT_SUFFIX}.csv",
            index=False,
        )

    Xtr_shared_all = (
        pd.concat(shared_blocks_tr, axis=1)
        if shared_blocks_tr
        else pd.DataFrame(index=meta_tr["SAMPLE_KEY"])
    )
    Xte_shared_all = (
        pd.concat(shared_blocks_te, axis=1)
        if shared_blocks_te
        else pd.DataFrame(index=meta_te["SAMPLE_KEY"])
    )

    # -------------------------------------------------------------------------
    # Per-cancer outputs
    # -------------------------------------------------------------------------
    mutation_selected_shared = []
    if USE_MUTATION and USE_MUTATION_PROFILE:
        ph("MUTATION FEATURE BLOCK / SHARED MUTATION PROFILE SELECTION")
        Xtr, Xte, rank, selected, scaler, var_report, cons = select_shared_features(
            meta_tr, meta_te, modalities, "mutation_profile", SHARED_TOP_K_MUTATION
        )
        mutation_selected_shared = selected["feat"].tolist()
        rank.to_csv(
            split_outdir / f"shared_mutation_profile_cox_ranking{OUTPUT_SUFFIX}.csv",
            index=False,
        )
        selected.to_csv(
            split_outdir / f"shared_mutation_profile_selected{OUTPUT_SUFFIX}.csv",
            index=False,
        )
        var_report.to_csv(
            split_outdir
            / f"shared_mutation_profile_prevalence_filter{OUTPUT_SUFFIX}.csv",
            index=False,
        )
        # Keep shared mutation-profile selections in the all-cancer/shared latent
        # input block, not only in the downstream covariate block.  This makes
        # input-origin shared/cancer-specific tagging consistent with the DGP
        # posthoc logic (mut_shared:: versus mut_specific_<cancer>::).
        shared_blocks_tr.append(_suffix_cols(Xtr, "mut_shared"))
        shared_blocks_te.append(_suffix_cols(Xte, "mut_shared"))
        shared_selected_by_family["mutation_profile"] = mutation_selected_shared

    # Rebuild after optional shared mutation-profile addition.
    Xtr_shared_all = (
        pd.concat(shared_blocks_tr, axis=1)
        if shared_blocks_tr
        else pd.DataFrame(index=meta_tr["SAMPLE_KEY"])
    )
    Xte_shared_all = (
        pd.concat(shared_blocks_te, axis=1)
        if shared_blocks_te
        else pd.DataFrame(index=meta_te["SAMPLE_KEY"])
    )

    dedup_audit_rows = []
    tmb_scaling_rows = []
    for code, cfg in CANCER_CONFIGS.items():
        tr = meta_tr[meta_tr["CANCER_CODE"] == code].copy().reset_index(drop=True)
        te = meta_te[meta_te["CANCER_CODE"] == code].copy().reset_index(drop=True)
        if len(tr) == 0 or len(te) == 0:
            continue

        ph(f"{code} / CANCER-SPECIFIC LATENT GP FEATURE SELECTION")
        specific_blocks_tr = []
        specific_blocks_te = []
        selected_specific = {}

        if USE_PATHWAY_ACTIVITY:
            Xtr, Xte, rank, selected, scaler, var_report, cons = (
                select_specific_features(
                    tr,
                    te,
                    modalities,
                    code,
                    "pathway_activity",
                    SPECIFIC_TOP_K_PATHWAY,
                    shared_exclude=shared_selected_by_family.get(
                        "pathway_activity", []
                    ),
                )
            )
            specific_blocks_tr.append(_suffix_cols(Xtr, f"act_specific_{code}"))
            specific_blocks_te.append(_suffix_cols(Xte, f"act_specific_{code}"))
            selected_specific["pathway_activity"] = selected["feat"].tolist()
            rank.to_csv(
                split_outdir
                / f"tcga_{code.lower()}_pathway_activity_cox_ranking{OUTPUT_SUFFIX}.csv",
                index=False,
            )
            selected.to_csv(
                split_outdir
                / f"tcga_{code.lower()}_pathway_activity_selected{OUTPUT_SUFFIX}.csv",
                index=False,
            )
            scaler.to_csv(
                split_outdir
                / f"tcga_{code.lower()}_pathway_activity_scaler{OUTPUT_SUFFIX}.csv",
                index=False,
            )
            if not cons.empty:
                cons.to_csv(
                    split_outdir
                    / f"tcga_{code.lower()}_pathway_consistency_scores{OUTPUT_SUFFIX}.csv",
                    index=False,
                )

        if USE_UDP:
            Xtr, Xte, rank, selected, scaler, var_report, cons = (
                select_specific_features(
                    tr,
                    te,
                    modalities,
                    code,
                    "udp",
                    SPECIFIC_TOP_K_UDP,
                    shared_exclude=shared_selected_by_family.get("udp", []),
                )
            )
            specific_blocks_tr.append(_suffix_cols(Xtr, f"udp_specific_{code}"))
            specific_blocks_te.append(_suffix_cols(Xte, f"udp_specific_{code}"))
            selected_specific["udp"] = selected["feat"].tolist()
            rank.to_csv(
                split_outdir
                / f"tcga_{code.lower()}_udp_cox_ranking{OUTPUT_SUFFIX}.csv",
                index=False,
            )
            selected.to_csv(
                split_outdir / f"tcga_{code.lower()}_udp_selected{OUTPUT_SUFFIX}.csv",
                index=False,
            )
            scaler.to_csv(
                split_outdir / f"tcga_{code.lower()}_udp_scaler{OUTPUT_SUFFIX}.csv",
                index=False,
            )
            var_report.to_csv(
                split_outdir
                / f"tcga_{code.lower()}_udp_train_variance_filter{OUTPUT_SUFFIX}.csv",
                index=False,
            )

        Xtr_specific = (
            pd.concat(specific_blocks_tr, axis=1)
            if specific_blocks_tr
            else pd.DataFrame(index=tr["SAMPLE_KEY"])
        )
        Xte_specific = (
            pd.concat(specific_blocks_te, axis=1)
            if specific_blocks_te
            else pd.DataFrame(index=te["SAMPLE_KEY"])
        )

        # Shared latent matrix subset for this cancer.
        Xtr_shared = Xtr_shared_all.loc[tr["SAMPLE_KEY"].tolist()].copy()
        Xte_shared = Xte_shared_all.loc[te["SAMPLE_KEY"].tolist()].copy()

        # Mutation profile selected features are mutation feature-block inputs.
        mut_tr_cov = None
        mut_te_cov = None
        selected_mut_specific = []
        if USE_MUTATION and USE_MUTATION_PROFILE:
            Xtr_m, Xte_m, rank_m, selected_m, scaler_m, var_report_m, cons_m = (
                select_specific_features(
                    tr,
                    te,
                    modalities,
                    code,
                    "mutation_profile",
                    SPECIFIC_TOP_K_MUTATION,
                    shared_exclude=mutation_selected_shared,
                )
            )
            selected_mut_specific = selected_m["feat"].tolist()
            mut_feats = sorted(set(selected_mut_specific) - set(mutation_selected_shared))
            for _f in sorted(set(mutation_selected_shared + selected_mut_specific)):
                dedup_audit_rows.append({"configuration": CONFIG_NAME, "split_seed": split_seed,
                    "cancer": code, "feature": _f, "in_shared_block": _f in mutation_selected_shared,
                    "in_specific_covariate_block": _f in mut_feats,
                    "duplicate_removed": (_f in mutation_selected_shared and _f in selected_mut_specific)})
            Xp = modalities[code]["mutation_profile"]
            mut_tr_cov = Xp.loc[
                tr["PATIENT_ID"].tolist(), [f for f in mut_feats if f in Xp.columns]
            ].copy()
            mut_tr_cov.index = tr["SAMPLE_KEY"].tolist()
            mut_tr_cov.attrs = {}
            mut_te_cov = Xp.loc[
                te["PATIENT_ID"].tolist(), [f for f in mut_feats if f in Xp.columns]
            ].copy()
            mut_te_cov.index = te["SAMPLE_KEY"].tolist()
            mut_te_cov.attrs = {}
            rank_m.to_csv(
                split_outdir
                / f"tcga_{code.lower()}_mutation_profile_cox_ranking{OUTPUT_SUFFIX}.csv",
                index=False,
            )
            selected_m.to_csv(
                split_outdir
                / f"tcga_{code.lower()}_mutation_profile_selected{OUTPUT_SUFFIX}.csv",
                index=False,
            )
            var_report_m.to_csv(
                split_outdir
                / f"tcga_{code.lower()}_mutation_profile_prevalence_filter{OUTPUT_SUFFIX}.csv",
                index=False,
            )

        tmb = modalities[code].get("tmb") if USE_MUTATION and USE_TMB else None
        if tmb is not None:
            _vals = tmb.loc[tr["PATIENT_ID"].tolist(), "LOG1P_TMB"].astype(float)
            tmb_scaling_rows.append({"configuration": CONFIG_NAME, "split_seed": split_seed, "cancer": code,
                "n_train": len(_vals), "log1p_tmb_mean_train": float(_vals.mean()), "log1p_tmb_sd_train": float(_vals.std(ddof=0))})
        Xtr_cov, Xte_cov = build_covariate_block(
            tr, te, tmb_by_patient=tmb, mut_tr=mut_tr_cov, mut_te=mut_te_cov,
            cancer_code=code,
        )

        Xtr_shared.attrs = {}
        Xte_shared.attrs = {}
        Xtr_specific.attrs = {}
        Xte_specific.attrs = {}
        Xtr_cov.attrs = {}
        Xte_cov.attrs = {}
        Xtr_design = pd.concat([Xtr_shared, Xtr_specific, Xtr_cov], axis=1)
        Xte_design = pd.concat([Xte_shared, Xte_specific, Xte_cov], axis=1)
        Xtr_design.attrs = {}
        Xte_design.attrs = {}

        print_design_audit(f"{code} / SHARED LATENT DESIGN", Xtr_shared, Xte_shared)
        print_design_audit(
            f"{code} / CANCER-SPECIFIC LATENT DESIGN", Xtr_specific, Xte_specific
        )
        print_design_audit(
            f"{code} / CLINICAL MUTATION COVARIATE DESIGN", Xtr_cov, Xte_cov
        )
        print_design_audit(f"{code} / FINAL COMBINED DESIGN", Xtr_design, Xte_design)
        assert_no_missing_values(f"{code} train final design", Xtr_design)
        assert_no_missing_values(f"{code} test final design", Xte_design)

        ph(f"{code} / FINAL MATRICES")
        print(f"[shared latent train]   {Xtr_shared.shape}")
        print(f"[specific latent train] {Xtr_specific.shape}")
        print(f"[clinical/mutation block train] {Xtr_cov.shape}")
        print(f"[combined design train] {Xtr_design.shape}")
        print(f"[combined design test]  {Xte_design.shape}")

        tr.to_csv(
            split_outdir / f"tcga_{code.lower()}_metadata_train{OUTPUT_SUFFIX}.csv",
            index=False,
        )
        te.to_csv(
            split_outdir / f"tcga_{code.lower()}_metadata_test{OUTPUT_SUFFIX}.csv",
            index=False,
        )
        Xtr_shared.to_csv(
            split_outdir
            / f"tcga_{code.lower()}_allcancer_latent_train{OUTPUT_SUFFIX}.csv"
        )
        Xte_shared.to_csv(
            split_outdir
            / f"tcga_{code.lower()}_allcancer_latent_test{OUTPUT_SUFFIX}.csv"
        )
        Xtr_specific.to_csv(
            split_outdir
            / f"tcga_{code.lower()}_cancerspecific_latent_train{OUTPUT_SUFFIX}.csv"
        )
        Xte_specific.to_csv(
            split_outdir
            / f"tcga_{code.lower()}_cancerspecific_latent_test{OUTPUT_SUFFIX}.csv"
        )
        Xtr_cov.to_csv(
            split_outdir
            / f"tcga_{code.lower()}_clinical_mutation_features_train{OUTPUT_SUFFIX}.csv"
        )
        Xte_cov.to_csv(
            split_outdir
            / f"tcga_{code.lower()}_clinical_mutation_features_test{OUTPUT_SUFFIX}.csv"
        )
        Xtr_design.to_csv(
            split_outdir / f"tcga_{code.lower()}_design_train{OUTPUT_SUFFIX}.csv"
        )
        Xte_design.to_csv(
            split_outdir / f"tcga_{code.lower()}_design_test{OUTPUT_SUFFIX}.csv"
        )

        reports.append(
            {
                "cancer_code": code,
                "cancer_label": cfg["cancer_label"],
                "train_n": int(len(tr)),
                "test_n": int(len(te)),
                "allcancer_latent_features": list(Xtr_shared.columns),
                "cancerspecific_latent_features": list(Xtr_specific.columns),
                "clinical_mutation_feature_block": list(Xtr_cov.columns),
                "selected_specific": selected_specific,
                "selected_mutation_specific": selected_mut_specific,
            }
        )

    meta.to_csv(
        split_outdir / f"tcga_allcancer_metadata_with_split{OUTPUT_SUFFIX}.csv",
        index=False,
    )
    pd.DataFrame(dedup_audit_rows).to_csv(split_outdir / "mutation_feature_deduplication_audit.csv", index=False)
    pd.DataFrame(tmb_scaling_rows).to_csv(split_outdir / "tmb_training_scaling_parameters_by_seed.csv", index=False)
    _prev_files = sorted(split_outdir.glob("*mutation_profile_prevalence_filter*.csv"))
    if _prev_files:
        pd.concat([pd.read_csv(x).assign(source_file=x.name, split_seed=split_seed) for x in _prev_files], ignore_index=True).to_csv(split_outdir / "mutation_feature_prevalence_by_seed.csv", index=False)
    with open(
        split_outdir / f"tcga_allcancer_preprocess_report{OUTPUT_SUFFIX}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "split_seed": split_seed,
                "groups": ["Breast", "Lung", "Colon"],
                "n_splits_total": len(MULTI_SEEDS) if RUN_MULTI_SEEDS else 1,
                "toggles": {
                    "USE_GENE_EXPRESSION_PCS": USE_GENE_EXPRESSION_PCS,
                    "USE_PATHWAY_ACTIVITY": USE_PATHWAY_ACTIVITY,
                    "USE_PATHWAY_CONSISTENCY_FILTER": USE_PATHWAY_CONSISTENCY_FILTER,
                    "USE_UDP": USE_UDP,
                    "USE_MUTATION": USE_MUTATION,
                    "USE_TMB": USE_TMB,
                    "USE_MUTATION_PROFILE": USE_MUTATION_PROFILE,
                    "REQUIRE_TREATMENT_RECORD": REQUIRE_TREATMENT_RECORD,
                    "REQUIRE_KNOWN_TREATMENT_STATUS": REQUIRE_KNOWN_TREATMENT_STATUS,
                    "MUTATION_PROFILE_COLLAPSE_TO_GENE": MUTATION_PROFILE_COLLAPSE_TO_GENE,
                    "MUTATION_DROP_UNPARSEABLE_EVENTS": MUTATION_DROP_UNPARSEABLE_EVENTS,
                    "MUTATION_MIN_PARSED_FRACTION_WARN": MUTATION_MIN_PARSED_FRACTION_WARN,
                },
                "workflow": (
                    "No gene-expression PC pipeline in the main model. "
                    "Pathway/network activity is screened with pooled stratified Cox for the all-cancer latent GP "
                    "and within-cancer Cox for the cancer-specific latent GP. "
                    "Consistency is used as a selection filter. "
                    "Clinical, treatment, TMB, and filtered gene-level mutation-profile features are input feature blocks with their own DGP kernel prefixes. "
                    "Treatment rows are required and unmatched or uninterpretable treatment records are not treated as untreated. "
                    "Mutation-event matrices are primary-tumor filtered, collapsed to gene-level indicators with parser-based labels, and parse reports are saved for audit. "
                    "Cox screening treats nonconverged fits as failed. No train mean/median missing-value imputation is used."
                ),
                "source_manifest": source_manifest,
                "shared_selected_by_family": shared_selected_by_family,
                "mutation_selected_shared": mutation_selected_shared,
                "reports": reports,
            },
            f,
            indent=2,
        )
    print(f"Saved split outputs under: {split_outdir}")


def run_canonical_harmonization():
    """Outcome-independent exact-path alignment shared by all split jobs."""
    validate_august_outdir(OUTDIR)
    cdir = OUTDIR / "canonical"
    cdir.mkdir(parents=True, exist_ok=True)
    feature_files = resolve_feature_files()
    meta, modalities, source_manifest = load_all_metadata_and_modalities(feature_files)
    rows = []
    for code, cfg in CANCER_CONFIGS.items():
        configured = {"survival": cfg["survival_candidates"][0],
                      "clinical": cfg["clinical_candidates"][0],
                      "treatment": cfg["treatment_path"], **feature_files[code]}
        for role, path in configured.items():
            rows.append(file_provenance(path, role, code))
    pd.DataFrame(rows).to_csv(cdir / "canonical_input_file_provenance.csv", index=False)
    meta.to_csv(cdir / "canonical_aligned_patient_metadata.csv", index=False)
    pd.DataFrame(source_manifest).to_csv(cdir / "canonical_cross_modality_overlap_audit.csv", index=False)
    meta.groupby("CANCER_CODE", as_index=False).agg(
        final_patients=("PATIENT_ID", "nunique"), pfi_events=("PFI_EVENT", "sum")
    ).to_csv(cdir / "canonical_cohort_attrition.csv", index=False)
    (cdir / "canonical_complete.json").write_text(json.dumps({
        "configuration": CONFIG_NAME, "status": "complete",
        "note": "No feature screening, scaling, cutpoint learning, or outcome-dependent selection is performed in canonical phase."
    }, indent=2))



def main():
    validate_august_outdir(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if os.getenv("CANONICAL_ONLY", "0").strip().lower() in {"1", "true", "yes"}:
        run_canonical_harmonization()
        return
    seeds = MULTI_SEEDS if RUN_MULTI_SEEDS else [SPLIT_SEED]
    for s in seeds:
        run_one_split(s)
    print("\nDone.")


if __name__ == "__main__":
    main()
