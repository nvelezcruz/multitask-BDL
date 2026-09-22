#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCGA Workflow C multitask DGP sweep runner for all-cancer preprocessing with Workflow C architecture, UQ additions, and early-failure analysis.

Uses the precomputed repeated outer train/test splits from the all-cancer preprocessing and evaluates Workflow C target configurations.
The meaningful architecture settings are:
- number of survival PCs
- shared latent dimension
- private latent dimension
- multitask shared latent rank

The output layer is group-specific in Workflow C, so output rank is not an architectural sweep dimension here.

No inner validation split is used in this version. Each configuration is fit on
the full outer-train split, checkpointed by training objective, and then screened
on the outer test split.
"""

import json
import os
import copy
import warnings
import re
import shutil
import inspect
import traceback
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from TCGA_treatment_conditioned_WFC_NPJ_August import run_corrected_wfc_partition
import torch.nn.functional as F
import gpytorch
import gpytorch.models.deep_gps
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    VariationalStrategy,
    LMCVariationalStrategy,
    MeanFieldVariationalDistribution,
    CholeskyVariationalDistribution,
)
from gpytorch.means import ZeroMean
from gpytorch.kernels import MaternKernel, ScaleKernel, LinearKernel
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from lifelines.utils import concordance_index as _lifelines_c
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)
# Force Matplotlib's internal mathtext renderer.  External LaTeX breaks on
# literal feature labels such as STAGE_IIIA and patient labels containing #.
plt.rcParams["text.usetex"] = False

warnings.filterwarnings("ignore", category=UserWarning)


class ExplicitFallbackWarning(RuntimeWarning):
    """Raised when the code changes grouping semantics via an explicit fallback path."""


warnings.simplefilter("always", ExplicitFallbackWarning)


def _warn_explicit_fallback(msg):
    warnings.warn(str(msg), ExplicitFallbackWarning, stacklevel=2)
    print(f"[WARNING] {msg}", flush=True)


# PATCH NOTES (2026-05-13):
# - Added robust chunked posterior score sampling with jitter retries for FC stability.
# - Added H=24 to FC, branch decomposition, treatment-effect, and treatment-conditioned outputs.
# - Replaced H=3 early-failure top-k diagnostics with H=24.
# - Fixed KM display-window median annotations so off-axis medians do not shrink figures.
# - Added AUROC/AUPRC side annotations to ROC/PR plots.
# - Treatment-conditioned FC now saves all-feature plots plus separate candidate-family plots.

# PATHWAY+CLINICAL 10RS VERSION (2026-06-01):
# - Uses processed clinical + pathway activity/consistency features only.
# - No UDP, no TMB, no sparse mutation-profile block, no gene-expression PCs.
# - Adds repeated-split feature-stratified treatment-risk contrasts and direction-stability summaries.
# - Keeps treatment-adjusted FC, treatment-conditioned FC, and branch FC; saved generic overall FC and FC-driven PC backprojection are disabled.

# NPJ corrected full run: preserve all original evaluation horizons; run the single corrected treatment-conditioned WFC at H=12,18,24,36 from saved artifacts.
# ------------------------ USER KNOBS ------------------------
# NPJ validation patch: repeated outer seeds remain enabled for predictive-performance
# stability; posthoc calibration uses 4-fold cross-fitting by default to match
# the referenced NPJ-style protocol; held-out early-failure metric variability
# uses 1000 bootstrap resamples by default.
CONFIG_NAME = "PathClin"
RUN_LABEL = "PathClin_NPJ_August"
SEED = 43
CV_MODE = False
REPEATED_SPLIT_MODE = True
RUN_OAT_SENSITIVITY = False
INNER_VAL_FRAC = 0.20
MULTI_SEEDS = list(range(42, 52))  # 10 repeated outer splits: 42 through 51

TCGA_ROOT = Path(os.environ["TCGA_ROOT"]).expanduser().resolve()
PREP_OUTDIR = Path(
    os.getenv(
        "PREP_OUTDIR",
        str(TCGA_ROOT / "NPJ_August_2026_FULLRERUN_PathClin_pathway_clinical_treatment_MT2" / "preprocessing"),
    )
)
OUTDIR = Path(
    os.getenv(
        "OUTDIR",
        str(TCGA_ROOT / "NPJ_August_2026_FULLRERUN_PathClin_pathway_clinical_treatment_MT2" / "seed42"),
    )
)
def validate_august_outdir(path):
    p = Path(path).expanduser().resolve()
    allowed_roots = [
        Path(os.getenv("NPJ_SCRATCH_ROOT", str(TCGA_ROOT))).expanduser().resolve(),
        TCGA_ROOT,
    ]
    if not any(p == root or root in p.parents for root in allowed_roots):
        raise ValueError(f"Unsafe OUTDIR outside approved roots {allowed_roots}: {p}")
    if "august" not in str(p).lower() or CONFIG_NAME.lower() not in str(p).lower():
        raise ValueError(f"Unsafe OUTDIR must contain August and {CONFIG_NAME}: {p}")
    return p


validate_august_outdir(OUTDIR)
OUTDIR.mkdir(parents=True, exist_ok=True)

ALL_CANCER_CODES = ["BRCA", "LUAD", "COAD"]

P_XTR = PREP_OUTDIR / "tcga_brca_design_train.csv"
P_XTE = PREP_OUTDIR / "tcga_brca_design_test.csv"
P_META_TR = PREP_OUTDIR / "tcga_brca_metadata_train.csv"
P_META_TE = PREP_OUTDIR / "tcga_brca_metadata_test.csv"

# Sweep grid
SWEEP_SURV_PC_LIST = [0]  # clinical + pathway design; no gene-expression PCs
SWEEP_SHARED_DIM_LIST = [2]
SWEEP_PRIVATE_DIM_LIST = [2]
SWEEP_MT_LATENT_RANK_LIST = [2]
SWEEP_OUTPUT_RANK_LIST = [1]

NUM_INDUCING = 64
REQUIRE_MT_BOTH = True
VERIFY_MT_ARCHITECTURE = False
VI_DIST = "meanfield"

EPOCHS = 900
PRINT_EVERY = 50
GRAD_CLIP = 1.0
KL_WARMUP = int(0.30 * EPOCHS)
LR_VAR = 0.02
LR_IND = 0.01
LR_HYP = 0.003
MC_TRAIN = 32

DT_BINS = 12
USE_MANUAL_DT_EDGES = True
DT_EDGES_MANUAL = [0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0, 48.0, 60.0, 84.0, 120.0]
EVAL_HORIZONS = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0]
EARLY_FAILURE_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0]
PRIMARY_EARLY_FAILURE_H = 6.0
USE_EARLY_BIN_WEIGHTS = False
EARLY_BIN_WEIGHT_GAMMA = 2.0
EARLY_BIN_WEIGHT_TAU = 6.0

# ------------------------ POSTHOC PLOTTING / FC ------------------------
POSTERIOR_H_MONTHS = 6.0
POSTERIOR_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0]
TAU_HIGH = 0.50
TAU_LOW = 0.85
P_STAR = 0.60
POST_GROUP_MIN_COUNT = 10
FINAL_EVAL_MC_SAMPLES = 256
FINAL_PRED_BAND_MC = 256
REPRESENTATIVE_PATIENTS_N = 3
REPRESENTATIVE_TRAJ_H_MAX = 60.0
REPRESENTATIVE_TRAJ_STEP = 6.0
REPRESENTATIVE_TRAJ_MC = 256
RISK_UQ_MC = 256
RISK_UQ_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0]
UQ_INTERVAL_Q = (0.05, 0.95)
UQ_CALIBRATION_BINS = 5
UQ_REPRESENTATIVE_DENSITIES_PER_GROUP = 3
BRANCH_DECOMP_MC = 64
BRANCH_DECOMP_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0]
FC_H_MONTHS = 6.0  # primary early-failure FC horizon; secondary FC plots/CSVs include 12, 18, 36, and 60 mo
FC_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0]
FC_S = 256
FC_REPEATS = 4
FC_TOPN = 15
FC_COLLAPSE_MODE = "group_shuffle"
FC_USE_CRN = True
KM_ADMIN_CAP_MONTHS = (
    None  # set to a float only if you explicitly want administrative capping
)
# Additional x-axis-limited KM copies for report figures.  These do not censor
# or change the data; they only truncate the displayed follow-up window so the
# unstable late KM tail does not dominate the figure.
KM_DISPLAY_LIMIT_MONTHS = [60.0, 36.0]
ENABLE_HEATMAPS = True
ENABLE_TREATMENT_COMBO_KM = False
TREATMENT_EFFECT_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0]
TREATMENT_EFFECT_RMST_TAU_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0]
TREATMENT_EFFECT_MC = 256
EARLY_FAILURE_MC = 256
EARLY_FAILURE_TOPK_FRACS = [0.05, 0.10, 0.20, 0.30]
EARLY_FAILURE_CALIBRATION_BINS = 5
EARLY_FAILURE_TOP_PATIENT_N = 30
# Final reporting choice: use the within-cancer top-10 patient posterior
# distribution plots (Breast / Colon / Lung separately) at early-failure horizons
# months, which yields 12 separate figures total.  The pooled top-N plot is kept
# available for debugging, but disabled by default so the saved outputs match the
# intended interpretation more directly.
EARLY_FAILURE_TOP_PATIENT_WITHIN_CANCER_N = 10
EARLY_FAILURE_SAVE_POOLED_TOP_PATIENT_PLOTS = False
EARLY_FAILURE_SAVE_WITHIN_CANCER_TOP_PATIENT_PLOTS = True
# Test-set performance uncertainty for early-failure endpoints.  This resamples
# test patients, not posterior draws.  For pooled estimates, resampling is
# stratified by cancer group and fixed-horizon binary status; within-cancer
# estimates are stratified by fixed-horizon binary status inside each cancer.
EARLY_FAILURE_BOOTSTRAP_N = 1000
EARLY_FAILURE_BOOTSTRAP_Q = (0.05, 0.95)
EARLY_FAILURE_BOOTSTRAP_SEED = 20260512
EARLY_FAILURE_SAVE_BOOTSTRAP_PLOTS = True
# Save separate ROC and precision-recall figures for Breast / Colon / Lung at
# each evaluation horizon, in addition to the pooled test curves.
SAVE_WITHIN_CANCER_AUROC_AUPRC_CURVES = True


# ------------------------ POSTHOC CROSS-FITTED CALIBRATION ------------------------
# Native DGP risks are still reported.  This optional block fits lightweight
# censoring-aware post-hoc recalibration models on out-of-fold predictions from
# the OUTER TRAINING split only, then applies the learned transformation to the
# final full-train DGP predictions on the outer TEST split.  The final DGP is
# still trained on all outer-train patients; no test outcomes are used to fit the
# calibrator.
ENABLE_POSTHOC_RECALIBRATION = True
POSTHOC_CALIBRATION_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0]
POSTHOC_CALIBRATION_K_FOLDS = 5
POSTHOC_CALIBRATION_EPOCHS = EPOCHS
POSTHOC_CALIBRATION_MC_TRAIN = MC_TRAIN
POSTHOC_CALIBRATION_PRED_MC = 128
POSTHOC_CALIBRATION_BINS = 5
POSTHOC_CALIBRATION_EPS = 1e-5
POSTHOC_CALIBRATION_L2 = 1e-4
POSTHOC_CALIBRATION_FORCE_REFIT = False
POSTHOC_CALIBRATION_PRIMARY_KIND = "intercept_only"
POSTHOC_CALIBRATION_MODEL_KINDS = ["intercept_only", "pooled", "group_intercept"]
POSTHOC_CALIBRATION_MIN_N_FOR_ANY_MODEL = 20
POSTHOC_CALIBRATION_MIN_EVENTS_FOR_ANY_MODEL = 5
POSTHOC_CALIBRATION_MIN_NONEVENTS_FOR_ANY_MODEL = 5
POSTHOC_CALIBRATION_MIN_EVENTS_FOR_SLOPE = 20
POSTHOC_CALIBRATION_MIN_NONEVENTS_FOR_SLOPE = 20
POSTHOC_CALIBRATION_MIN_EVENTS_PER_GROUP = 10
POSTHOC_CALIBRATION_MIN_NONEVENTS_PER_GROUP = 10
POSTHOC_CALIBRATION_CONSTRAIN_SLOPE_NONNEGATIVE = True
POSTHOC_CALIBRATION_FOLD_RETRY_SEEDS = 3
POSTHOC_CALIBRATION_FOLD_PARTITION_RETRIES = 3
POSTHOC_CALIBRATION_ALLOW_PARTIAL_OOF = False
POSTHOC_CALIBRATION_MIN_OOF_FRACTION = 1.00

TREATMENT_FLAG_COLS = [
    "TX_CHEMO",
    "TX_HORMONE",
    "TX_RADIATION",
    "TX_PHARMACEUTICAL",
]
TREATMENT_FLAG_LABELS = {
    "TX_CHEMO": "Chemo",
    "TX_HORMONE": "Hormone",
    "TX_RADIATION": "Radiation",
    "TX_PHARMACEUTICAL": "Pharmaceutical",
}


# Design matrices from the pathway preprocessing prefix treatment flags as
# clinbin::TX_*.  Metadata tables keep the unprefixed TX_* columns.  Keep both
# namespaces explicit so treatment-conditioned posthoc analyses do not silently
# skip treatment features.
def _base_treatment_col(col):
    s = str(col)
    return s.split("::")[-1] if "::" in s else s


def _treatment_label(col):
    base = _base_treatment_col(col)
    return TREATMENT_FLAG_LABELS.get(base, base)


def _resolve_treatment_feature_columns(feat_names):
    feat_names = [str(x) for x in feat_names]
    fset = set(feat_names)
    out = []
    for base in TREATMENT_FLAG_COLS:
        candidates = [base, f"clinbin::{base}"]
        hit = next((c for c in candidates if c in fset), None)
        if hit is not None:
            out.append((base, hit))
    return out


TREATMENT_POSTERIOR_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0]
TREATMENT_COMBO_MIN_N = 10
TREATMENT_COMBO_MIN_EVENTS = 3
TREATMENT_COMBO_MAX_PER_GROUP = 6
TREATMENT_RISK_STRATA_MIN_COUNT = 5

# Supplemental single-treatment-positive context FC. Disabled by default because
# the primary treatment-specific analysis uses exact, non-overlapping complete
# treatment-profile cells, including multi-treatment profiles.
ENABLE_TREATMENT_CONDITIONED_FC = False  # disabled: supplemental non-exclusive single-treatment-context FC; exact-profile treatment contrasts remain ON
TREATMENT_CONTEXT_FC_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0]
TREATMENT_CONTEXT_FC_MIN_N = 20
TREATMENT_CONTEXT_FC_MIN_EVENTS = 3
TREATMENT_CONTEXT_FC_TOPN = 20
TREATMENT_CONTEXT_FC_COMPARE_TO_NO_RECOGNIZED = True
TREATMENT_CONTEXT_FC_FAMILIES = ["pathway", "clinical_binary", "clinical_continuous"]


# Treatment-conditioned full-profile FC holds the complete multi-treatment profile fixed while
# ranking all non-treatment input features. Patients with multiple treatments
# are represented by the full binary treatment vector, not by mutually exclusive
# single-treatment groups.
ENABLE_TREATMENT_ADJUSTED_FC = False  # legacy implementation is definition-only; corrected artifact-array WFC is authoritative
TREATMENT_ADJUSTED_FC_H_LIST = list(FC_H_LIST)
TREATMENT_ADJUSTED_FC_TOPN = FC_TOPN
TREATMENT_ADJUSTED_FC_MIN_N = 20
TREATMENT_ADJUSTED_FC_MIN_EVENTS = 3
TREATMENT_ADJUSTED_FC_STRATIFY_BY_CANCER_GROUP = True
TREATMENT_ADJUSTED_FC_EXCLUDE_TREATMENT_FLAGS_FROM_RANKING = True
# Exact treatment-profile FC rankings are non-overlapping: each patient belongs
# to exactly one complete treatment-profile cell such as TX=None, TX=Chemo, or
# TX=Chemo+Radiation.  Multi-treatment profile rankings require larger support.
TREATMENT_EXACT_PROFILE_FC_MIN_N = TREATMENT_ADJUSTED_FC_MIN_N
TREATMENT_EXACT_PROFILE_FC_MIN_EVENTS = TREATMENT_ADJUSTED_FC_MIN_EVENTS
TREATMENT_EXACT_PROFILE_FC_MULTI_MIN_N = 40
TREATMENT_EXACT_PROFILE_FC_MULTI_MIN_EVENTS = 5

# May24: skip saved generic overall FC and FC-driven PC backprojection; keep treatment-adjusted, treatment-conditioned, and branch FC.
ENABLE_OVERALL_AND_BRANCH_FC = False  # keep non-treatment overall FC CSV/plots off
ENABLE_BRANCH_FC = False  # disabled by default: shared/cancer-specific interpretation is input-origin based
ENABLE_OVERALL_FC = False  # do not save/plot generic overall FC outputs

# Repeated-split feature-stratified treatment-comparison risk contrasts.
# For each outer split, the fitted DGP is evaluated under TX=1 and TX=0 for the same
# test patients, then treatment-risk differences are summarized within feature strata.
ENABLE_REPEATED_SPLIT_FEATURE_STRATIFIED_TX_CONTRASTS = True
FEATURE_STRATIFIED_TX_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0]
FEATURE_STRATIFIED_TX_FAMILIES = [
    "pathway",
    "clinical_continuous",
    "clinical_binary",
]
FEATURE_STRATIFIED_TX_MIN_N = 12
FEATURE_STRATIFIED_TX_MC = 128
FEATURE_STRATIFIED_TX_EPS = 1e-8
FEATURE_STRATIFIED_TX_PAIRWISE_EXACT_PROFILES = True
# Treatment-risk contrasts are required on every outer split so their mean, SD,
# posterior uncertainty, and direction stability can be reported across splits.
# The sentinel -1 is also accepted from FEATURE_STRATIFIED_TX_CONTRAST_SEEDS=all.
FEATURE_STRATIFIED_TX_CONTRAST_SEEDS = [-1]
FEATURE_STRATIFIED_TX_USE_TRAIN_SCOPE_THRESHOLDS = True
FEATURE_STRATIFIED_TX_INCLUDE_OVERALL_STRATUM = True
FEATURE_STRATIFIED_TX_CONTINUOUS_STRATA = "train_scope_tertile_tails"

# Final joined posthoc output: among the top WFC-ranked shared pathway features,
# report how exact treatment-profile changes alter predicted risk overall and
# inside low/high pathway-activity strata.
JOIN_TOP_WFC_TX_CONTRASTS = False
JOIN_TOP_WFC_TX_TOPN = 20
JOIN_TOP_WFC_TX_PLOT_TOPN = 12
JOIN_TOP_WFC_TX_SHARED_ONLY = True
JOIN_TOP_WFC_TX_FAMILIES = ["pathway"]
JOIN_TOP_WFC_TX_PROFILE_VS_NONE_ONLY_FOR_PLOTS = True

# FC interpretation guardrail: pooled rows are a patient/evaluation scope, not a
# claim that the ranked feature itself is shared across cancers. Shared-across-
# cancers feature FC is defined only by all-cancer latent feature prefixes written
# by preprocessing, e.g. act_shared::..., udp_shared::..., and future mut_shared::... columns. Cancer-specific
# latent features keep act_specific_<CANCER>::... / udp_specific_<CANCER>::....
SHARED_ACROSS_CANCERS_FEATURE_PREFIXES = ("act_shared::", "udp_shared::", "mut_shared::")
CANCER_SPECIFIC_FEATURE_PREFIXES = ("act_specific_", "udp_specific_", "mut_specific_")

# Run the primary treatment-conditioned FC on every repeated split, not only on
# the best split. Seed-level CSVs are aggregated into repeated-split FC stability
# tables. Per-seed FC plots are disabled to save wall time and file count.
ENABLE_REPEATED_SPLIT_TREATMENT_ADJUSTED_FC = False
# May26 urgent fix: save DGP-learned shared/private branch attribution for every split.
ENABLE_REPEATED_SPLIT_DGP_BRANCH_ATTRIBUTION = False
REPEATED_SPLIT_TREATMENT_ADJUSTED_FC_SAVE_PLOTS = False  # legacy WFC is disabled; corrected WFC runs from saved artifacts
ENABLE_BEST_SPLIT_POSTHOC_TREATMENT_ADJUSTED_FC = False

# May26 rescue patch: run full posthoc outputs for EVERY repeated split,
# not only the best pooled split. This includes KM/UQ/calibration plots and
# cross-fitted recalibration outputs for each seed-specific test split.
ENABLE_POSTHOC_ANALYSIS_FOR_EACH_SPLIT = True
# A seed is successful only when every required NPJ output block finishes and
# writes its source CSV. This prevents a fit from being marked complete when a
# required posthoc analysis was caught and skipped.
REQUIRE_COMPLETE_NPJ_OUTPUTS = True
# Keep the old best-split refit/posthoc optional. Set True only if you also want
# the duplicate best-split analysis after all per-split outputs are saved.
ENABLE_BEST_SPLIT_REFIT_POSTHOC_ANALYSIS = False


# ------------------------ RESISTANCE TRAJECTORY LANDSCAPE 3D ------------------------
# Posthoc trajectory landscapes use the trained DGP latent representation; no new model is fit.
# Patient-horizon states are projected into shared 2D coordinate systems, with prediction
# horizon on the z-axis. Low/medium/high risk cohort paths are horizon-ordered centroid paths.
# Late-horizon trajectory endpoints are annotated with FC-prioritized feature signatures.
ENABLE_RESISTANCE_LANDSCAPES = False
ENABLE_RESISTANCE_PSEUDOTRAJECTORY_3D = False
RESISTANCE_PSEUDOTRAJECTORY_H_LIST = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0]
RESISTANCE_PSEUDOTRAJECTORY_LATENT_SAMPLE = False  # False = posterior latent mean; True = one posterior latent draw
RESISTANCE_PSEUDOTRAJECTORY_MIN_N = 20
RESISTANCE_PSEUDOTRAJECTORY_MIN_EVENTS = 3
RESISTANCE_PSEUDOTRAJECTORY_MIN_BIN_N = 5
RESISTANCE_PSEUDOTRAJECTORY_TOP_FEATURES = 25
RESISTANCE_PSEUDOTRAJECTORY_FC_ANNOTATION_TOPN = 6
RESISTANCE_PSEUDOTRAJECTORY_USE_CALIBRATED_PRIMARY_RISK = True
RESISTANCE_PSEUDOTRAJECTORY_PRIMARY_CALIBRATION_KIND = POSTHOC_CALIBRATION_PRIMARY_KIND
RESISTANCE_PSEUDOTRAJECTORY_UMAP_N_NEIGHBORS = 15
RESISTANCE_PSEUDOTRAJECTORY_UMAP_MIN_DIST = 0.15
RESISTANCE_PSEUDOTRAJECTORY_UMAP_RANDOM_STATE = 20260519
# Risk groups are anchored away from very sparse 3/6-month horizons.
# Default: tertiles of mean predicted risk over 18 and 24 months.
RESISTANCE_PSEUDOTRAJECTORY_STRATIFY_H_LIST = [18.0, 24.0]
# Save both projection versions from the exact same patient-horizon state matrix.
RESISTANCE_PSEUDOTRAJECTORY_PROJECTION_METHODS = ["pca", "umap"]
# Late endpoint neighborhoods are used only for FC-feature annotation, not for
# defining true dynamical endpoints or transition dynamics.
RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_H = 60.0
RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_TOP_FC_FEATURES = 20
RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_NEIGHBOR_FRAC = 0.20
RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_MIN_NEIGHBORS = 10
# Exact multi-treatment trajectory panels are only drawn for larger cells.
RESISTANCE_PSEUDOTRAJECTORY_MULTI_TREATMENT_MIN_N = 40
RESISTANCE_PSEUDOTRAJECTORY_MULTI_TREATMENT_MIN_EVENTS = 5
RESISTANCE_PSEUDOTRAJECTORY_INCLUDE_FC_FEATURES_IN_PROJECTION = False

# Endpoint feature signatures and trajectory-centroid uncertainty.
# These are posthoc descriptive/inferential summaries for the trajectory figure;
# they do not change the trained DGP or fit a new trajectory model.
RESISTANCE_PSEUDOTRAJECTORY_CENTROID_BOOTSTRAP_N = 500
RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_ENRICH_BOOTSTRAP_N = 500
RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_ENRICH_PERMUTATIONS = 500
RESISTANCE_PSEUDOTRAJECTORY_BOOTSTRAP_Q = (0.05, 0.95)
RESISTANCE_PSEUDOTRAJECTORY_INFERENCE_SEED = 20260519
# Backward-compatible aliases used by older helper code/metadata.
RESISTANCE_PSEUDOTRAJECTORY_PRIMARY_RISK_H = 24.0
RESISTANCE_PSEUDOTRAJECTORY_TERMINAL_H_LIST = [RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_H]
RESISTANCE_PSEUDOTRAJECTORY_TERMINAL_REGION_TOP_FC_FEATURES = RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_TOP_FC_FEATURES


# ------------------------ DGP-FC PRIORITIZED EXPRESSION PSEUDOTIME ------------------------
# Disabled for the Clinical+Pathway run. This run has no gene-expression PC
# loadings/UDP/mutation block, so gene-level DGP-FC expression pseudotime would
# not be a valid backprojection. The resistance landscape/pseudotrajectory block
# above remains enabled and uses the trained DGP latent representation.
ENABLE_DGP_FC_GENE_EXPRESSION_PSEUDOTIME = False
DGP_FC_PSEUDOTIME_TOP_BACKPROJECTED_GENES_PER_CANCER = 350
DGP_FC_PSEUDOTIME_MIN_N = 30
DGP_FC_PSEUDOTIME_MIN_GENES = 20
DGP_FC_PSEUDOTIME_COVARIATE_MIN_N = 5
DGP_FC_PSEUDOTIME_MAP_EPOCHS = 1200
DGP_FC_PSEUDOTIME_MAP_LR = 0.03
DGP_FC_PSEUDOTIME_L2 = 1e-3
DGP_FC_PSEUDOTIME_TAU_INIT_PENALTY = 1e-3
DGP_FC_PSEUDOTIME_TREND_PERMUTATIONS = 500
DGP_FC_PSEUDOTIME_HEATMAP_TOP_GENES = 40
DGP_FC_PSEUDOTIME_RANDOM_STATE = 20260520
DGP_FC_PSEUDOTIME_RISK_ORIENTATION_H_LIST = list(FC_H_LIST)

ENABLE_DUPLICATE_BEST_SPLIT_POSTHOC_REFIT = False


# ------------------------ SLURM / REPRODUCIBILITY ENV OVERRIDES ------------------------
# These overrides are intentionally placed after the default knobs so array jobs can
# run one seed and one architecture at a time without editing the Python file.
def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(str(raw).strip())


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(str(raw).strip())


def _env_int_list(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return list(default)
    vals = []
    for tok in re.split(r"[,;\s]+", str(raw).strip()):
        if tok:
            vals.append(int(tok))
    return vals if vals else list(default)


def _env_seed_list(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return list(default)
    txt = str(raw).strip().lower()
    if txt in {"all", "*", "any"}:
        return [-1]
    vals = []
    for tok in re.split(r"[,;\s]+", str(raw).strip()):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals if vals else list(default)


def _env_float_list(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return list(default)
    vals = []
    for tok in re.split(r"[,;\s]+", str(raw).strip()):
        if tok:
            vals.append(float(tok))
    return vals if vals else list(default)


def _env_target_configs(name, default):
    """Parse TARGET_CONFIGS='0,3,3,1,1;0,3,2,1,1' or JSON-like lists.

    The fourth entry, mt_shared_rank, is preserved from TARGET_CONFIGS; these MT2 files
    use latent multitask rank 2.  The fifth entry is retained as output-rank
    metadata for backwards-compatible filenames/reporting.
    """
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        configs = list(default)
    else:
        text = str(raw).strip()
        configs = []
        try:
            parsed = json.loads(text)
            if len(parsed) == 5 and all(not isinstance(x, (list, tuple)) for x in parsed):
                parsed = [parsed]
            for cfg in parsed:
                configs.append(tuple(int(x) for x in cfg))
        except Exception:
            for chunk in text.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                vals = [int(x) for x in re.split(r"[,\s]+", chunk) if x]
                configs.append(tuple(vals))
    fixed = []
    for cfg in configs:
        if len(cfg) != 5:
            raise ValueError(
                f"Each TARGET_CONFIGS entry must have 5 integers: "
                f"surv_top_k, shared_dim, private_dim, mt_shared_rank, output_rank_metadata. Got {cfg!r}."
            )
        a, b, c, mt, e = [int(x) for x in cfg]
        fixed.append((a, b, c, mt, e))
    return fixed


SEED = _env_int("SEED", SEED)
REPEATED_SPLIT_MODE = _env_bool("REPEATED_SPLIT_MODE", REPEATED_SPLIT_MODE)
MULTI_SEEDS = _env_int_list("MULTI_SEEDS", MULTI_SEEDS)
EPOCHS = _env_int("EPOCHS", EPOCHS)
KL_WARMUP = _env_int("KL_WARMUP", int(0.30 * EPOCHS))
MC_TRAIN = _env_int("MC_TRAIN", MC_TRAIN)
FINAL_EVAL_MC_SAMPLES = _env_int("FINAL_EVAL_MC_SAMPLES", FINAL_EVAL_MC_SAMPLES)
FINAL_PRED_BAND_MC = _env_int("FINAL_PRED_BAND_MC", FINAL_PRED_BAND_MC)
RISK_UQ_MC = _env_int("RISK_UQ_MC", RISK_UQ_MC)
REPRESENTATIVE_TRAJ_MC = _env_int("REPRESENTATIVE_TRAJ_MC", REPRESENTATIVE_TRAJ_MC)
BRANCH_DECOMP_MC = _env_int("BRANCH_DECOMP_MC", BRANCH_DECOMP_MC)
FC_S = _env_int("FC_S", FC_S)
FC_REPEATS = _env_int("FC_REPEATS", FC_REPEATS)
TREATMENT_EFFECT_MC = _env_int("TREATMENT_EFFECT_MC", TREATMENT_EFFECT_MC)
EARLY_FAILURE_MC = _env_int("EARLY_FAILURE_MC", EARLY_FAILURE_MC)
EARLY_FAILURE_BOOTSTRAP_N = _env_int("EARLY_FAILURE_BOOTSTRAP_N", EARLY_FAILURE_BOOTSTRAP_N)
POSTHOC_CALIBRATION_K_FOLDS = _env_int("POSTHOC_CALIBRATION_K_FOLDS", POSTHOC_CALIBRATION_K_FOLDS)
POSTHOC_CALIBRATION_EPOCHS = _env_int("POSTHOC_CALIBRATION_EPOCHS", EPOCHS)
POSTHOC_CALIBRATION_MC_TRAIN = _env_int("POSTHOC_CALIBRATION_MC_TRAIN", MC_TRAIN)
POSTHOC_CALIBRATION_PRED_MC = _env_int("POSTHOC_CALIBRATION_PRED_MC", POSTHOC_CALIBRATION_PRED_MC)
POSTHOC_CALIBRATION_FOLD_RETRY_SEEDS = _env_int("POSTHOC_CALIBRATION_FOLD_RETRY_SEEDS", POSTHOC_CALIBRATION_FOLD_RETRY_SEEDS)
POSTHOC_CALIBRATION_FOLD_PARTITION_RETRIES = _env_int("POSTHOC_CALIBRATION_FOLD_PARTITION_RETRIES", POSTHOC_CALIBRATION_FOLD_PARTITION_RETRIES)
POSTHOC_CALIBRATION_ALLOW_PARTIAL_OOF = _env_bool("POSTHOC_CALIBRATION_ALLOW_PARTIAL_OOF", POSTHOC_CALIBRATION_ALLOW_PARTIAL_OOF)
POSTHOC_CALIBRATION_MIN_OOF_FRACTION = _env_float("POSTHOC_CALIBRATION_MIN_OOF_FRACTION", POSTHOC_CALIBRATION_MIN_OOF_FRACTION)
FEATURE_STRATIFIED_TX_MC = _env_int("FEATURE_STRATIFIED_TX_MC", FEATURE_STRATIFIED_TX_MC)
FEATURE_STRATIFIED_TX_CONTRAST_SEEDS = _env_seed_list(
    "FEATURE_STRATIFIED_TX_CONTRAST_SEEDS", FEATURE_STRATIFIED_TX_CONTRAST_SEEDS
)
JOIN_TOP_WFC_TX_TOPN = _env_int("JOIN_TOP_WFC_TX_TOPN", JOIN_TOP_WFC_TX_TOPN)
JOIN_TOP_WFC_TX_PLOT_TOPN = _env_int("JOIN_TOP_WFC_TX_PLOT_TOPN", JOIN_TOP_WFC_TX_PLOT_TOPN)

POSTERIOR_H_LIST = _env_float_list("POSTERIOR_H_LIST", POSTERIOR_H_LIST)
RISK_UQ_H_LIST = _env_float_list("RISK_UQ_H_LIST", RISK_UQ_H_LIST)
BRANCH_DECOMP_H_LIST = _env_float_list("BRANCH_DECOMP_H_LIST", BRANCH_DECOMP_H_LIST)
FC_H_LIST = _env_float_list("FC_H_LIST", FC_H_LIST)
EARLY_FAILURE_H_LIST = _env_float_list("EARLY_FAILURE_H_LIST", EARLY_FAILURE_H_LIST)
POSTHOC_CALIBRATION_H_LIST = _env_float_list("POSTHOC_CALIBRATION_H_LIST", POSTHOC_CALIBRATION_H_LIST)
TREATMENT_ADJUSTED_FC_H_LIST = _env_float_list("TREATMENT_ADJUSTED_FC_H_LIST", TREATMENT_ADJUSTED_FC_H_LIST)
FEATURE_STRATIFIED_TX_H_LIST = _env_float_list("FEATURE_STRATIFIED_TX_H_LIST", FEATURE_STRATIFIED_TX_H_LIST)

ENABLE_POSTHOC_RECALIBRATION = _env_bool("ENABLE_POSTHOC_RECALIBRATION", ENABLE_POSTHOC_RECALIBRATION)
ENABLE_POSTHOC_ANALYSIS_FOR_EACH_SPLIT = _env_bool("ENABLE_POSTHOC_ANALYSIS_FOR_EACH_SPLIT", ENABLE_POSTHOC_ANALYSIS_FOR_EACH_SPLIT)
REQUIRE_COMPLETE_NPJ_OUTPUTS = _env_bool("REQUIRE_COMPLETE_NPJ_OUTPUTS", REQUIRE_COMPLETE_NPJ_OUTPUTS)
# Lock every legacy/generic feature-collapse execution off. The corrected
# treatment-conditioned artifact-array WFC is the sole authoritative WFC.
ENABLE_REPEATED_SPLIT_TREATMENT_ADJUSTED_FC = False
REPEATED_SPLIT_TREATMENT_ADJUSTED_FC_SAVE_PLOTS = _env_bool("REPEATED_SPLIT_TREATMENT_ADJUSTED_FC_SAVE_PLOTS", REPEATED_SPLIT_TREATMENT_ADJUSTED_FC_SAVE_PLOTS)
ENABLE_REPEATED_SPLIT_DGP_BRANCH_ATTRIBUTION = _env_bool("ENABLE_REPEATED_SPLIT_DGP_BRANCH_ATTRIBUTION", ENABLE_REPEATED_SPLIT_DGP_BRANCH_ATTRIBUTION)
ENABLE_REPEATED_SPLIT_FEATURE_STRATIFIED_TX_CONTRASTS = _env_bool("ENABLE_REPEATED_SPLIT_FEATURE_STRATIFIED_TX_CONTRASTS", ENABLE_REPEATED_SPLIT_FEATURE_STRATIFIED_TX_CONTRASTS)
FEATURE_STRATIFIED_TX_USE_TRAIN_SCOPE_THRESHOLDS = _env_bool(
    "FEATURE_STRATIFIED_TX_USE_TRAIN_SCOPE_THRESHOLDS",
    FEATURE_STRATIFIED_TX_USE_TRAIN_SCOPE_THRESHOLDS,
)
FEATURE_STRATIFIED_TX_INCLUDE_OVERALL_STRATUM = _env_bool(
    "FEATURE_STRATIFIED_TX_INCLUDE_OVERALL_STRATUM",
    FEATURE_STRATIFIED_TX_INCLUDE_OVERALL_STRATUM,
)
JOIN_TOP_WFC_TX_CONTRASTS = _env_bool("JOIN_TOP_WFC_TX_CONTRASTS", JOIN_TOP_WFC_TX_CONTRASTS)
JOIN_TOP_WFC_TX_SHARED_ONLY = _env_bool("JOIN_TOP_WFC_TX_SHARED_ONLY", JOIN_TOP_WFC_TX_SHARED_ONLY)
JOIN_TOP_WFC_TX_PROFILE_VS_NONE_ONLY_FOR_PLOTS = _env_bool(
    "JOIN_TOP_WFC_TX_PROFILE_VS_NONE_ONLY_FOR_PLOTS",
    JOIN_TOP_WFC_TX_PROFILE_VS_NONE_ONLY_FOR_PLOTS,
)
EARLY_FAILURE_SAVE_BOOTSTRAP_PLOTS = _env_bool("EARLY_FAILURE_SAVE_BOOTSTRAP_PLOTS", EARLY_FAILURE_SAVE_BOOTSTRAP_PLOTS)
EARLY_FAILURE_SAVE_POOLED_TOP_PATIENT_PLOTS = _env_bool("EARLY_FAILURE_SAVE_POOLED_TOP_PATIENT_PLOTS", EARLY_FAILURE_SAVE_POOLED_TOP_PATIENT_PLOTS)
EARLY_FAILURE_SAVE_WITHIN_CANCER_TOP_PATIENT_PLOTS = _env_bool("EARLY_FAILURE_SAVE_WITHIN_CANCER_TOP_PATIENT_PLOTS", EARLY_FAILURE_SAVE_WITHIN_CANCER_TOP_PATIENT_PLOTS)
SAVE_WITHIN_CANCER_AUROC_AUPRC_CURVES = _env_bool("SAVE_WITHIN_CANCER_AUROC_AUPRC_CURVES", SAVE_WITHIN_CANCER_AUROC_AUPRC_CURVES)
ENABLE_BRANCH_FC = False
ENABLE_OVERALL_FC = False
ENABLE_HEATMAPS = _env_bool("ENABLE_HEATMAPS", ENABLE_HEATMAPS)

# Fast-but-still-publishable mode for repeated-seed jobs.  It keeps the KM,
# calibration/metrics CSVs, test-set bootstrap CSVs, treatment-adjusted WFC CSVs,
# branch-attribution CSVs, and treatment-contrast CSVs, while skipping expensive
# per-seed decorative plots. Bootstrap resampling is not clamped here because it
# resamples held-out predictions rather than refitting the DGP.
if _env_bool("ESSENTIAL_POSTHOC_ONLY", False):
    EARLY_FAILURE_SAVE_POOLED_TOP_PATIENT_PLOTS = False
    EARLY_FAILURE_SAVE_WITHIN_CANCER_TOP_PATIENT_PLOTS = False
    EARLY_FAILURE_SAVE_BOOTSTRAP_PLOTS = False
    REPEATED_SPLIT_TREATMENT_ADJUSTED_FC_SAVE_PLOTS = False
    SAVE_WITHIN_CANCER_AUROC_AUPRC_CURVES = False
    FINAL_PRED_BAND_MC = min(int(FINAL_PRED_BAND_MC), 128)
    RISK_UQ_MC = min(int(RISK_UQ_MC), 128)
    TREATMENT_EFFECT_MC = min(int(TREATMENT_EFFECT_MC), 128)
    EARLY_FAILURE_MC = min(int(EARLY_FAILURE_MC), 128)
    POSTHOC_CALIBRATION_PRED_MC = min(int(POSTHOC_CALIBRATION_PRED_MC), 128)

# Absolute guard: latent multitask rank is fixed at 2 in these MT2 files.
SWEEP_MT_LATENT_RANK_LIST = [2]

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def harrell_c(times, scores, events, auto_orient=False):
    t = np.asarray(times, float)
    s = np.asarray(scores, float)
    e = np.asarray(events, int)
    c = float(_lifelines_c(t, s, e))
    if auto_orient and c < 0.5:
        c_flip = float(_lifelines_c(t, -s, e))
        if c_flip > c:
            return c_flip, True
    return c, False


def train_orientation_sign(times, events, scores):
    c_raw = float(
        _lifelines_c(
            np.asarray(times, float), np.asarray(scores, float), np.asarray(events, int)
        )
    )
    return -1.0 if c_raw < 0.5 else 1.0


def apply_orientation(scores, sign):
    return np.asarray(scores, float) * float(sign)


def infer_blocks_from_feat_names(feat_names: list[str]):
    blocks = {
        k: []
        for k in [
            "gene",
            "pathway",
            "udp",
            "clin_cont",
            "clin_bin",
            "tmb",
            "mut",
            "ges",
            "act",
            "cons",
            "cna",
        ]
    }
    for j, nm in enumerate(feat_names):
        n = str(nm)

        # New preprocessing prefixes. These preserve separate kernels by feature type.
        if n.startswith(("act_shared::", "act_specific_")):
            blocks["pathway"].append(j)
            continue
        if n.startswith(("udp_shared::", "udp_specific_")):
            blocks["udp"].append(j)
            continue
        if n.startswith("clincont::"):
            blocks["clin_cont"].append(j)
            continue
        if n.startswith("clinbin::"):
            blocks["clin_bin"].append(j)
            continue
        if n.startswith("tmb::"):
            blocks["tmb"].append(j)
            continue
        if n.startswith(("mut::", "mut_shared::", "mut_specific_")):
            blocks["mut"].append(j)
            continue

        # Backward-compatible old prefixes.
        if re.match(r"^(BRCA|LUAD|COAD)_PC\d+$", n) or re.match(r"^PC\d+$", n):
            blocks["gene"].append(j)
            continue
        if n in ("AGE", "AGE_Z", "AGE_AT_DIAGNOSIS"):
            blocks["clin_cont"].append(j)
            continue
        if n.startswith(("TX_", "STAGE_", "GRADE_", "AJCC_T_")) or n == "RADIATION_ANY":
            blocks["clin_bin"].append(j)
            continue
        if n.startswith("cna_"):
            blocks["cna"].append(j)
            continue
        if n.startswith("ges_"):
            blocks["ges"].append(j)
            continue
        if n.startswith(("actPC", "pathActPC", "PATHACTPC", "ACTPC")):
            blocks["act"].append(j)
            continue
        if n.startswith(("consPC", "pathConsPC", "PATHCONSPC", "CONSPC")):
            blocks["cons"].append(j)
            continue
        blocks["clin_cont"].append(j)
    return blocks


def _np_quantile_bins(t, n_bins):
    t = np.asarray(t, float)
    t = t[np.isfinite(t) & (t >= 0)]
    edges = np.quantile(t, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        mx = float(np.max(t)) if t.size else 1.0
        _warn_explicit_fallback(
            "Quantile-based discrete-time bin construction produced fewer than two unique edges; "
            "falling back to evenly spaced bins."
        )
        edges = np.linspace(0.0, max(mx, 1.0), n_bins + 1)
    if edges[0] > 0:
        edges[0] = 0.0
    return edges.astype(float)


def make_discrete_time_edges(t, n_bins=DT_BINS):
    """Construct discrete-time survival edges.

    The early-failure version uses fixed early-resolution bins so that 3-, 6-,
    12-, and 18-month failure probabilities are directly estimable.  Pure
    quantile bins can place the first right edge well after 3 or 6 months,
    making early-failure risks artificially flat.
    """
    t = np.asarray(t, float)
    t = t[np.isfinite(t) & (t >= 0.0)]
    if t.size == 0:
        return np.asarray(
            DT_EDGES_MANUAL if USE_MANUAL_DT_EDGES else [0.0, 1.0], dtype=float
        )
    if not USE_MANUAL_DT_EDGES:
        return _np_quantile_bins(t, n_bins)
    mx = float(np.max(t))
    base = np.asarray(DT_EDGES_MANUAL, dtype=float)
    base = np.unique(base[np.isfinite(base) & (base >= 0.0)])
    if base.size < 2:
        return _np_quantile_bins(t, n_bins)
    edges = base[base < mx]
    if edges.size == 0 or edges[0] > 0.0:
        edges = np.concatenate([[0.0], edges])
    if not np.any(np.isclose(edges, mx)):
        edges = np.concatenate([edges, [mx]])
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.asarray([0.0, max(mx, 1.0)], dtype=float)
    return edges.astype(float)


def make_dt_bin_weights(dt_edges):
    """Return normalized bin weights that emphasize early failure windows."""
    right_edges = np.asarray(dt_edges[1:], dtype=float)
    if not USE_EARLY_BIN_WEIGHTS:
        return np.ones(right_edges.shape[0], dtype=float)
    w = 1.0 + float(EARLY_BIN_WEIGHT_GAMMA) * np.exp(
        -right_edges / max(float(EARLY_BIN_WEIGHT_TAU), 1e-8)
    )
    w = w / max(float(np.mean(w)), 1e-8)
    return w.astype(float)


def discretize_times(times, edges):
    times = np.asarray(times, float)
    K = len(edges) - 1
    idx = np.searchsorted(edges[1:], times, side="left")
    idx = np.clip(idx, 0, K - 1)
    return idx.astype(int)


def discrete_time_nll_from_score(score, event_bin, event_ind, alpha_dt):
    logits = alpha_dt.unsqueeze(0) + score.unsqueeze(1)
    haz = torch.sigmoid(logits)
    log1m = torch.log1p(-haz.clamp(min=1e-8, max=1 - 1e-8))
    logh = torch.log(haz.clamp(min=1e-8, max=1 - 1e-8))

    idx_grid = torch.arange(alpha_dt.shape[0], device=score.device).view(1, -1)
    eb = event_bin.view(-1, 1)
    survive_before = (idx_grid < eb).float()

    ll_pre = (survive_before * log1m).sum(dim=1)
    evt_ll = ll_pre + logh.gather(1, eb).squeeze(1)
    cens_ll = ll_pre
    total_ll = torch.where(event_ind > 0, evt_ll, cens_ll)
    return -total_ll.mean()


def risk_from_score_at_H(score_np, edges, alpha_np, H):
    score_np = np.asarray(score_np, float)
    alpha_np = np.asarray(alpha_np, float)
    logits = score_np[:, None] + alpha_np[None, :]
    haz = 1.0 / (1.0 + np.exp(-logits))
    right_edges = np.asarray(edges[1:], float)
    alive_bins = np.where(right_edges <= float(H))[0]
    if alive_bins.size == 0:
        return np.zeros(score_np.shape[0], dtype=float)
    S = np.prod(1.0 - haz[:, alive_bins], axis=1)
    S = np.clip(S, 1e-12, 1.0)
    return -np.log(S)


def make_vs(
    model,
    inducing_points,
    variational_distribution,
    learn_inducing_locations=True,
    prefer_whiten=True,
):
    sig = inspect.signature(VariationalStrategy.__init__)
    if "whiten" in sig.parameters:
        return VariationalStrategy(
            model,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
            whiten=prefer_whiten,
        )
    return VariationalStrategy(
        model,
        inducing_points,
        variational_distribution,
        learn_inducing_locations=learn_inducing_locations,
    )


def _select_task(vals, g_idx, num_tasks):
    if vals.dim() == 2:
        return vals[torch.arange(vals.size(0), device=vals.device), g_idx]
    if vals.dim() == 3:
        idx = g_idx.view(1, -1, 1).expand(vals.size(0), -1, 1)
        return torch.gather(vals, dim=2, index=idx).squeeze(2)
    raise ValueError(f"Unexpected vals shape for task selection: {tuple(vals.shape)}")


def _mk_matern52_block(active_dims, batch_shape):
    if active_dims is None or len(active_dims) == 0:
        return None
    active_dims = list(map(int, active_dims))
    base = MaternKernel(
        nu=2.5,
        ard_num_dims=len(active_dims),
        batch_shape=batch_shape,
        active_dims=active_dims,
    )
    return ScaleKernel(base, batch_shape=batch_shape)


def _mk_linear_block(active_dims, batch_shape):
    if active_dims is None or len(active_dims) == 0:
        return None
    active_dims = list(map(int, active_dims))
    return LinearKernel(batch_shape=batch_shape, active_dims=active_dims)


def _mk_input_kernel(in_dim, batch_shape, blocks=None):
    if blocks and any(len(v) > 0 for v in blocks.values()):
        pieces = []

        # Smooth/continuous feature families.
        for key in ("gene", "pathway", "udp", "clin_cont", "tmb", "act", "cons", "ges"):
            kblk = _mk_matern52_block(blocks.get(key, []), batch_shape=batch_shape)
            if kblk is not None:
                pieces.append(kblk)

        # Binary/sparse feature families.
        for key in ("clin_bin", "mut", "cna"):
            kblk = _mk_linear_block(blocks.get(key, []), batch_shape=batch_shape)
            if kblk is not None:
                pieces.append(kblk)

        if len(pieces) == 0:
            return ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=in_dim, batch_shape=batch_shape),
                batch_shape=batch_shape,
            )
        kern = pieces[0]
        for p in pieces[1:]:
            kern = kern + p
        return kern
    return ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=in_dim, batch_shape=batch_shape),
        batch_shape=batch_shape,
    )


def _init_matern52_scale_kernel_(kern, lengthscale0=1.5, outputscale0=1.5):
    if kern is None:
        return
    if isinstance(kern, gpytorch.kernels.AdditiveKernel):
        for sub in kern.kernels:
            _init_matern52_scale_kernel_(
                sub, lengthscale0=lengthscale0, outputscale0=outputscale0
            )
        return
    try:
        if hasattr(kern, "outputscale"):
            kern.outputscale = outputscale0
    except Exception:
        pass
    base = getattr(kern, "base_kernel", None)
    if base is not None and hasattr(base, "lengthscale"):
        try:
            base.lengthscale = lengthscale0
        except Exception:
            pass


class MultitaskSharedLatentDimICM(ApproximateGP):
    def __init__(self, in_dim, Xref, seed, num_groups, rank, blocks=None):
        self.num_groups = int(num_groups)
        self.rank = int(max(1, rank))
        batch_shape = torch.Size([self.rank])
        M = int(min(NUM_INDUCING, Xref.size(0)))
        if M >= 2:
            with torch.no_grad():
                km = KMeans(n_clusters=M, random_state=seed, n_init=10)
                centers = km.fit(Xref.detach().cpu().numpy()).cluster_centers_
                inducing_base = torch.from_numpy(centers).to(
                    dtype=Xref.dtype, device=Xref.device
                )
            inducing = inducing_base.unsqueeze(0).expand(self.rank, -1, -1).contiguous()
        else:
            inducing = (
                torch.randn(self.rank, M, in_dim, device=Xref.device, dtype=Xref.dtype)
                * 0.05
            )
        q = (
            MeanFieldVariationalDistribution(M, batch_shape=batch_shape)
            if VI_DIST.lower() == "meanfield"
            else CholeskyVariationalDistribution(M, batch_shape=batch_shape)
        )
        base_vs = make_vs(
            self, inducing, q, learn_inducing_locations=True, prefer_whiten=True
        )
        lmc_vs = LMCVariationalStrategy(
            base_vs, num_tasks=self.num_groups, num_latents=self.rank, latent_dim=-1
        )
        super().__init__(lmc_vs)
        self.mean_module = ZeroMean(batch_shape=batch_shape)
        self.covar_module = _mk_input_kernel(
            in_dim, batch_shape=batch_shape, blocks=blocks
        )
        _init_matern52_scale_kernel_(
            self.covar_module, lengthscale0=1.5, outputscale0=1.5
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class MultitaskSharedLatentLayer(torch.nn.Module):
    def __init__(self, in_dim, Xref, seed, num_groups, rank, latent_dim, blocks=None):
        super().__init__()
        self.num_groups = int(num_groups)
        self.latent_dim = int(latent_dim)
        self.rank = int(rank)
        self.dims = torch.nn.ModuleList()
        for k in range(self.latent_dim):
            self.dims.append(
                MultitaskSharedLatentDimICM(
                    in_dim=in_dim,
                    Xref=Xref,
                    seed=seed + 7777 + 100 * k,
                    num_groups=self.num_groups,
                    rank=self.rank,
                    blocks=blocks,
                )
            )

    def forward(self, x, g_idx, sample=True):
        g_idx = g_idx.to(device=x.device, dtype=torch.long).view(-1)
        hs = []
        for gp_dim in self.dims:
            f_all = gp_dim(x)
            vals = f_all.rsample() if sample else f_all.mean
            hk = _select_task(vals, g_idx, num_tasks=self.num_groups)
            hs.append(hk)
        return torch.stack(hs, dim=1).contiguous()


class PrivateLayer(ApproximateGP):
    def __init__(self, in_dim, private_dim, Xref, seed, blocks=None):
        batch_shape = torch.Size([private_dim])
        M = int(min(NUM_INDUCING, Xref.size(0)))
        if M >= 2:
            with torch.no_grad():
                km = KMeans(n_clusters=M, random_state=seed, n_init=10)
                centers = km.fit(Xref.detach().cpu().numpy()).cluster_centers_
                inducing_base = torch.from_numpy(centers).to(
                    dtype=Xref.dtype, device=Xref.device
                )
        else:
            idx = torch.randperm(Xref.size(0), device=Xref.device)[:M]
            inducing_base = Xref[idx]
        inducing = inducing_base.unsqueeze(0).expand(private_dim, -1, -1).contiguous()
        q = (
            MeanFieldVariationalDistribution(M, batch_shape=batch_shape)
            if VI_DIST.lower() == "meanfield"
            else CholeskyVariationalDistribution(M, batch_shape=batch_shape)
        )
        strat = make_vs(
            self, inducing, q, learn_inducing_locations=True, prefer_whiten=True
        )
        super().__init__(strat)
        self.mean_module = ZeroMean(batch_shape=batch_shape)
        self.covar_module = _mk_input_kernel(
            in_dim, batch_shape=batch_shape, blocks=blocks
        )
        _init_matern52_scale_kernel_(
            self.covar_module, lengthscale0=1.5, outputscale0=1.5
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class OutputLayerGroup(ApproximateGP):
    def __init__(self, in_dim, Xref, seed):
        M = int(min(NUM_INDUCING, Xref.size(0)))
        inducing = torch.randn(M, in_dim, device=Xref.device, dtype=Xref.dtype) * 0.05
        q = (
            MeanFieldVariationalDistribution(M)
            if VI_DIST.lower() == "meanfield"
            else CholeskyVariationalDistribution(M)
        )
        strat = make_vs(
            self, inducing, q, learn_inducing_locations=True, prefer_whiten=True
        )
        super().__init__(strat)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=in_dim))
        _init_matern52_scale_kernel_(
            self.covar_module, lengthscale0=1.5, outputscale0=2.0
        )

    def forward(self, h):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(h), self.covar_module(h)
        )


class DGP(gpytorch.models.deep_gps.DeepGP):
    def __init__(
        self,
        in_dim,
        Xref,
        num_groups,
        shared_dim,
        private_dim,
        mt_shared_rank,
        output_rank,
        blocks=None,
        seed=SEED,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.mt_shared_rank = int(mt_shared_rank)
        self.shared_dim = int(shared_dim)
        self.private_dim = int(private_dim)
        self.output_rank = int(
            output_rank
        )  # metadata only in Workflow C; output layer is group-specific, not multitask  # metadata only in Workflow C; output layer is group-specific, not multitask
        self.shared_latent_mt = MultitaskSharedLatentLayer(
            in_dim=in_dim,
            Xref=Xref,
            seed=seed,
            num_groups=self.num_groups,
            rank=self.mt_shared_rank,
            latent_dim=self.shared_dim,
            blocks=blocks,
        )
        self.private = torch.nn.ModuleList(
            [
                PrivateLayer(
                    in_dim=in_dim,
                    private_dim=self.private_dim,
                    Xref=Xref,
                    seed=seed + 123 * (g + 1),
                    blocks=blocks,
                )
                for g in range(self.num_groups)
            ]
        )
        self.output = torch.nn.ModuleList(
            [
                OutputLayerGroup(
                    in_dim=self.shared_dim + self.private_dim,
                    Xref=Xref,
                    seed=seed + 999 + 1009 * g,
                )
                for g in range(self.num_groups)
            ]
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        with torch.no_grad():
            self.likelihood.noise = 1e-3

    def forward_latents(self, x, g_idx, sample=True):
        g_idx = g_idx.to(device=x.device, dtype=torch.long).view(-1)
        h_shared = self.shared_latent_mt(x, g_idx, sample=sample)
        h_private = torch.zeros(
            x.size(0), self.private_dim, device=x.device, dtype=x.dtype
        )
        for g in range(self.num_groups):
            idx = (g_idx == g).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            xg = x.index_select(0, idx)
            dist = self.private[g](xg)
            vals = dist.rsample() if sample else dist.mean
            pg = vals.transpose(-1, -2).contiguous()
            h_private.index_copy_(0, idx, pg)
        return torch.cat([h_shared, h_private], dim=-1).contiguous()

    def forward(self, x, g_idx):
        return self.forward_score(x, g_idx, sample=True)[0]

    def forward_score(self, x, g_idx, sample=True):
        g_idx = g_idx.to(device=x.device, dtype=torch.long).view(-1)
        h = self.forward_latents(x, g_idx, sample=sample)
        s = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for g in range(self.num_groups):
            idx = (g_idx == g).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            hg = h.index_select(0, idx)
            dist = self.output[g](hg)
            vals = dist.rsample() if sample else dist.mean
            vals = vals if vals.dim() == 1 else vals.squeeze(-1)
            s.index_copy_(0, idx, vals)
        return s, h


def total_kl_details(model):
    kl_s = torch.tensor(0.0, device=device)
    if getattr(model, "shared_latent_mt", None) is not None:
        for gp_dim in model.shared_latent_mt.dims:
            kl_s = kl_s + gp_dim.variational_strategy.kl_divergence().sum()

    kl_p = torch.tensor(0.0, device=device)
    if hasattr(model, "private") and model.private is not None:
        for pgp in model.private:
            kl_p = kl_p + pgp.variational_strategy.kl_divergence().sum()

    kl_o = torch.tensor(0.0, device=device)
    if hasattr(model, "output") and model.output is not None:
        if isinstance(model.output, torch.nn.ModuleList):
            for ogp in model.output:
                kl_o = kl_o + ogp.variational_strategy.kl_divergence().sum()
        else:
            kl_o = model.output.variational_strategy.kl_divergence().sum()
    return kl_s, kl_p, kl_o


def verify_architecture(model):
    info = {}
    has_mt_latent = getattr(model, "shared_latent_mt", None) is not None
    output_group_specific = isinstance(
        getattr(model, "output", None), torch.nn.ModuleList
    )
    info["has_mt_latent"] = bool(has_mt_latent)
    info["output_group_specific"] = bool(output_group_specific)
    info["mt_shared_rank"] = int(getattr(model, "mt_shared_rank", -1))
    info["output_rank"] = int(getattr(model, "output_rank", -1))
    info["mt_latent_num_dims"] = int(len(model.shared_latent_mt.dims))
    latent_ok = all(
        isinstance(gp_dim.variational_strategy, LMCVariationalStrategy)
        for gp_dim in model.shared_latent_mt.dims
    )
    output_ok = output_group_specific and len(model.output) == model.num_groups
    info["latent_ok"] = bool(latent_ok)
    info["output_ok"] = bool(output_ok)
    info["workflow_c_ok"] = bool(latent_ok and output_ok)
    if not info["workflow_c_ok"]:
        raise RuntimeError(f"Workflow C verification failed: {info}")
    return info


def build_optimizer(model, alpha_dt):
    var_params, ind_params = [], []
    VS = VariationalStrategy
    MFVD = MeanFieldVariationalDistribution
    CVD = CholeskyVariationalDistribution

    for mod in model.modules():
        if isinstance(mod, (CVD, MFVD)):
            var_params += list(mod.parameters())
        vs = getattr(mod, "variational_strategy", None)
        if vs is None:
            continue
        if isinstance(vs, VS) and hasattr(vs, "inducing_points"):
            ind_params.append(vs.inducing_points)

    def _unique_params(params):
        seen = set()
        out = []
        for p in params:
            if id(p) not in seen:
                out.append(p)
                seen.add(id(p))
        return out

    var_params = _unique_params(var_params)
    ind_params = _unique_params(ind_params)
    var_ids = {id(p) for p in var_params}
    ind_ids = {id(p) for p in ind_params}
    hyp_params = [p for p in model.parameters() if id(p) not in (var_ids | ind_ids)]

    optim_groups = []
    if len(var_params):
        optim_groups.append({"params": var_params, "lr": LR_VAR})
    if len(ind_params):
        optim_groups.append({"params": ind_params, "lr": LR_IND})
    if len(hyp_params):
        optim_groups.append({"params": hyp_params, "lr": LR_HYP})
    if alpha_dt is not None:
        optim_groups.append({"params": [alpha_dt], "lr": 0.01})
    return torch.optim.Adam(optim_groups)


def _prefix_group_specific_pc_columns(df, cancer_code):
    rename = {}
    for c in df.columns:
        s = str(c)
        if re.match(r"^PC\d+$", s):
            rename[c] = f"{cancer_code}_{s}"
    return df.rename(columns=rename)


def _concat_align_frames(frames: list[pd.DataFrame]):
    cols = []
    seen = set()
    for df in frames:
        for c in df.columns:
            if c not in seen:
                cols.append(c)
                seen.add(c)
    return pd.concat(
        [df.reindex(columns=cols, fill_value=0.0) for df in frames], axis=0
    )


def _first_existing_path(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing required input file. Tried:\n" + "\n".join(map(str, candidates))
    )


def _dedup_paths(paths):
    out = []
    seen = set()
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def _candidate_prep_roots():
    """Candidate preprocessing roots.

    This DGP file writes new POSTHOCFIRST results, but it should be able to
    read preprocessing outputs produced by any May24 PathClin local version.
    In particular, the original local preprocessing script wrote to
    ``...LOCAL_UNIQUE...`` while later DGP-only variants wrote results to
    ``...LOCAL_TXFC_BRANCH_EARLY...``.  Do not force a preprocessing rerun just
    because the DGP output label changed.
    """
    # August runs never fall back to July/10RS preprocessing roots.
    return [PREP_OUTDIR]

    roots = []

    def add(x):
        if x is not None:
            roots.append(Path(x))

    add(PREP_OUTDIR)
    add(os.getenv("PREP_OUTDIR"))

    # Known preprocessing folders that may already exist from the local runs.
    known_names = [
        "prep_PathClin_10RS",
        "tcga_preprocess_outputs_resistance_workflow_C_May24_PathClin_RS3_LOCAL_TXFC_EARLY_UNIQUE_PER_SPLIT_BRANCHATTR_PAIRWISE",
        "tcga_preprocess_outputs_resistance_workflow_C_May24_PathClin_RS3_LOCAL_TXFC_BRANCH_EARLY_UNIQUE_PER_SPLIT_BRANCHATTR_PAIRWISE",
        "tcga_preprocess_outputs_resistance_workflow_C_May24_PathClin_RS3_LOCAL_TXFC_BRANCH_EARLY_POSTHOCFIRST_FINAL_PREPSEARCHFIX_RECALFIX_ALPHAFIX_H3BOOT_EXACTTX_UNIQUE_PER_SPLIT_BRANCHATTR_PAIRWISE",
    ]
    for name in known_names:
        add(TCGA_ROOT / name)

    # If a user renamed a preprocessing folder but kept the May24 PathClin prefix,
    # discover it automatically under TCGA_ROOT.
    try:
        for hit in sorted(TCGA_ROOT.glob("tcga_preprocess_outputs_resistance_workflow_C_May24_PathClin_RS3*")):
            add(hit)
    except Exception:
        pass

    # Handle accidental double-PAIRWISE names from earlier experiments.
    for p in list(roots):
        s = str(p)
        if s.endswith("_PAIRWISE_PAIRWISE"):
            add(s[: -len("_PAIRWISE")])
        elif s.endswith("_PAIRWISE"):
            add(s + "_PAIRWISE")

    # Rescue only: one older sbatch accidentally wrote preprocessing outputs
    # under the DGP results OUTDIR.  Keep this as a fallback, but it is not the
    # intended location.
    add(OUTDIR)
    add(os.getenv("OUTDIR"))

    return _dedup_paths(roots)

def _seed_prep_dir(seed):
    suffixes = [f"split_seed{int(seed)}_{CONFIG_NAME}"]
    # Historical suffixes below are intentionally unreachable in NPJ August.
    _historical_suffixes = [
        f"split_seed{int(seed)}_PathClin",
        f"split_seed{int(seed)}_Pathway",
        f"split_seed{int(seed)}_ClinicalPathway",
    ]
    tried = []
    for root in _candidate_prep_roots():
        for suffix in suffixes:
            candidate = root / suffix
            tried.append(candidate)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "Missing seed-specific preprocessing outputs for "
        f"seed {int(seed)}. Run preprocessing with OUTDIR set to PREP_OUTDIR. Tried directories:\n"
        + "\n".join(map(str, tried))
    )


def _load_one_cancer_split(prep_dir: Path, cancer_code: str):
    code = str(cancer_code).lower()
    p_xtr = _first_existing_path(
        [
            prep_dir / f"tcga_{code}_design_train_PathClin.csv",
            prep_dir / f"tcga_{code}_design_train_Pathway.csv",
            prep_dir / f"tcga_{code}_design_train_ClinicalPathway.csv",
            prep_dir / f"tcga_{code}_design_train.csv",
        ]
    )
    p_xte = _first_existing_path(
        [
            prep_dir / f"tcga_{code}_design_test_PathClin.csv",
            prep_dir / f"tcga_{code}_design_test_Pathway.csv",
            prep_dir / f"tcga_{code}_design_test_ClinicalPathway.csv",
            prep_dir / f"tcga_{code}_design_test.csv",
        ]
    )
    p_meta_tr = _first_existing_path(
        [
            prep_dir / f"tcga_{code}_metadata_train_PathClin.csv",
            prep_dir / f"tcga_{code}_metadata_train_Pathway.csv",
            prep_dir / f"tcga_{code}_metadata_train_ClinicalPathway.csv",
            prep_dir / f"tcga_{code}_metadata_train.csv",
        ]
    )
    p_meta_te = _first_existing_path(
        [
            prep_dir / f"tcga_{code}_metadata_test_PathClin.csv",
            prep_dir / f"tcga_{code}_metadata_test_Pathway.csv",
            prep_dir / f"tcga_{code}_metadata_test_ClinicalPathway.csv",
            prep_dir / f"tcga_{code}_metadata_test.csv",
        ]
    )
    Xtr = pd.read_csv(p_xtr, index_col=0)
    Xte = pd.read_csv(p_xte, index_col=0)
    meta_tr = pd.read_csv(p_meta_tr)
    meta_te = pd.read_csv(p_meta_te)
    meta_tr["SAMPLE_KEY"] = meta_tr["SAMPLE_KEY"].astype(str)
    meta_te["SAMPLE_KEY"] = meta_te["SAMPLE_KEY"].astype(str)
    Xtr = Xtr.loc[meta_tr["SAMPLE_KEY"].tolist()].copy()
    Xte = Xte.loc[meta_te["SAMPLE_KEY"].tolist()].copy()
    Xtr = _prefix_group_specific_pc_columns(Xtr, cancer_code=str(cancer_code))
    Xte = _prefix_group_specific_pc_columns(Xte, cancer_code=str(cancer_code))
    return Xtr, Xte, meta_tr, meta_te


def _load_exact_august_cancer_split(prep_dir: Path, cancer_code: str):
    code = str(cancer_code).lower()
    suffix = CONFIG_NAME
    paths = {
        "xtr": prep_dir / f"tcga_{code}_design_train_{suffix}.csv",
        "xte": prep_dir / f"tcga_{code}_design_test_{suffix}.csv",
        "mtr": prep_dir / f"tcga_{code}_metadata_train_{suffix}.csv",
        "mte": prep_dir / f"tcga_{code}_metadata_test_{suffix}.csv",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing exact NPJ August preprocessing inputs:\n" + "\n".join(missing))
    Xtr, Xte = pd.read_csv(paths["xtr"], index_col=0), pd.read_csv(paths["xte"], index_col=0)
    meta_tr, meta_te = pd.read_csv(paths["mtr"]), pd.read_csv(paths["mte"])
    meta_tr["SAMPLE_KEY"], meta_te["SAMPLE_KEY"] = meta_tr["SAMPLE_KEY"].astype(str), meta_te["SAMPLE_KEY"].astype(str)
    Xtr, Xte = Xtr.loc[meta_tr["SAMPLE_KEY"].tolist()].copy(), Xte.loc[meta_te["SAMPLE_KEY"].tolist()].copy()
    Xtr = _prefix_group_specific_pc_columns(Xtr, cancer_code=str(cancer_code))
    Xte = _prefix_group_specific_pc_columns(Xte, cancer_code=str(cancer_code))
    return Xtr, Xte, meta_tr, meta_te



def load_inputs(seed: int | None = None):
    prep_dir = PREP_OUTDIR if seed is None else _seed_prep_dir(seed)
    print(f"[LOAD_INPUTS] seed={seed} prep_dir={prep_dir}", flush=True)

    Xtr_parts, Xte_parts = [], []
    meta_tr_parts, meta_te_parts = [], []
    for cancer_code in ALL_CANCER_CODES:
        Xtr_i, Xte_i, meta_tr_i, meta_te_i = _load_exact_august_cancer_split(
            prep_dir, cancer_code
        )
        Xtr_parts.append(Xtr_i)
        Xte_parts.append(Xte_i)
        meta_tr_parts.append(meta_tr_i)
        meta_te_parts.append(meta_te_i)

    Xtr = _concat_align_frames(Xtr_parts)
    Xte = _concat_align_frames(Xte_parts)
    meta_tr = pd.concat(meta_tr_parts, axis=0, ignore_index=True)
    meta_te = pd.concat(meta_te_parts, axis=0, ignore_index=True)

    feat_names = list(map(str, Xtr.columns.tolist()))
    blocks = infer_blocks_from_feat_names(feat_names)
    Xtr_np = Xtr.to_numpy(dtype=float)
    Xte_np = Xte.to_numpy(dtype=float)
    t_tr = meta_tr["PFI_TIME_MONTHS"].to_numpy(dtype=float)
    t_te = meta_te["PFI_TIME_MONTHS"].to_numpy(dtype=float)
    e_tr = meta_tr["PFI_EVENT"].to_numpy(dtype=int)
    e_te = meta_te["PFI_EVENT"].to_numpy(dtype=int)
    groups = sorted(
        pd.concat([meta_tr["GROUP_LABEL"], meta_te["GROUP_LABEL"]])
        .astype(str)
        .unique()
        .tolist()
    )
    group_to_idx = {g: i for i, g in enumerate(groups)}
    g_tr = meta_tr["GROUP_LABEL"].astype(str).map(group_to_idx).to_numpy(dtype=int)
    g_te = meta_te["GROUP_LABEL"].astype(str).map(group_to_idx).to_numpy(dtype=int)
    return {
        "Xtr_np": Xtr_np,
        "Xte_np": Xte_np,
        "t_tr": t_tr,
        "t_te": t_te,
        "e_tr": e_tr,
        "e_te": e_te,
        "g_tr": g_tr,
        "g_te": g_te,
        "meta_tr": meta_tr,
        "meta_te": meta_te,
        "feat_names": feat_names,
        "blocks": blocks,
        "groups": groups,
        "prep_dir": str(prep_dir),
    }


def standardize_from_train(Xtr_np, Xother_np):
    mu = Xtr_np.mean(axis=0, keepdims=True)
    sd = Xtr_np.std(axis=0, keepdims=True)
    sd = np.where((sd <= 0.0) | ~np.isfinite(sd), 1.0, sd)
    Xtr_s = (Xtr_np - mu) / sd
    Xother_s = (Xother_np - mu) / sd
    return (
        torch.tensor(Xtr_s, dtype=torch.float32, device=device),
        torch.tensor(Xother_s, dtype=torch.float32, device=device),
    )


def subset_feature_matrices(Xtr_np, Xte_np, feat_names, surv_top_k):
    # Preprocessing already performs feature selection.
    # If no gene-PC columns are present, keep all columns.
    has_pc = any(
        re.match(r"^(BRCA|LUAD|COAD)_PC\d+$", str(f)) or re.match(r"^PC\d+$", str(f))
        for f in feat_names
    )
    if not has_pc:
        return Xtr_np, Xte_np, list(feat_names)

    def _pc_num(name):
        m = re.search(r"PC(\d+)$", str(name))
        return int(m.group(1)) if m else 10**9

    prefixed_pc_idxs = []
    prefixed_pc_all = []
    for cancer_code in ALL_CANCER_CODES:
        idxs = [
            i
            for i, f in enumerate(feat_names)
            if re.match(rf"^{cancer_code}_PC\d+$", str(f))
        ]
        idxs = sorted(idxs, key=lambda i: _pc_num(feat_names[i]))
        if len(idxs) > 0:
            if len(idxs) < int(surv_top_k):
                raise ValueError(
                    f"Requested SURV_TOP_K={surv_top_k}, but only {len(idxs)} prefixed PC columns exist for {cancer_code}."
                )
            prefixed_pc_idxs.extend(idxs[: int(surv_top_k)])
            prefixed_pc_all.extend(idxs)

    if len(prefixed_pc_idxs) == 0:
        pc_idxs = [i for i, f in enumerate(feat_names) if re.match(r"^PC\d+$", str(f))]
        other_idxs = [
            i for i, f in enumerate(feat_names) if not re.match(r"^PC\d+$", str(f))
        ]
        if len(pc_idxs) < int(surv_top_k):
            raise ValueError(
                f"Requested SURV_TOP_K={surv_top_k}, but only {len(pc_idxs)} PC columns exist in the design matrix."
            )
        keep_idxs = pc_idxs[: int(surv_top_k)] + other_idxs
    else:
        other_idxs = [
            i for i in range(len(feat_names)) if i not in set(prefixed_pc_all)
        ]
        keep_idxs = prefixed_pc_idxs + other_idxs

    keep_names = [feat_names[i] for i in keep_idxs]
    Xtr_sub = Xtr_np[:, keep_idxs]
    Xte_sub = Xte_np[:, keep_idxs]
    return Xtr_sub, Xte_sub, keep_names


def make_inner_val_split(g_tr, e_tr, val_frac=INNER_VAL_FRAC, seed=SEED):
    n = len(g_tr)
    idx = np.arange(n)
    strat = pd.Series(g_tr).astype(str) + "__" + pd.Series(e_tr).astype(str)
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=seed
        )
        fit_idx, val_idx = next(splitter.split(np.zeros((n, 1)), strat))
    except Exception:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        n_val = max(1, int(round(val_frac * n)))
        val_idx = np.sort(perm[:n_val])
        fit_idx = np.sort(perm[n_val:])
    return np.asarray(fit_idx, dtype=int), np.asarray(val_idx, dtype=int)


def make_dt_masks(times_np, events_np, edges_np):
    t = torch.tensor(times_np, dtype=torch.float32, device=device)
    e = torch.tensor(events_np, dtype=torch.float32, device=device)
    edges_t = torch.tensor(edges_np, dtype=torch.float32, device=device)
    J = edges_t.numel() - 1
    j_star = torch.bucketize(t, edges_t[1:], right=False)
    j_star = torch.clamp(j_star, 0, J - 1)
    grid = torch.arange(J, device=device)
    survive_before = (grid[None, :] < j_star[:, None]).float()
    event_mask = torch.zeros_like(survive_before)
    event_mask.scatter_(1, j_star[:, None], e[:, None])
    return survive_before, event_mask


def expected_dt_ll_mc(
    model, X, g_idx, at_risk, event_mask, alpha, S=MC_TRAIN, bin_weights=None
):
    def logs_logistic(eta):
        logh = -F.softplus(-eta)
        log1m = -F.softplus(eta)
        return logh, log1m

    g_idx = g_idx.to(device=X.device, dtype=torch.long)
    if bin_weights is None:
        bw = torch.ones(alpha.shape[0], device=X.device, dtype=X.dtype)
    else:
        bw = torch.as_tensor(bin_weights, device=X.device, dtype=X.dtype).view(1, -1)
    vals = []
    with gpytorch.settings.cholesky_jitter(_active_cholesky_jitter()):
        for _ in range(int(S)):
            f, _ = model.forward_score(X, g_idx, sample=True)
            eta = f[:, None] + alpha[None, :]
            logh, log1m = logs_logistic(eta)
            # at_risk marks bins strictly before the observed event/censoring bin.
            # Therefore event_mask contributes the event-bin log hazard directly.
            # Do not use logh - log1m here unless at_risk also includes the event bin.
            term1 = torch.where(at_risk > 0, log1m, torch.zeros_like(log1m)) * bw
            term2 = torch.where(event_mask > 0, logh, torch.zeros_like(logh)) * bw
            ll = term1.sum() + term2.sum()
            if not torch.isfinite(ll):
                logh = torch.where(torch.isfinite(logh), logh, torch.zeros_like(logh))
                log1m = torch.where(
                    torch.isfinite(log1m), log1m, torch.zeros_like(log1m)
                )
                term1 = torch.where(at_risk > 0, log1m, torch.zeros_like(log1m)) * bw
                term2 = torch.where(event_mask > 0, logh, torch.zeros_like(logh)) * bw
                ll = term1.sum() + term2.sum()
            vals.append(ll)
    return torch.stack(vals).mean()


def beta_schedule(it):
    return min(1.0, float(it) / max(1, KL_WARMUP))


def get_eval_horizons_with_median(t_fit, e_fit):
    hs = [float(h) for h in EVAL_HORIZONS]
    t_fit = np.asarray(t_fit, float)
    e_fit = np.asarray(e_fit, int)
    m = np.isfinite(t_fit) & (t_fit > 0) & np.isfinite(e_fit)
    t_use = t_fit[m]
    e_use = e_fit[m]
    if t_use.size > 0:
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(t_use, e_use)
            h_med = float(kmf.median_survival_time_)
            if not np.isfinite(h_med):
                h_med = float(np.median(t_use))
        except Exception:
            h_med = float(np.median(t_use))
        if not any(abs(h_med - h0) < 1e-8 for h0 in hs):
            hs.append(h_med)
    hs = sorted(hs)
    return hs


def pooled_c_on_eval_with_train_orientation(
    score_fit_mean, score_eval_mean, t_fit, t_eval, e_fit, e_eval, dt_edges, alpha_dt, H
):
    risk_fit_raw = risk_from_score_at_H(
        score_fit_mean, dt_edges, alpha_dt.detach().cpu().numpy(), H
    )
    risk_eval_raw = risk_from_score_at_H(
        score_eval_mean, dt_edges, alpha_dt.detach().cpu().numpy(), H
    )
    sign = train_orientation_sign(t_fit, e_fit, risk_fit_raw)
    risk_eval = apply_orientation(risk_eval_raw, sign)
    c_eval, _ = harrell_c(t_eval, risk_eval, e_eval, auto_orient=False)
    return float(c_eval)


def evaluate_model(
    score_fit_mean,
    score_eval_mean,
    t_fit,
    t_eval,
    e_fit,
    e_eval,
    g_fit,
    g_eval,
    groups,
    dt_edges,
    alpha_dt,
    eval_label="test",
    include_fit=False,
):
    row = {}
    for H in EVAL_HORIZONS:
        risk_fit_raw = risk_from_score_at_H(
            score_fit_mean, dt_edges, alpha_dt.detach().cpu().numpy(), H
        )
        risk_eval_raw = risk_from_score_at_H(
            score_eval_mean, dt_edges, alpha_dt.detach().cpu().numpy(), H
        )
        sign = train_orientation_sign(t_fit, e_fit, risk_fit_raw)
        risk_fit = apply_orientation(risk_fit_raw, sign)
        risk_eval = apply_orientation(risk_eval_raw, sign)
        c_fit, _ = harrell_c(t_fit, risk_fit, e_fit, auto_orient=False)
        c_eval, _ = harrell_c(t_eval, risk_eval, e_eval, auto_orient=False)
        if include_fit:
            row[f"c_train_H{int(H)}"] = float(c_fit)
        row[f"c_{eval_label}_H{int(H)}"] = float(c_eval)
        fit_txt = f"{c_fit:.4f}" if include_fit else "-"
        print(f"[Pooled] H={int(H):3d} train={fit_txt} | {eval_label}={c_eval:.4f}")
        for gi, gname in enumerate(groups):
            mfit = g_fit == gi
            meval = g_eval == gi
            c_gfit = np.nan
            c_geval = np.nan
            if (
                include_fit
                and int(mfit.sum()) >= 10
                and int(np.asarray(e_fit)[mfit].sum()) >= 3
            ):
                try:
                    c_gfit, _ = harrell_c(
                        np.asarray(t_fit)[mfit],
                        np.asarray(risk_fit)[mfit],
                        np.asarray(e_fit)[mfit],
                        auto_orient=False,
                    )
                except Exception:
                    c_gfit = np.nan
            if int(meval.sum()) >= 10 and int(np.asarray(e_eval)[meval].sum()) >= 3:
                try:
                    c_geval, _ = harrell_c(
                        np.asarray(t_eval)[meval],
                        np.asarray(risk_eval)[meval],
                        np.asarray(e_eval)[meval],
                        auto_orient=False,
                    )
                except Exception:
                    c_geval = np.nan
            if include_fit:
                row[f"{gname}_c_train_H{int(H)}"] = c_gfit
            row[f"{gname}_c_{eval_label}_H{int(H)}"] = c_geval
            fit_group_txt = f"{c_gfit:.4f}" if np.isfinite(c_gfit) else "nan"
            eval_group_txt = f"{c_geval:.4f}" if np.isfinite(c_geval) else "nan"
            if include_fit:
                print(
                    f"    [{gname}] H={int(H):3d} train={fit_group_txt} | {eval_label}={eval_group_txt}"
                )
            else:
                print(f"    [{gname}] H={int(H):3d} {eval_label}={eval_group_txt}")
    return row


def evaluate_model_mc(
    model,
    Xfit_t,
    Xeval_t,
    g_fit_t,
    g_eval_t,
    t_fit,
    t_eval,
    e_fit,
    e_eval,
    g_fit,
    g_eval,
    groups,
    dt_edges,
    alpha_dt,
    eval_label="test",
    include_fit=False,
    S_eval=FINAL_EVAL_MC_SAMPLES,
    eval_horizons=None,
):
    row = {}
    horizons = (
        get_eval_horizons_with_median(t_fit, e_fit)
        if eval_horizons is None
        else [float(h) for h in eval_horizons]
    )
    row["eval_median_horizon_months"] = (
        float(np.median(np.asarray(t_fit, float)))
        if len(np.asarray(t_fit, float))
        else np.nan
    )

    t_eval_np = np.asarray(t_eval, float)
    e_eval_np = np.asarray(e_eval, int)
    g_eval_np = np.asarray(g_eval, int)
    t_fit_np = np.asarray(t_fit, float)
    e_fit_np = np.asarray(e_fit, int)
    g_fit_np = np.asarray(g_fit, int)

    t_edges_eval, S_edges_eval = posterior_survival_curve_edges(
        model, Xeval_t, g_eval_t, alpha_dt, dt_edges, S=S_eval
    )
    mean_S_edges_eval = np.mean(S_edges_eval, axis=0)
    brier_time_grid_eval = np.asarray(t_edges_eval[1:], float)
    pooled_brier_curve_eval = []
    group_brier_curve_eval = {str(gname): [] for gname in groups}
    for j, t_point in enumerate(brier_time_grid_eval, start=1):
        pooled_brier_curve_eval.append(
            brier_score_ipcw_at_H(
                mean_S_edges_eval[:, j], t_eval_np, e_eval_np, float(t_point)
            )
        )
        for gi, gname in enumerate(groups):
            meval = g_eval_np == gi
            if int(meval.sum()) >= 10 and int(e_eval_np[meval].sum()) >= 3:
                group_brier_curve_eval[str(gname)].append(
                    brier_score_ipcw_at_H(
                        mean_S_edges_eval[meval, j],
                        t_eval_np[meval],
                        e_eval_np[meval],
                        float(t_point),
                    )
                )
            else:
                group_brier_curve_eval[str(gname)].append(np.nan)
    pooled_brier_curve_eval = np.asarray(pooled_brier_curve_eval, float)
    for key in list(group_brier_curve_eval.keys()):
        group_brier_curve_eval[key] = np.asarray(group_brier_curve_eval[key], float)

    for H in horizons:
        risk_fit_raw = posterior_neglogS_mean_at_H(
            model, Xfit_t, g_fit_t, H, alpha_dt, dt_edges, S=S_eval
        )
        risk_eval_raw = posterior_neglogS_mean_at_H(
            model, Xeval_t, g_eval_t, H, alpha_dt, dt_edges, S=S_eval
        )
        sign = train_orientation_sign(t_fit, e_fit, risk_fit_raw)
        risk_fit = apply_orientation(risk_fit_raw, sign)
        risk_eval = apply_orientation(risk_eval_raw, sign)
        c_fit, _ = harrell_c(t_fit, risk_fit, e_fit, auto_orient=False)
        c_eval, _ = harrell_c(t_eval, risk_eval, e_eval, auto_orient=False)
        dxy_eval = float(2.0 * c_eval - 1.0) if np.isfinite(c_eval) else np.nan
        surv_eval_H = np.mean(
            posterior_survival_at_H(
                model, Xeval_t, g_eval_t, H, alpha_dt, dt_edges, S=S_eval
            ),
            axis=0,
        )
        event_prob_eval_H = 1.0 - np.clip(surv_eval_H, 0.0, 1.0)
        brier_eval = brier_score_ipcw_at_H(surv_eval_H, t_eval_np, e_eval_np, float(H))
        ibs_eval = _ibs_from_brier_curve(
            brier_time_grid_eval, pooled_brier_curve_eval, float(H)
        )
        auroc_eval_row = _auroc_auprc_summary(
            event_prob_eval_H, t_eval_np, e_eval_np, float(H)
        )
        auroc_eval = (
            float(auroc_eval_row["auroc"])
            if np.isfinite(auroc_eval_row["auroc"])
            else np.nan
        )
        auprc_eval = (
            float(auroc_eval_row["auprc"])
            if np.isfinite(auroc_eval_row["auprc"])
            else np.nan
        )
        if include_fit:
            row[f"c_train_H{int(H)}"] = float(c_fit)
        row[f"c_{eval_label}_H{int(H)}"] = float(c_eval)
        row[f"somers_dxy_{eval_label}_H{int(H)}"] = (
            float(dxy_eval) if np.isfinite(dxy_eval) else np.nan
        )
        row[f"brier_{eval_label}_H{int(H)}"] = (
            float(brier_eval) if np.isfinite(brier_eval) else np.nan
        )
        row[f"ibs_{eval_label}_H{int(H)}"] = (
            float(ibs_eval) if np.isfinite(ibs_eval) else np.nan
        )
        row[f"auroc_{eval_label}_H{int(H)}"] = (
            float(auroc_eval) if np.isfinite(auroc_eval) else np.nan
        )
        row[f"auprc_{eval_label}_H{int(H)}"] = (
            float(auprc_eval) if np.isfinite(auprc_eval) else np.nan
        )
        fit_txt = f"{c_fit:.4f}" if include_fit else "-"
        c_eval_txt = f"{c_eval:.4f}" if np.isfinite(c_eval) else "nan"
        dxy_eval_txt = f"{dxy_eval:.4f}" if np.isfinite(dxy_eval) else "nan"
        ibs_eval_txt = f"{ibs_eval:.4f}" if np.isfinite(ibs_eval) else "nan"
        auroc_eval_txt = f"{auroc_eval:.4f}" if np.isfinite(auroc_eval) else "nan"
        auprc_eval_txt = f"{auprc_eval:.4f}" if np.isfinite(auprc_eval) else "nan"
        print(f"[Pooled-MC] H={int(H):3d} train={fit_txt} | {eval_label}={c_eval_txt}")
        print(
            f"            SomersDxy_{eval_label}={dxy_eval_txt} | IBS_{eval_label}={ibs_eval_txt} | AUROC_{eval_label}={auroc_eval_txt} | AUPRC_{eval_label}={auprc_eval_txt}"
        )
        for gi, gname in enumerate(groups):
            mfit = g_fit_np == gi
            meval = g_eval_np == gi
            c_gfit = np.nan
            c_geval = np.nan
            dxy_geval = np.nan
            brier_geval = np.nan
            ibs_geval = np.nan
            if include_fit and int(mfit.sum()) >= 10 and int(e_fit_np[mfit].sum()) >= 3:
                try:
                    c_gfit, _ = harrell_c(
                        t_fit_np[mfit],
                        np.asarray(risk_fit)[mfit],
                        e_fit_np[mfit],
                        auto_orient=False,
                    )
                except Exception:
                    c_gfit = np.nan
            if int(meval.sum()) >= 10 and int(e_eval_np[meval].sum()) >= 3:
                try:
                    c_geval, _ = harrell_c(
                        t_eval_np[meval],
                        np.asarray(risk_eval)[meval],
                        e_eval_np[meval],
                        auto_orient=False,
                    )
                except Exception:
                    c_geval = np.nan
                dxy_geval = (
                    float(2.0 * c_geval - 1.0) if np.isfinite(c_geval) else np.nan
                )
                try:
                    brier_geval = brier_score_ipcw_at_H(
                        surv_eval_H[meval], t_eval_np[meval], e_eval_np[meval], float(H)
                    )
                except Exception:
                    brier_geval = np.nan
                ibs_geval = _ibs_from_brier_curve(
                    brier_time_grid_eval,
                    group_brier_curve_eval[str(gname)],
                    float(H),
                )
                auroc_group_row = _auroc_auprc_summary(
                    event_prob_eval_H[meval],
                    t_eval_np[meval],
                    e_eval_np[meval],
                    float(H),
                )
                auroc_geval = (
                    float(auroc_group_row["auroc"])
                    if np.isfinite(auroc_group_row["auroc"])
                    else np.nan
                )
                auprc_geval = (
                    float(auroc_group_row["auprc"])
                    if np.isfinite(auroc_group_row["auprc"])
                    else np.nan
                )
            else:
                auroc_geval = np.nan
                auprc_geval = np.nan
            if include_fit:
                row[f"{gname}_c_train_H{int(H)}"] = c_gfit
            row[f"{gname}_c_{eval_label}_H{int(H)}"] = c_geval
            row[f"{gname}_somers_dxy_{eval_label}_H{int(H)}"] = (
                float(dxy_geval) if np.isfinite(dxy_geval) else np.nan
            )
            row[f"{gname}_brier_{eval_label}_H{int(H)}"] = (
                float(brier_geval) if np.isfinite(brier_geval) else np.nan
            )
            row[f"{gname}_ibs_{eval_label}_H{int(H)}"] = (
                float(ibs_geval) if np.isfinite(ibs_geval) else np.nan
            )
            row[f"{gname}_auroc_{eval_label}_H{int(H)}"] = (
                float(auroc_geval) if np.isfinite(auroc_geval) else np.nan
            )
            row[f"{gname}_auprc_{eval_label}_H{int(H)}"] = (
                float(auprc_geval) if np.isfinite(auprc_geval) else np.nan
            )
            fit_group_txt = f"{c_gfit:.4f}" if np.isfinite(c_gfit) else "nan"
            eval_group_txt = f"{c_geval:.4f}" if np.isfinite(c_geval) else "nan"
            dxy_group_txt = f"{dxy_geval:.4f}" if np.isfinite(dxy_geval) else "nan"
            ibs_group_txt = f"{ibs_geval:.4f}" if np.isfinite(ibs_geval) else "nan"
            auroc_group_txt = (
                f"{auroc_geval:.4f}" if np.isfinite(auroc_geval) else "nan"
            )
            auprc_group_txt = (
                f"{auprc_geval:.4f}" if np.isfinite(auprc_geval) else "nan"
            )
            if include_fit:
                print(
                    f"    [{gname}] H={int(H):3d} train={fit_group_txt} | {eval_label}={eval_group_txt}"
                )
            else:
                print(f"    [{gname}] H={int(H):3d} {eval_label}={eval_group_txt}")
            print(
                f"             SomersDxy_{eval_label}={dxy_group_txt} | IBS_{eval_label}={ibs_group_txt} | AUROC_{eval_label}={auroc_group_txt} | AUPRC_{eval_label}={auprc_group_txt}"
            )
    return row


def run_one_config(
    Xtr_np,
    Xte_np,
    t_tr,
    t_te,
    e_tr,
    e_te,
    g_tr,
    g_te,
    groups,
    feat_names,
    blocks,
    surv_top_k,
    shared_dim,
    private_dim,
    mt_latent_rank,
    output_rank,
):
    Xtr_sub_np, Xte_sub_np, feat_names_sub = subset_feature_matrices(
        Xtr_np, Xte_np, feat_names, surv_top_k
    )
    blocks_sub = infer_blocks_from_feat_names(feat_names_sub)

    Xfit_np = Xtr_sub_np
    t_fit, e_fit, g_fit = t_tr, e_tr, g_tr

    Xfit_t, Xte_t = standardize_from_train(Xfit_np, Xte_sub_np)
    g_fit_t = torch.tensor(g_fit, dtype=torch.long, device=device)
    g_te_t = torch.tensor(g_te, dtype=torch.long, device=device)

    dt_edges = make_discrete_time_edges(t_fit, DT_BINS)
    event_bin_fit = discretize_times(t_fit, dt_edges)
    event_bin_fit_t = torch.tensor(event_bin_fit, dtype=torch.long, device=device)
    e_fit_t = torch.tensor(e_fit, dtype=torch.float32, device=device)

    model = DGP(
        in_dim=Xfit_t.shape[1],
        Xref=Xfit_t,
        num_groups=len(groups),
        shared_dim=shared_dim,
        private_dim=private_dim,
        mt_shared_rank=mt_latent_rank,
        output_rank=output_rank,
        blocks=blocks_sub,
    ).to(device)
    verify_architecture(model)
    alpha_dt = torch.nn.Parameter(
        torch.zeros(len(dt_edges) - 1, device=device, dtype=torch.float32)
    )
    opt = build_optimizer(model, alpha_dt)
    AR_fit, EV_fit = make_dt_masks(t_fit, e_fit, dt_edges)
    bin_weights_t = torch.tensor(
        make_dt_bin_weights(dt_edges), dtype=torch.float32, device=device
    )

    best_train_obj = np.inf
    best_state = None
    for it in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        ell = expected_dt_ll_mc(
            model,
            Xfit_t,
            g_fit_t,
            AR_fit,
            EV_fit,
            alpha_dt,
            S=MC_TRAIN,
            bin_weights=bin_weights_t,
        )
        kl_s, kl_p, kl_o = total_kl_details(model)
        kl_total = kl_s + kl_p + kl_o
        kl_beta = beta_schedule(it)
        elbo = ell - kl_beta * kl_total
        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [alpha_dt], GRAD_CLIP)
        opt.step()

        train_obj = float(loss.detach().cpu())
        if it % PRINT_EVERY == 0 or it == 1:
            n_ev = max(1, int(np.sum(e_fit)))
            print(
                f"[{it:04d}] ELL/evt={float(ell.detach().cpu())/n_ev:.4f} | "
                f"KL/evt={(float(kl_total.detach().cpu())/n_ev):.4f} | "
                f"KLsh={(float(kl_s.detach().cpu())/n_ev):.4f} | "
                f"KLpr={(float(kl_p.detach().cpu())/n_ev):.4f} | "
                f"KLout={(float(kl_o.detach().cpu())/n_ev):.4f} | "
                f"beta={kl_beta:.3f} | ELBO={float(elbo.detach().cpu()):.4f}"
            )

        if train_obj < best_train_obj:
            best_train_obj = train_obj
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "alpha_dt": alpha_dt.detach().cpu().clone(),
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        alpha_dt.data.copy_(
            best_state["alpha_dt"].to(device=device, dtype=torch.float32)
        )

    model.eval()

    out = {}
    out.update(
        evaluate_model_mc(
            model,
            Xfit_t,
            Xte_t,
            g_fit_t,
            g_te_t,
            t_fit,
            t_te,
            e_fit,
            e_te,
            g_fit,
            g_te,
            groups,
            dt_edges,
            alpha_dt,
            eval_label="test",
            include_fit=True,
            S_eval=FINAL_EVAL_MC_SAMPLES,
            eval_horizons=get_eval_horizons_with_median(t_fit, e_fit),
        )
    )
    out["train_n"] = int(len(t_fit))
    out["test_n"] = int(len(t_te))
    out["inner_train_n"] = int(len(t_fit))
    out["inner_val_n"] = 0
    out["n_features_used"] = int(Xfit_t.shape[1])
    out["pc_features_per_group"] = int(surv_top_k)
    out["pc_features_used_total"] = int(
        sum(
            1
            for f in feat_names_sub
            if re.match(r"^(BRCA|LUAD|COAD)_PC\d+$", str(f))
            or re.match(r"^PC\d+$", str(f))
        )
    )
    out["extra_nonpc_features_used"] = int(
        Xfit_t.shape[1] - out["pc_features_used_total"]
    )
    out["best_train_objective"] = float(best_train_obj)
    return out


CURRENT_SPLIT_PLOT_LABEL = None


def set_current_split_plot_label(label=None):
    global CURRENT_SPLIT_PLOT_LABEL
    CURRENT_SPLIT_PLOT_LABEL = None if label is None else str(label)


def _stamp_current_split_label(fig=None):
    label = globals().get("CURRENT_SPLIT_PLOT_LABEL", None)
    if not label:
        return
    try:
        fig = fig if fig is not None else plt.gcf()
        if getattr(fig, "_split_label_stamped", None) == label:
            return
        fig.text(
            0.995,
            0.005,
            label,
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=0.80,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
        )
        setattr(fig, "_split_label_stamped", label)
    except Exception:
        pass


def _save_figure_with_source(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_path = path.with_name(path.name + ".source_data.csv")
    rows = []
    for ai, ax in enumerate(fig.axes):
        for li, line in enumerate(ax.get_lines()):
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            for k in range(min(len(x), len(y))):
                rows.append({"axis": ai, "artist_type": "line", "artist_index": li,
                             "point_index": k, "x": x[k], "y": y[k]})
        for ci, coll in enumerate(ax.collections):
            try:
                offsets = np.asarray(coll.get_offsets())
                for k, xy in enumerate(offsets):
                    rows.append({"axis": ai, "artist_type": "collection", "artist_index": ci,
                                 "point_index": k, "x": xy[0], "y": xy[1]})
            except Exception:
                pass
        for pi, rect in enumerate(ax.patches):
            if hasattr(rect, "get_x") and hasattr(rect, "get_height"):
                rows.append({"axis": ai, "artist_type": "patch", "artist_index": pi,
                             "point_index": 0, "x": rect.get_x(), "y": rect.get_y(),
                             "width": rect.get_width(), "height": rect.get_height()})
        for ii, im in enumerate(ax.images):
            try:
                A = np.asarray(im.get_array())
                if A.size <= 1000000:
                    for idx in np.ndindex(A.shape):
                        rows.append({"axis": ai, "artist_type": "image", "artist_index": ii,
                                     "point_index": ":".join(map(str, idx)), "value": A[idx]})
            except Exception:
                pass
    if not rows:
        rows = [{"axis": np.nan, "artist_type": "metadata_only_no_numeric_artist",
                 "artist_index": np.nan, "point_index": np.nan,
                 "title": " | ".join(ax.get_title() for ax in fig.axes)}]
    pd.DataFrame(rows).to_csv(source_path, index=False)
    # Validate source data before plotting, as required for NPJ source-data recovery.
    pd.read_csv(source_path)
    _stamp_current_split_label(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if path.suffix.lower() == ".png" and os.getenv("NPJ_SAVE_PDF_COMPANIONS", "1").lower() in {"1", "true", "yes"}:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def savefig(path):
    fig = plt.gcf()
    _save_figure_with_source(fig, path)
    plt.close(fig)


def _safe_fname(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:180] if len(s) > 180 else s


def _np1d(x, dtype=float):
    return np.ravel(np.asarray(x, dtype=dtype))


def _raw_spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 3:
        return float("nan")
    xr = pd.Series(x[m]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[m]).rank(method="average").to_numpy(dtype=float)
    if xr.size < 3:
        return float("nan")
    rho = np.corrcoef(xr, yr)[0, 1]
    return float(rho) if np.isfinite(rho) else float("nan")


_PC_FEATURE_RE = re.compile(r"^(?:(BRCA|LUAD|COAD)_)?(PC\d+)$")


def _parse_pc_feature_name(feature_name):
    m = _PC_FEATURE_RE.match(str(feature_name))
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _has_pc_features(feat_names):
    return any(_parse_pc_feature_name(str(f))[1] is not None for f in feat_names)


def _corr_from_cov(C):
    C = np.asarray(C, float)
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / (d[:, None] * d[None, :])


def _extract_lmc_task_corr(vs, num_tasks):
    try:
        W = vs.lmc_coefficients.detach().cpu().numpy()
    except Exception:
        return np.eye(int(num_tasks), dtype=float)
    W = np.asarray(W, float)
    if W.ndim == 2:
        if W.shape[0] == num_tasks:
            B = W @ W.T
        elif W.shape[1] == num_tasks:
            B = W.T @ W
        else:
            B = np.eye(int(num_tasks), dtype=float)
    else:
        B = np.eye(int(num_tasks), dtype=float)
    return _corr_from_cov(B)


# Posterior prediction can be numerically harder than training because posthoc
# FC repeatedly samples large test-set GP marginals after deliberately
# perturbing/collapsing one feature at a time.  Chunking preserves the marginal
# posterior draws needed by patient-wise FC while avoiding large, nearly singular
# covariance root decompositions.  The jitter ladder handles duplicated or
# near-duplicated rows created by within-group shuffling of sparse/binary columns.
PREDICT_SCORE_CHUNK_SIZE = 96
PREDICT_SCORE_JITTERS = (1e-4, 1e-3, 1e-2, 5e-2)
PREDICT_SCORE_ALLOW_MEAN_FALLBACK = os.getenv(
    "NPJ_ALLOW_POSTERIOR_MEAN_FALLBACK", "0"
).strip().lower() in {"1", "true", "yes"}


def _active_cholesky_jitter(default=1e-3):
    """Jitter selected by the current deterministic fit retry attempt."""
    try:
        return float(os.getenv("NPJ_ACTIVE_CHOLESKY_JITTER", str(default)))
    except Exception:
        return float(default)


def _is_linalg_numerical_failure(exc):
    msg = str(exc).lower()
    return (
        "linalg.eigh" in msg
        or "cholesky" in msg
        or "not positive definite" in msg
        or "ill-conditioned" in msg
        or "singular" in msg
        or "nan" in msg
    )


def _forward_score_sample_block_with_retries(model, X_block, g_block, sample=True):
    last_exc = None
    for jitter in PREDICT_SCORE_JITTERS:
        try:
            with gpytorch.settings.cholesky_jitter(float(jitter)):
                with gpytorch.settings.max_cholesky_size(1000000):
                    with gpytorch.settings.fast_computations(
                        covar_root_decomposition=False,
                        log_prob=False,
                        solves=False,
                    ):
                        f, _ = model.forward_score(X_block, g_block, sample=sample)
            if torch.isfinite(f).all():
                return f
            last_exc = RuntimeError("nonfinite values in posterior score sample")
        except Exception as exc:
            last_exc = exc
            if not _is_linalg_numerical_failure(exc):
                raise

    # If a block still fails, recursively split it.  This keeps posterior
    # sampling stochastic for all patients where the GP marginal can be sampled.
    if sample and X_block.shape[0] > 1:
        mid = X_block.shape[0] // 2
        f_left = _forward_score_sample_block_with_retries(
            model, X_block[:mid], g_block[:mid], sample=True
        )
        f_right = _forward_score_sample_block_with_retries(
            model, X_block[mid:], g_block[mid:], sample=True
        )
        return torch.cat([f_left, f_right], dim=0)

    # Publication mode does not silently replace posterior draws with the mean.
    # An opt-in compatibility fallback remains available, but is off by default.
    if PREDICT_SCORE_ALLOW_MEAN_FALLBACK:
        print(
            "[POSTHOC][WARN] Posterior score sampling failed after jitter retries; "
            "using posterior mean for this draw/block. Original error: "
            f"{type(last_exc).__name__}: {last_exc}",
            flush=True,
        )
        with torch.no_grad():
            f, _ = model.forward_score(X_block, g_block, sample=False)
        return f

    raise RuntimeError(
        "posterior score sampling failed after the complete jitter ladder"
    ) from last_exc


def predict_score_samples(model, X_t, g_t, S=FINAL_EVAL_MC_SAMPLES, chunk_size=PREDICT_SCORE_CHUNK_SIZE):
    Fs = []
    model.eval()
    n = int(X_t.shape[0])
    chunk_size = int(max(1, min(int(chunk_size), max(n, 1))))
    with torch.no_grad():
        for _ in range(int(S)):
            parts = []
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                f_block = _forward_score_sample_block_with_retries(
                    model,
                    X_t[start:stop],
                    g_t[start:stop],
                    sample=True,
                )
                parts.append(f_block.detach().cpu().numpy())
            Fs.append(np.concatenate(parts, axis=0))
    return np.asarray(Fs, float)


def posterior_survival_curve_edges(
    model, X_t, g_t, alpha_dt, dt_edges, S=FINAL_PRED_BAND_MC
):
    F = predict_score_samples(model, X_t, g_t, S=S)  # [S,N]
    alpha = _np1d(alpha_dt.detach().cpu().numpy(), dtype=float)
    logits = F[:, :, None] + alpha[None, None, :]
    haz = 1.0 / (1.0 + np.exp(-logits))
    Scurves = np.cumprod(1.0 - haz, axis=2)
    Sedges = np.concatenate(
        [np.ones((Scurves.shape[0], Scurves.shape[1], 1), dtype=float), Scurves], axis=2
    )
    t_edges = np.concatenate([[0.0], np.asarray(dt_edges[1:], float)])
    return t_edges, Sedges


def posterior_survival_at_H(
    model, X_t, g_t, H, alpha_dt, dt_edges, S=FINAL_EVAL_MC_SAMPLES
):
    F = predict_score_samples(model, X_t, g_t, S=S)
    alpha = _np1d(alpha_dt.detach().cpu().numpy(), dtype=float)
    logits = F[:, :, None] + alpha[None, None, :]
    haz = 1.0 / (1.0 + np.exp(-logits))
    right_edges = np.asarray(dt_edges[1:], float)
    alive_bins = np.where(right_edges <= float(H))[0]
    if alive_bins.size == 0:
        return np.ones((F.shape[0], F.shape[1]), dtype=float)
    S_H = np.prod(1.0 - haz[:, :, alive_bins], axis=2)
    return np.clip(S_H, 1e-12, 1.0)


def posterior_neglogS_mean_at_H(
    model, X_t, g_t, H, alpha_dt, dt_edges, S=FINAL_EVAL_MC_SAMPLES
):
    SH = posterior_survival_at_H(
        model, X_t, g_t, float(H), alpha_dt, dt_edges, S=int(S)
    )
    return np.mean(-np.log(np.clip(SH, 1e-12, 1.0)), axis=0)


def neglogS_samples_at_H_mt(model, X_ref_s, g_ref_t, H, alpha_dt, dt_edges, S=FC_S):
    X = torch.tensor(np.asarray(X_ref_s, float), device=device, dtype=torch.float32)
    SH = posterior_survival_at_H(
        model, X, g_ref_t, float(H), alpha_dt, dt_edges, S=int(S)
    )
    return -np.log(np.clip(SH, 1e-12, 1.0))


def _wasserstein1_empirical_per_row(Y0, Y1, q_grid=None):
    """Patient-wise empirical 1-Wasserstein distances for posterior risk draws.

    Y0 and Y1 are arrays with shape [posterior_draw, patient]. The function
    compares the baseline and collapsed posterior predictive risk distribution
    separately for each patient. It is deliberately nan-safe because a single
    nonfinite draw should not erase the entire feature-collapsing score.
    """
    Y0 = np.asarray(Y0, float)
    Y1 = np.asarray(Y1, float)
    if Y0.ndim == 1:
        Y0 = Y0[:, None]
    if Y1.ndim == 1:
        Y1 = Y1[:, None]
    if Y0.shape[1] != Y1.shape[1]:
        raise ValueError(
            f"Y0 and Y1 must have the same number of patients; got {Y0.shape} and {Y1.shape}."
        )
    if q_grid is None:
        q_grid = np.linspace(0.005, 0.995, 199)
    out = np.full(Y0.shape[1], np.nan, dtype=float)
    for i in range(Y0.shape[1]):
        a = Y0[:, i]
        b = Y1[:, i]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            continue
        if a.size == b.size:
            out[i] = float(np.mean(np.abs(np.sort(a) - np.sort(b))))
        else:
            qa = np.quantile(a, q_grid)
            qb = np.quantile(b, q_grid)
            out[i] = float(np.mean(np.abs(qa - qb)))
    return out


def _nanmean_or_zero(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else 0.0


def _nanstd_or_zero(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if x.size >= 2 else 0.0


def post_groups(SH, context="posterior-H tertiles", warn_on_fallback=True):
    """Assign low/medium/high predicted-risk groups by posterior-mean survival tertiles.

    Low predicted risk = highest posterior mean survival at H.
    High predicted risk = lowest posterior mean survival at H.

    This function intentionally does not use TAU_HIGH, TAU_LOW, or P_STAR.
    Those constants are retained elsewhere only for backward compatibility with
    older outputs/configuration files.
    """
    SH = np.asarray(SH, float)
    if SH.ndim != 2:
        raise ValueError(
            f"post_groups expects a 2D array of posterior survival samples; got shape {SH.shape}."
        )

    n = int(SH.shape[1])
    mean_S = np.mean(SH, axis=0) if n else np.zeros(0, dtype=float)

    # Kept for backwards-compatible return shape.  Downstream callers currently
    # ignore these arrays, but they now encode the tertile basis rather than tau
    # probabilities: p_lo = mean survival, p_hi = mean failure/risk score.
    p_lo = mean_S.copy()
    p_hi = 1.0 - mean_S

    gL = np.zeros(n, dtype=bool)
    gM = np.zeros(n, dtype=bool)
    gH = np.zeros(n, dtype=bool)
    stats = {
        "context": str(context),
        "method": "posterior_mean_survival_rank_tertiles",
        "grouping_basis": "posterior_mean_survival_at_H",
        "tau_based": False,
        "fallback_used": False,
        "fallback_reason": "",
    }

    if n == 0:
        stats.update({"n_low": 0, "n_med": 0, "n_high": 0})
        return gL, gM, gH, p_lo, p_hi, stats

    if n < 3:
        stats.update(
            {
                "method": "nonestimable_three_tertiles",
                "fallback_used": False,
                "fallback_reason": "",
                "estimable": False,
                "nonestimable_reason": "too few patients for three tertiles",
            }
        )
    else:
        # Sort by mean survival: lowest survival = highest predicted risk.
        # Rank-based tertiles avoid empty groups from tied quantile cutpoints.
        order = np.argsort(mean_S, kind="mergesort")
        k = max(1, n // 3)
        high_idx = order[:k]
        low_idx = order[-k:]
        gH[high_idx] = True
        gL[low_idx] = True
        gM = (~gL) & (~gH)

        stats.update(
            {
                "q_low_mean_survival_cut": float(np.max(mean_S[high_idx])) if high_idx.size else np.nan,
                "q_high_mean_survival_cut": float(np.min(mean_S[low_idx])) if low_idx.size else np.nan,
                "mean_survival_min": float(np.min(mean_S)) if mean_S.size else np.nan,
                "mean_survival_max": float(np.max(mean_S)) if mean_S.size else np.nan,
            }
        )

    stats.update(
        {
            "n_low": int(np.sum(gL)),
            "n_med": int(np.sum(gM)),
            "n_high": int(np.sum(gH)),
        }
    )
    return gL, gM, gH, p_lo, p_hi, stats


def train_defined_tertiles(SH_train, SH_apply, context="train-defined risk tertiles"):
    """Learn q1/q2 from training event risk and apply unchanged; never fallback."""
    SH_train = np.asarray(SH_train, float)
    SH_apply = np.asarray(SH_apply, float)
    if SH_train.ndim != 2 or SH_apply.ndim != 2:
        raise ValueError("train_defined_tertiles requires [draw, patient] matrices")
    rtr = 1.0 - np.nanmean(SH_train, axis=0)
    rap = 1.0 - np.nanmean(SH_apply, axis=0)
    ok = np.isfinite(rtr)
    n = int(ok.sum())
    q1 = q2 = np.nan
    reason = ""
    if n < 3:
        reason = "fewer than three finite training risk scores"
    else:
        q1, q2 = np.quantile(rtr[ok], [1.0 / 3.0, 2.0 / 3.0])
        if not (np.isfinite(q1) and np.isfinite(q2) and q1 < q2):
            reason = "training tertile cutpoints collapsed because of tied risk scores"
    if reason:
        low = med = high = np.zeros(len(rap), dtype=bool)
    else:
        # Freeze the training cutpoints and apply the prespecified boundaries:
        # low < q1, intermediate q1 <= risk < q2, and high >= q2.
        low = np.isfinite(rap) & (rap < q1)
        med = np.isfinite(rap) & (rap >= q1) & (rap < q2)
        high = np.isfinite(rap) & (rap >= q2)
        if min(int(low.sum()), int(med.sum()), int(high.sum())) < 1:
            reason = "one or more held-out tertiles are empty under training cutpoints"
            # Preserve every valid held-out assignment. Only the three-group KM
            # comparison is non-estimable; test groups are not forced to be equal.
    stats = {"context": str(context), "method": "training_event_risk_quantile_cutpoints",
             "grouping_basis": "posterior_mean_event_probability_1_minus_S",
             "q1_train": q1, "q2_train": q2, "n_train_scores": n,
             "n_low": int(low.sum()), "n_med": int(med.sum()), "n_high": int(high.sum()),
             "estimable": not bool(reason), "nonestimable_reason": reason,
             "fallback_used": False, "fallback_reason": ""}
    return low, med, high, 1.0 - rap, rap, stats


def _format_logrank_p(pval):
    if not np.isfinite(pval):
        return None
    if float(pval) < 1e-16:
        return r"Log-rank $p < 10^{-16}$"
    if float(pval) < 1e-3:
        return rf"Log-rank $p = {float(pval):.2e}$"
    return rf"Log-rank $p = {float(pval):.4f}$"


def _clean_km_title(title):
    title = str(title)
    title = title.replace("posterior-risk tertiles", "predicted-risk tertiles")
    title = title.replace("Posterior-H strata", "predicted-risk tertiles")
    title = title.replace("Empirical Kaplan–Meier", "Kaplan–Meier survival")
    title = title.replace("Test-set Kaplan–Meier", "Test-set Kaplan–Meier survival")
    title = title.replace("Test KM", "Test-set Kaplan–Meier survival")
    return title


def _km_xlim(t, x_max=None):
    t = np.asarray(t, float)
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return (0.0, float(x_max) if x_max is not None else 1.0)
    if x_max is None:
        right = float(np.nanmax(finite))
    else:
        right = min(float(x_max), float(np.nanmax(finite)))
    return (0.0, max(1.0, right))


def km_plot_groups(t, e, masks, labels, title, outname, admin_cap=False, x_max=None):
    """Kaplan-Meier plot with safe display-window median annotations.

    Important plotting detail: when x_max is used, medians can lie outside the
    displayed x-axis window.  Drawing off-axis median labels makes Matplotlib's
    bbox_inches="tight" save a huge blank canvas.  Therefore median lines are
    only drawn on the axis when the median lies inside the current display
    window; otherwise the median/NR information is written in the side panel.
    """
    t = np.asarray(t, float).copy()
    e = np.asarray(e, int).copy()
    if admin_cap and (KM_ADMIN_CAP_MONTHS is not None):
        over = t > float(KM_ADMIN_CAP_MONTHS)
        e[over] = 0
        t[over] = float(KM_ADMIN_CAP_MONTHS)

    x_left, x_right_display = _km_xlim(t, x_max=x_max)
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    ax.set_xlim(x_left, x_right_display)
    ax.set_ylim(0.0, 1.02)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6, lw=1.0)

    km_fitters = []
    kept = []
    kept_labels = []
    for m, lab in zip(masks, labels):
        m = np.asarray(m, bool)
        if int(m.sum()) < 2:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(t[m], e[m], label=str(lab))
        kmf.plot_survival_function(ax=ax, ci_show=True)
        color = ax.get_lines()[-1].get_color() if len(ax.get_lines()) else None
        km_fitters.append((kmf, color))
        kept.append(m)
        kept_labels.append(str(lab))

    pval = np.nan
    if len(kept) >= 2:
        grp = np.full(len(t), -1, dtype=int)
        for i, m in enumerate(kept):
            grp[m] = i
        mm = grp >= 0
        try:
            pval = float(multivariate_logrank_test(t[mm], grp[mm], e[mm]).p_value)
        except Exception:
            pval = np.nan

    # Median labels: draw on-axis only if visible; otherwise move them to the side panel.
    side_lines = []
    for kmf, color in km_fitters:
        med = float(kmf.median_survival_time_)
        label = str(getattr(kmf, "_label", "stratum"))
        if np.isfinite(med):
            if (med >= x_left) and (med <= x_right_display):
                ax.vlines(med, 0.0, 0.5, colors=color, linestyles="--", lw=1.2, alpha=0.95)
                ax.text(
                    med,
                    0.03,
                    f"{med:.1f} mo",
                    rotation=90,
                    color=color,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    clip_on=True,
                )
            else:
                side_lines.append((color, f"{label}: median {med:.1f} mo"))
        else:
            side_lines.append((color, f"{label}: NR"))

    p_txt = _format_logrank_p(pval)

    ax.set_title(_clean_km_title(title))
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_xlim(x_left, x_right_display)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.18)

    if len(km_fitters):
        handles, labs = ax.get_legend_handles_labels()
        if len(handles):
            ax.legend(
                handles,
                labs,
                frameon=True,
                loc="upper left",
                bbox_to_anchor=(1.03, 1.0),
                borderaxespad=0.0,
                title="Risk stratum",
            )
        add_at_risk_counts(*[kmf for kmf, _ in km_fitters], ax=ax)

    y_side = 0.70
    if p_txt is not None:
        ax.text(
            1.03,
            y_side,
            p_txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0),
        )
        y_side -= 0.08
    if x_max is not None:
        ax.text(
            1.03,
            y_side,
            f"Display window: 0–{int(float(x_max))} mo",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0),
        )
        y_side -= 0.07
    for color, line in side_lines:
        ax.text(
            1.03,
            y_side,
            line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=color,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )
        y_side -= 0.055

    fig.subplots_adjust(bottom=0.30, right=0.76)
    outname = Path(outname)
    outname.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_with_source(fig, outname)
    plt.close(fig)
    return pval


def _save_km_display_window_variants(
    t, e, masks, labels, title, base_outname, admin_cap=False
):
    """Save additional x-axis-limited KM copies for report figures.

    These copies only change the displayed x-axis range. They do not censor,
    subset, or alter any event/censoring information used to fit the KM curves.
    """
    base = Path(base_outname)
    for x_max in [float(x) for x in KM_DISPLAY_LIMIT_MONTHS]:
        suffix = f"__xmax{int(x_max)}mo"
        limited_out = base.with_name(f"{base.stem}{suffix}{base.suffix}")
        km_plot_groups(
            t,
            e,
            masks,
            labels,
            f"{title} — 0–{int(x_max)} mo display",
            limited_out,
            admin_cap=admin_cap,
            x_max=x_max,
        )


def plot_model_pred_survival_bands(
    t_edges, S_edges, masks, title, outpath, q_lo=0.05, q_hi=0.95
):
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    for lab, m in masks.items():
        m = np.asarray(m, bool)
        if int(m.sum()) < 1:
            continue
        G = np.mean(S_edges[:, m, :], axis=1)
        mu = np.mean(G, axis=0)
        lo = np.quantile(G, q_lo, axis=0)
        hi = np.quantile(G, q_hi, axis=0)
        ax.step(t_edges, mu, where="post", label=str(lab))
        ax.fill_between(t_edges, lo, hi, step="post", alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("Months")
    ax.set_ylabel("Predicted survival")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.18)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            legend_labels,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    fig.subplots_adjust(right=0.80)
    savefig(outpath)


def plot_km_and_pred_survival_side_by_side(
    t,
    e,
    masks,
    labels,
    km_title,
    pred_title,
    outpath,
    admin_cap,
    t_edges,
    S_edges,
    pred_masks,
    q_lo=0.05,
    q_hi=0.95,
):
    t = np.asarray(t, float).copy()
    e = np.asarray(e, int).copy()
    if admin_cap and (KM_ADMIN_CAP_MONTHS is not None):
        over = t > float(KM_ADMIN_CAP_MONTHS)
        e[over] = 0
        t[over] = float(KM_ADMIN_CAP_MONTHS)
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.2))
    axs[0].axhline(0.5, ls="--", color="gray", alpha=0.6, lw=1.0)
    km_fitters = []
    kept = []
    kmf = KaplanMeierFitter()
    for m, lab in zip(masks, labels):
        m = np.asarray(m, bool)
        if int(m.sum()) < 2:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(t[m], e[m], label=str(lab))
        kmf.plot_survival_function(ax=axs[0], ci_show=True)
        color = axs[0].get_lines()[-1].get_color() if len(axs[0].get_lines()) else None
        km_fitters.append((kmf, color))
        kept.append(m)
    pval = np.nan
    if len(kept) >= 2:
        grp = np.full(len(t), -1, dtype=int)
        for i, m in enumerate(kept):
            grp[m] = i
        mm = grp >= 0
        try:
            pval = float(multivariate_logrank_test(t[mm], grp[mm], e[mm]).p_value)
        except Exception:
            pval = np.nan
    x_right = float(np.nanmax(t)) if np.isfinite(np.nanmax(t)) else 0.0
    y_text = 0.98
    for kmf, color in km_fitters:
        med = float(kmf.median_survival_time_)
        if np.isfinite(med):
            axs[0].vlines(
                med, 0.0, 0.5, colors=color, linestyles="--", lw=1.0, alpha=0.95
            )
        else:
            axs[0].text(
                x_right * 0.97,
                y_text,
                f"{kmf._label}: NR",
                color=color,
                ha="right",
                va="top",
                fontsize=9,
            )
            y_text -= 0.08
    p_txt = _format_logrank_p(pval)
    axs[0].set_title(_clean_km_title(km_title))
    axs[0].set_xlabel("Time (months)")
    axs[0].set_ylabel("Survival probability")
    axs[0].grid(True, alpha=0.18)
    axs[0].set_ylim(0.0, 1.02)
    axs[0].set_xlim(*_km_xlim(t, x_max=None))
    for lab, m in pred_masks.items():
        m = np.asarray(m, bool)
        if int(m.sum()) < 1:
            continue
        G = np.mean(S_edges[:, m, :], axis=1)
        mu = np.mean(G, axis=0)
        lo = np.quantile(G, q_lo, axis=0)
        hi = np.quantile(G, q_hi, axis=0)
        axs[1].step(t_edges, mu, where="post", label=str(lab))
        axs[1].fill_between(t_edges, lo, hi, step="post", alpha=0.18)
    axs[1].set_title(_clean_km_title(pred_title))
    axs[1].set_xlabel("Months")
    axs[1].set_ylabel("Predicted survival")
    axs[1].grid(True, alpha=0.18)
    axs[1].set_ylim(0.0, 1.02)
    h_leg, l_leg = axs[1].get_legend_handles_labels()
    if len(h_leg):
        fig.legend(
            h_leg,
            l_leg,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(0.86, 0.98),
            borderaxespad=0.0,
        )
    if p_txt is not None:
        fig.text(0.86, 0.72, p_txt, ha="left", va="top", fontsize=10)
    fig.subplots_adjust(right=0.82, wspace=0.28)
    savefig(outpath)


def _shared_latent_mean_for_group(model, Xg, gi):
    g_local = torch.full((Xg.size(0),), int(gi), dtype=torch.long, device=Xg.device)
    return model.shared_latent_mt(Xg, g_local, sample=False).detach().cpu().numpy()


def _private_latent_mean_for_group(model, Xg, gi):
    dist = model.private[int(gi)](Xg)
    vals = dist.mean
    if vals.dim() == 3:
        vals = vals.mean(dim=0)
    return vals.transpose(0, 1).contiguous().detach().cpu().numpy()


def _plot_latent_structure_heatmap(model, Xte_t, g_te_t, groups, outdir, suffix=""):
    rows_priv, rows_shared, labels = [], [], []
    with torch.no_grad():
        for gi, gname in enumerate(groups):
            m = g_te_t == gi
            if not bool(m.any()):
                continue
            Xg = Xte_t[m]
            priv = _private_latent_mean_for_group(model, Xg, gi)
            sh = _shared_latent_mean_for_group(model, Xg, gi)
            rows_priv.append(np.mean(priv, axis=0))
            rows_shared.append(np.mean(sh, axis=0))
            labels.append(str(gname))
    if len(labels) == 0:
        return
    M_priv = np.asarray(rows_priv, float)
    M_shared = np.asarray(rows_shared, float)
    raw = np.concatenate([M_priv, M_shared], axis=1)
    df = pd.DataFrame(
        raw,
        index=labels,
        columns=[f"P{i+1}" for i in range(M_priv.shape[1])]
        + [f"S{i+1}" for i in range(M_shared.shape[1])],
    )
    df.insert(0, "group", df.index)
    df.to_csv(Path(outdir) / f"latent_structure_group_means{suffix}.csv", index=False)
    if not ENABLE_HEATMAPS:
        return
    Z = raw.copy()
    Z = (Z - np.mean(Z, axis=0, keepdims=True)) / np.where(
        np.std(Z, axis=0, keepdims=True) <= 1e-8, 1.0, np.std(Z, axis=0, keepdims=True)
    )
    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.7 * Z.shape[1]), max(4.8, 0.55 * Z.shape[0]))
    )
    im = ax.imshow(Z, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(Z.shape[1]))
    ax.set_xticklabels(df.columns[1:], rotation=45, ha="right")
    ax.set_yticks(np.arange(Z.shape[0]))
    ax.set_yticklabels(labels)
    ax.set_title("Latent structure heatmap (cancer-group mean z-scores)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    savefig(Path(outdir) / f"latent_structure_heatmap{suffix}.png")


def _export_mt_shared_latent_matrices(model, groups, outdir, suffix=""):
    mats = []
    for gp_dim in model.shared_latent_mt.dims:
        mats.append(_extract_lmc_task_corr(gp_dim.variational_strategy, len(groups)))
    if len(mats) == 0:
        return
    Cm = np.mean(np.stack(mats, axis=0), axis=0)
    df = pd.DataFrame(
        Cm, index=[str(g) for g in groups], columns=[str(g) for g in groups]
    )
    df.insert(0, "group", df.index)
    df.to_csv(Path(outdir) / f"mt_shared_latent_group_corr{suffix}.csv", index=False)
    if not ENABLE_HEATMAPS:
        return
    fig, ax = plt.subplots(
        figsize=(max(6.4, 0.52 * len(groups)), max(5.0, 0.52 * len(groups)))
    )
    im = ax.imshow(Cm, aspect="equal", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels([str(g) for g in groups])
    ax.set_title("Implied multitask shared latent correlation across cancer groups")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    savefig(Path(outdir) / f"mt_shared_latent_group_corr_heatmap{suffix}.png")


def _export_mt_output_matrices(model, groups, outdir, suffix=""):
    if isinstance(getattr(model, "output", None), torch.nn.ModuleList):
        Corr = np.eye(len(groups), dtype=float)
        df = pd.DataFrame(
            Corr, index=[str(g) for g in groups], columns=[str(g) for g in groups]
        )
        df.insert(0, "group", df.index)
        df.to_csv(
            Path(outdir) / f"group_specific_output_identity_corr{suffix}.csv",
            index=False,
        )
        rows = []
        for gi, ogp in enumerate(model.output):
            ls = getattr(
                getattr(ogp.covar_module, "base_kernel", None), "lengthscale", None
            )
            os = getattr(ogp.covar_module, "outputscale", None)
            rows.append(
                {
                    "group": str(groups[gi]),
                    "median_lengthscale": (
                        float(np.median(ls.detach().cpu().numpy()))
                        if ls is not None
                        else np.nan
                    ),
                    "outputscale": (
                        float(os.detach().cpu().numpy()) if os is not None else np.nan
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(
            Path(outdir) / f"group_specific_output_kernel_stats{suffix}.csv",
            index=False,
        )
        return
    Corr = _extract_lmc_task_corr(model.output.variational_strategy, len(groups))
    df = pd.DataFrame(
        Corr, index=[str(g) for g in groups], columns=[str(g) for g in groups]
    )
    df.insert(0, "group", df.index)
    df.to_csv(Path(outdir) / f"mt_output_group_corr{suffix}.csv", index=False)
    if not ENABLE_HEATMAPS:
        return
    fig, ax = plt.subplots(
        figsize=(max(6.4, 0.52 * len(groups)), max(5.0, 0.52 * len(groups)))
    )
    im = ax.imshow(Corr, aspect="equal", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels([str(g) for g in groups])
    ax.set_title("Multitask output correlation across cancer groups")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    savefig(Path(outdir) / f"mt_output_group_corr_heatmap{suffix}.png")


def _collapse_feature_matrix(
    X_base,
    g_base_np,
    feat_idx,
    mode="group_shuffle",
    rng=None,
    condition_labels=None,
    strict_conditioning=False,
):
    X_base = np.asarray(X_base, float)
    g_base_np = np.asarray(g_base_np, int)
    idxs = np.atleast_1d(np.asarray(feat_idx, int))
    Xc = X_base.copy()
    if rng is None:
        rng = np.random.default_rng(0)
    mode = str(mode).strip().lower()

    # For treatment-conditioned FC, collapse/shuffle must occur only inside the
    # supplied condition cells, e.g. cancer + complete treatment profile.  This
    # prevents treatment flags or multi-treatment membership from being changed
    # implicitly while ranking non-treatment features.
    if condition_labels is not None:
        strata = np.asarray(condition_labels).astype(str)
        if strata.shape[0] != X_base.shape[0]:
            raise ValueError(
                "condition_labels length must match number of rows in X_base "
                f"({strata.shape[0]} != {X_base.shape[0]})"
            )
    else:
        strata = g_base_np

    if mode in ("group_shuffle", "within_group_shuffle", "conditional_shuffle"):
        unique_groups = np.unique(strata)
        for j in idxs:
            col = np.asarray(X_base[:, j], float)
            any_group_done = False
            for gi in unique_groups:
                mask = (strata == gi) & np.isfinite(col)
                ii = np.where(mask)[0]
                if ii.size >= 2:
                    Xc[ii, j] = col[ii][rng.permutation(ii.size)]
                    any_group_done = True
            # Legacy overall FC used a global fallback if no group had enough
            # observations.  Do NOT use that fallback for strict conditioned FC,
            # because it would break the exact treatment/cancer condition cells.
            if (not any_group_done) and (not bool(strict_conditioning)):
                ii = np.where(np.isfinite(col))[0]
                if ii.size >= 2:
                    Xc[ii, j] = col[ii][rng.permutation(ii.size)]
    elif mode in ("mean", "global_mean", "mean_collapse"):
        means = np.nanmean(X_base, axis=0)
        Xc[:, idxs] = means[idxs]
    else:
        raise ValueError(f"Unknown FC_COLLAPSE_MODE={mode}")
    return Xc


def overall_fc_df(
    model,
    X_ref_s,
    g_ref_np,
    feat_names,
    alpha_dt,
    dt_edges,
    H=FC_H_MONTHS,
    S=FC_S,
    R=FC_REPEATS,
    use_crn=FC_USE_CRN,
    method="wasserstein-1",
    condition_labels=None,
    skip_feature_idxs=None,
    fc_context_label="overall",
):
    """Distributional feature-collapsing importance for posterior risk.

    The primary score is an equal-group macro-average of patient-wise empirical
    W1 distances between baseline and collapsed posterior risk distributions.
    This avoids Breast-dominated pooled ranking when group sizes differ. The
    CSV also includes pooled W1, signed mean risk shift, absolute mean risk
    shift, repeat-level SEs, and a scale-normalized score for comparing horizons.
    """
    X_ref_s = np.asarray(X_ref_s, float)
    g_ref_np = np.asarray(g_ref_np, int)
    g_ref_t = torch.tensor(g_ref_np, device=device, dtype=torch.long)
    P = X_ref_s.shape[1]
    group_ids = np.asarray(sorted(np.unique(g_ref_np).tolist()), dtype=int)

    if condition_labels is not None:
        condition_labels = np.asarray(condition_labels).astype(str)
        if condition_labels.shape[0] != X_ref_s.shape[0]:
            raise ValueError(
                "condition_labels length must match X_ref_s rows "
                f"({condition_labels.shape[0]} != {X_ref_s.shape[0]})"
            )
        strict_conditioning = True
        n_condition_cells = int(pd.Index(condition_labels).nunique())
    else:
        strict_conditioning = False
        n_condition_cells = 0

    if skip_feature_idxs is None:
        skip_feature_idxs = set()
    else:
        skip_feature_idxs = set(
            int(j) for j in np.atleast_1d(np.asarray(skip_feature_idxs, dtype=int)).tolist()
        )

    score_reps = []
    score_pooled_reps = []
    dmean_reps = []
    abs_dmean_reps = []
    base_sd_reps = []
    base_iqr_reps = []
    nonfinite_reps = []
    group_score_reps = {int(gi): [] for gi in group_ids}

    for r in range(int(R)):
        seed_r = int(SEED) + 1009 * r + 7919
        if use_crn:
            np.random.seed(seed_r)
            torch.manual_seed(seed_r)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_r)
            np_state0 = np.random.get_state()
            torch_state0 = torch.get_rng_state()
            cuda_state0 = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
        Y0 = neglogS_samples_at_H_mt(
            model, X_ref_s, g_ref_t, float(H), alpha_dt, dt_edges, S=int(S)
        )
        meanY0 = np.nanmean(Y0, axis=0)
        base_sd0 = np.nanstd(Y0, axis=0, ddof=1)
        base_iqr0 = np.nanpercentile(Y0, 75, axis=0) - np.nanpercentile(Y0, 25, axis=0)

        scores_macro = np.zeros(P, dtype=float)
        scores_pooled = np.zeros(P, dtype=float)
        dmean = np.zeros(P, dtype=float)
        abs_dmean = np.zeros(P, dtype=float)
        nonfinite = np.zeros(P, dtype=float)
        group_scores_this = {int(gi): np.zeros(P, dtype=float) for gi in group_ids}

        for j in range(P):
            if j in skip_feature_idxs:
                continue
            if not np.isfinite(X_ref_s[:, j]).any():
                nonfinite[j] = 1.0
                continue
            if use_crn:
                np.random.set_state(np_state0)
                torch.set_rng_state(torch_state0)
                if cuda_state0 is not None:
                    torch.cuda.set_rng_state_all(cuda_state0)
            Xc = _collapse_feature_matrix(
                X_ref_s,
                g_ref_np,
                j,
                mode=FC_COLLAPSE_MODE,
                rng=np.random.default_rng(seed_r + 7919 + j),
                condition_labels=condition_labels,
                strict_conditioning=strict_conditioning,
            )
            try:
                Y1 = neglogS_samples_at_H_mt(
                    model, Xc, g_ref_t, float(H), alpha_dt, dt_edges, S=int(S)
                )
                meanY1 = np.nanmean(Y1, axis=0)
                if str(method).lower() in ("wasserstein-1", "wasserstein1", "wasserstein"):
                    w1_patient = _wasserstein1_empirical_per_row(Y0, Y1)
                    scores_pooled[j] = _nanmean_or_zero(w1_patient)
                    gvals = []
                    for gi in group_ids:
                        m = g_ref_np == int(gi)
                        gscore = _nanmean_or_zero(w1_patient[m]) if np.any(m) else 0.0
                        group_scores_this[int(gi)][j] = gscore
                        if np.any(m):
                            gvals.append(gscore)
                    scores_macro[j] = _nanmean_or_zero(gvals)
                    bad = (
                        (~np.isfinite(Y0)).sum()
                        + (~np.isfinite(Y1)).sum()
                        + (~np.isfinite(w1_patient)).sum()
                    )
                    nonfinite[j] = float(bad)
                    dy = meanY1 - meanY0
                    dmean[j] = _nanmean_or_zero(dy)
                    abs_dmean[j] = _nanmean_or_zero(np.abs(dy))
                else:
                    raise ValueError(f"Unknown FC method: {method}")
            except Exception as exc:
                # Do not drop the entire horizon because one collapsed feature is
                # numerically pathological.  Leave that feature at zero score and
                # record the failure in FC_nonfinite_count.
                nonfinite[j] = float(max(nonfinite[j], 1.0))
                print(
                    f"[POSTHOC][WARN] FC feature failed @ H={float(H):.1f}, "
                    f"feature={feat_names[j]}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        score_reps.append(scores_macro)
        score_pooled_reps.append(scores_pooled)
        dmean_reps.append(dmean)
        abs_dmean_reps.append(abs_dmean)
        base_sd_reps.append(base_sd0)
        base_iqr_reps.append(base_iqr0)
        nonfinite_reps.append(nonfinite)
        for gi in group_ids:
            group_score_reps[int(gi)].append(group_scores_this[int(gi)])

    score_mat = (
        np.vstack(score_reps) if len(score_reps) else np.zeros((1, P), dtype=float)
    )
    pooled_mat = (
        np.vstack(score_pooled_reps)
        if len(score_pooled_reps)
        else np.zeros((1, P), dtype=float)
    )
    dmean_mat = (
        np.vstack(dmean_reps) if len(dmean_reps) else np.zeros((1, P), dtype=float)
    )
    abs_dmean_mat = (
        np.vstack(abs_dmean_reps)
        if len(abs_dmean_reps)
        else np.zeros((1, P), dtype=float)
    )
    base_sd_mat = (
        np.vstack(base_sd_reps)
        if len(base_sd_reps)
        else np.zeros((1, X_ref_s.shape[0]), dtype=float)
    )
    base_iqr_mat = (
        np.vstack(base_iqr_reps)
        if len(base_iqr_reps)
        else np.zeros((1, X_ref_s.shape[0]), dtype=float)
    )
    nonfinite_mat = (
        np.vstack(nonfinite_reps)
        if len(nonfinite_reps)
        else np.zeros((1, P), dtype=float)
    )

    fc_score = np.nanmean(score_mat, axis=0)
    fc_pooled = np.nanmean(pooled_mat, axis=0)
    fc_dmean = np.nanmean(dmean_mat, axis=0)
    fc_abs_dmean = np.nanmean(abs_dmean_mat, axis=0)
    fc_score_sd = np.apply_along_axis(_nanstd_or_zero, 0, score_mat)
    fc_score_se = fc_score_sd / np.sqrt(max(int(R), 1))
    horizon_scale_sd = _nanmean_or_zero(np.nanmedian(base_sd_mat, axis=1))
    horizon_scale_iqr = _nanmean_or_zero(np.nanmedian(base_iqr_mat, axis=1))
    if horizon_scale_iqr <= 0.0:
        horizon_scale_iqr = 1.0
    if horizon_scale_sd <= 0.0:
        horizon_scale_sd = 1.0

    data = {
        "feature": list(feat_names),
        "FC_score": np.nan_to_num(fc_score, nan=0.0, posinf=0.0, neginf=0.0),
        "FC_score_group_macro_w1": np.nan_to_num(
            fc_score, nan=0.0, posinf=0.0, neginf=0.0
        ),
        "FC_score_pooled_patient_w1": np.nan_to_num(
            fc_pooled, nan=0.0, posinf=0.0, neginf=0.0
        ),
        "FC_score_se": np.nan_to_num(fc_score_se, nan=0.0, posinf=0.0, neginf=0.0),
        "FC_delta_mean_nlS": np.nan_to_num(fc_dmean, nan=0.0, posinf=0.0, neginf=0.0),
        "FC_abs_delta_mean_nlS": np.nan_to_num(
            fc_abs_dmean, nan=0.0, posinf=0.0, neginf=0.0
        ),
        "FC_score_std_by_baseline_iqr": np.nan_to_num(
            fc_score / horizon_scale_iqr, nan=0.0, posinf=0.0, neginf=0.0
        ),
        "FC_score_std_by_baseline_sd": np.nan_to_num(
            fc_score / horizon_scale_sd, nan=0.0, posinf=0.0, neginf=0.0
        ),
        "FC_nonfinite_count_mean": np.nanmean(nonfinite_mat, axis=0),
        "FC_method": str(method),
        "FC_collapse_mode": str(FC_COLLAPSE_MODE),
        "FC_S": int(S),
        "FC_use_crn": bool(use_crn),
        "FC_repeats": int(R),
        "FC_horizon_months": float(H),
        "FC_context_label": str(fc_context_label),
        "FC_conditioned": bool(condition_labels is not None),
        "FC_condition_cell_count": int(n_condition_cells),
        "FC_skipped_feature": [int(j) in skip_feature_idxs for j in range(P)],
        "FC_skip_reason": [
            "conditioning_treatment_flag_not_ranked" if int(j) in skip_feature_idxs else ""
            for j in range(P)
        ],
        "FC_primary_score_definition": "equal-group macro mean of patient-wise empirical W1 posterior risk distances",
    }
    for gi in group_ids:
        mat = (
            np.vstack(group_score_reps[int(gi)])
            if len(group_score_reps[int(gi)])
            else np.zeros((1, P), dtype=float)
        )
        data[f"FC_score_group{int(gi):02d}_w1"] = np.nan_to_num(
            np.nanmean(mat, axis=0), nan=0.0, posinf=0.0, neginf=0.0
        )

    df = pd.DataFrame(data)
    return df.sort_values("FC_score", ascending=False).reset_index(drop=True)


def _get_shared_private_latents(model, X_t, g_np):
    g_t = torch.tensor(np.asarray(g_np, int), device=device, dtype=torch.long)
    with torch.no_grad():
        h_shared = model.shared_latent_mt(X_t, g_t, sample=False).detach().cpu().numpy()
        h_private = np.zeros((X_t.size(0), model.private_dim), dtype=float)
        for gi in np.unique(np.asarray(g_np, int)):
            mask = np.asarray(g_np, int) == int(gi)
            if not np.any(mask):
                continue
            hp = _private_latent_mean_for_group(model, X_t[mask], int(gi))
            h_private[mask, :] = hp
    return h_shared, h_private


def branch_specific_fc_df(
    model, X_ref_s, g_ref_np, feat_names, collapse_mode=FC_COLLAPSE_MODE
):
    """Feature-collapsing sensitivity of latent branches.

    The exported primary branch scores are dimension-normalized RMS latent
    displacements.  Raw mean Euclidean norms are also exported, but the
    dimension-normalized columns are used by default so shared/private branch
    comparisons are not inflated solely by a larger latent dimensionality.
    """
    X_ref_s = np.asarray(X_ref_s, float)
    X_ref_t = torch.tensor(X_ref_s, device=device, dtype=torch.float32)
    g_ref_np = np.asarray(g_ref_np, int)
    h_sh_0, h_pr_0 = _get_shared_private_latents(model, X_ref_t, g_ref_np)

    def _mean_norm(diff):
        diff = np.asarray(diff, float)
        if diff.ndim != 2 or diff.shape[0] == 0:
            return np.nan
        return float(np.mean(np.linalg.norm(diff, axis=1)))

    def _mean_dim_normalized_rms(diff):
        diff = np.asarray(diff, float)
        if diff.ndim != 2 or diff.shape[0] == 0:
            return np.nan
        denom = max(int(diff.shape[1]), 1)
        return float(np.mean(np.sqrt(np.sum(diff * diff, axis=1) / denom)))

    rows = []
    for j, feat in enumerate(feat_names):
        Xc = _collapse_feature_matrix(
            X_ref_s,
            g_ref_np,
            j,
            mode=collapse_mode,
            rng=np.random.default_rng(SEED + 177 + j),
        )
        Xc_t = torch.tensor(Xc, device=device, dtype=torch.float32)
        h_sh_1, h_pr_1 = _get_shared_private_latents(model, Xc_t, g_ref_np)
        d_sh = h_sh_1 - h_sh_0
        d_pr = h_pr_1 - h_pr_0
        row = {
            "feature": str(feat),
            "shared_latent_fc": _mean_dim_normalized_rms(d_sh),
            "private_latent_fc": _mean_dim_normalized_rms(d_pr),
            "shared_latent_fc_raw_norm_mean": _mean_norm(d_sh),
            "private_latent_fc_raw_norm_mean": _mean_norm(d_pr),
            "branch_fc_metric": "dimension-normalized RMS latent displacement after within-group feature collapse",
        }
        for gi in np.unique(g_ref_np):
            m = g_ref_np == int(gi)
            row[f"private_latent_fc_group{int(gi):02d}"] = (
                _mean_dim_normalized_rms(d_pr[m]) if np.any(m) else np.nan
            )
            row[f"private_latent_fc_group{int(gi):02d}_raw_norm_mean"] = (
                _mean_norm(d_pr[m]) if np.any(m) else np.nan
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("shared_latent_fc", ascending=False)
        .reset_index(drop=True)
    )



def _safe_feature_family_for_branchattr(feature):
    try:
        return _feature_family_for_mechanism(str(feature))
    except Exception:
        return "unknown"


def compute_dgp_branch_feature_attribution(art, seed=None, cfg_tag=None):
    """Compute DGP-learned shared/private latent branch attribution for one trained split.

    This is the table needed for claims like "DGP-identified shared feature" versus
    "DGP-identified cancer-specific feature." It is NOT based on pooled scope and
    does NOT use act_shared/act_specific prefixes to assign the branch call.

    A feature is called shared when collapse of that feature changes the trained
    shared latent branch more than every private branch. A feature is called
    <cancer>_private when the largest private-branch displacement occurs in that
    cancer and is larger than the shared displacement.
    """
    if not ENABLE_REPEATED_SPLIT_DGP_BRANCH_ATTRIBUTION:
        return pd.DataFrame()

    model = art["model"]
    Xte_s = art["Xte_t"].detach().cpu().numpy()
    g_te = np.asarray(art["g_te"], dtype=int)
    feat_names = [str(x) for x in art["feat_names"]]
    groups = [str(x) for x in art["groups"]]

    dfb = branch_specific_fc_df(
        model,
        Xte_s,
        g_te,
        feat_names,
        collapse_mode=FC_COLLAPSE_MODE,
    )
    if dfb is None or len(dfb) == 0:
        return pd.DataFrame()

    dfb = dfb.copy()
    dfb.insert(0, "seed", int(seed) if seed is not None else np.nan)
    dfb.insert(1, "config_tag", str(cfg_tag) if cfg_tag is not None else "")
    dfb["feature_family"] = [
        _safe_feature_family_for_branchattr(f) for f in dfb["feature"].astype(str)
    ]

    # Map group00/group01/group02 to the actual cancer labels from this run.
    private_cols = []
    for gi, gname in enumerate(groups):
        old = f"private_latent_fc_group{int(gi):02d}"
        new = f"{str(gname)}_private_latent_fc"
        if old in dfb.columns:
            dfb[new] = pd.to_numeric(dfb[old], errors="coerce")
            private_cols.append(new)

    dfb["shared_latent_fc"] = pd.to_numeric(dfb.get("shared_latent_fc", np.nan), errors="coerce")
    if "private_latent_fc" in dfb.columns:
        dfb["private_latent_fc"] = pd.to_numeric(dfb["private_latent_fc"], errors="coerce")

    if private_cols:
        private_vals = dfb[private_cols].apply(pd.to_numeric, errors="coerce")
        dfb["max_private_latent_fc"] = private_vals.max(axis=1, skipna=True)
        dfb["max_private_group"] = private_vals.idxmax(axis=1).astype(str).str.replace("_private_latent_fc", "", regex=False)
    else:
        dfb["max_private_latent_fc"] = pd.to_numeric(dfb.get("private_latent_fc", np.nan), errors="coerce")
        dfb["max_private_group"] = ""

    shared = pd.to_numeric(dfb["shared_latent_fc"], errors="coerce").fillna(-np.inf)
    priv = pd.to_numeric(dfb["max_private_latent_fc"], errors="coerce").fillna(-np.inf)
    dfb["dgp_branch_call"] = np.where(
        shared >= priv,
        "shared",
        dfb["max_private_group"].astype(str) + "_private",
    )
    dfb["dgp_branch_call_basis"] = "trained DGP latent branch feature-collapse attribution"
    dfb["dgp_shared_minus_max_private_fc"] = shared.replace(-np.inf, np.nan) - priv.replace(-np.inf, np.nan)
    dfb["dgp_shared_to_max_private_fc_ratio"] = (
        shared.replace(-np.inf, np.nan) /
        np.maximum(priv.replace(-np.inf, np.nan), 1e-12)
    )

    # Make an unambiguous source column so downstream figures never mistake pooled scope for shared branch.
    dfb["shared_definition"] = "DGP shared latent branch attribution, not pooled scope"
    return dfb


def merge_dgp_branch_attribution_into_feature_tx(df_tx, df_branch):
    """Attach DGP branch calls to the feature-stratified treatment contrast CSV."""
    if df_tx is None or len(df_tx) == 0:
        return df_tx
    out = df_tx.copy()
    if df_branch is None or len(df_branch) == 0:
        out["dgp_branch_call"] = ""
        out["dgp_branch_call_basis"] = "missing branch attribution"
        return out

    keep_cols = [
        "seed", "config_tag", "feature", "feature_family",
        "dgp_branch_call", "dgp_branch_call_basis",
        "shared_latent_fc", "private_latent_fc",
        "max_private_latent_fc", "max_private_group",
        "dgp_shared_minus_max_private_fc", "dgp_shared_to_max_private_fc_ratio",
        "shared_definition",
    ]
    # Include named cancer-private branch scores when present.
    keep_cols += [c for c in df_branch.columns if c.endswith("_private_latent_fc")]
    keep_cols = [c for c in keep_cols if c in df_branch.columns]

    br = df_branch[keep_cols].drop_duplicates(subset=[c for c in ["seed", "config_tag", "feature"] if c in keep_cols]).copy()
    merge_cols = ["feature"]
    if "seed" in out.columns and "seed" in br.columns:
        merge_cols.append("seed")
    if "config_tag" in out.columns and "config_tag" in br.columns:
        merge_cols.append("config_tag")

    out = out.merge(br, on=merge_cols, how="left", suffixes=("", "_branchattr"))
    if "feature_family_branchattr" in out.columns and "feature_family" in out.columns:
        out = out.drop(columns=["feature_family_branchattr"])
    return out


def aggregate_dgp_branch_attribution_stability(df_branch):
    """Aggregate DGP branch calls across repeated split seeds."""
    if df_branch is None or len(df_branch) == 0:
        return pd.DataFrame()
    work = df_branch.copy()
    for c in ["shared_latent_fc", "private_latent_fc", "max_private_latent_fc", "dgp_shared_minus_max_private_fc", "dgp_shared_to_max_private_fc_ratio"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    rows = []
    group_cols = ["config_tag", "feature", "feature_family"]
    for keys, sub in work.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        calls = sub["dgp_branch_call"].astype(str)
        vc = calls.value_counts(dropna=False)
        dominant = str(vc.index[0]) if len(vc) else ""
        n = int(len(sub))
        row["n_splits_with_branch_attribution"] = n
        row["dominant_dgp_branch_call"] = dominant
        row["dominant_branch_fraction"] = float(vc.iloc[0] / max(n, 1)) if len(vc) else np.nan
        row["branch_calls_observed"] = ";".join([f"{k}:{int(v)}" for k, v in vc.items()])
        for c in ["shared_latent_fc", "private_latent_fc", "max_private_latent_fc", "dgp_shared_minus_max_private_fc", "dgp_shared_to_max_private_fc_ratio"]:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce")
                row[f"{c}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                row[f"{c}_sd"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["dominant_branch_fraction", "shared_latent_fc_mean"], ascending=[False, False]).reset_index(drop=True)


def save_dgp_branch_attribution_outputs(rows, cfg_dir, cfg_tag):
    """Save split-level and repeated-split DGP branch attribution tables."""
    cfg_dir = Path(cfg_dir)
    if rows is None or len(rows) == 0:
        # Branch attribution is an optional supplemental analysis and is
        # intentionally disabled for the NPJ run.  Do not create a misleading
        # "skipped" artifact that could be mistaken for a failed required block.
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(rows, axis=0, ignore_index=True) if isinstance(rows, list) else pd.DataFrame(rows)
    df.to_csv(cfg_dir / f"{cfg_tag}_dgp_branch_feature_attribution_by_split.csv", index=False)
    stability = aggregate_dgp_branch_attribution_stability(df)
    stability.to_csv(cfg_dir / f"{cfg_tag}_dgp_branch_feature_attribution_stability.csv", index=False)
    if len(stability):
        stability[stability["dominant_dgp_branch_call"].eq("shared")].to_csv(
            cfg_dir / f"{cfg_tag}_dgp_identified_shared_features.csv", index=False
        )
        stability[stability["dominant_dgp_branch_call"].astype(str).str.endswith("_private")].to_csv(
            cfg_dir / f"{cfg_tag}_dgp_identified_cancer_specific_features.csv", index=False
        )
    return df, stability


def _plot_topN_fc(df, score_col, title, outpath, topN=FC_TOPN, xlabel=None):
    df = df.copy()
    if score_col not in df.columns:
        return
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[np.isfinite(df[score_col].to_numpy(dtype=float))].copy()
    if len(df) == 0:
        return
    top = df.sort_values(score_col, ascending=False).head(int(topN)).iloc[::-1].copy()
    fig_h = max(4.5, 0.42 * len(top) + 1.5)
    plt.figure(figsize=(8.0, fig_h))
    y = np.arange(len(top))
    vals = pd.to_numeric(top[score_col], errors="coerce").to_numpy(dtype=float)
    plt.hlines(y, 0.0, vals, color="#D6DEE6", lw=1.8, zorder=1)
    plt.scatter(
        vals, y, s=88, color="#4E79A7", edgecolors="white", linewidths=0.9, zorder=2
    )
    plt.yticks(y, top["feature"].astype(str).tolist(), fontsize=10)
    plt.xlabel(xlabel or score_col)
    plt.title(title, loc="left")
    plt.grid(True, axis="x", alpha=0.16)
    vmax = np.nanmax(vals) if np.any(np.isfinite(vals)) else 0.0
    plt.xlim(0.0, max(vmax * 1.08, 1e-8))
    savefig(outpath)


def _prep_fc_plot_df(df_plot, branch_col, delta_col, sort_col=None, topN=FC_TOPN):
    df_plot = df_plot.copy()
    if len(df_plot) == 0:
        return df_plot
    for c in ["risk_score", branch_col, delta_col]:
        if c not in df_plot.columns:
            df_plot[c] = np.nan
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    if sort_col is None:
        sort_col = branch_col
    if sort_col not in df_plot.columns:
        df_plot[sort_col] = np.nan
    df_plot[sort_col] = pd.to_numeric(df_plot[sort_col], errors="coerce")
    keep = np.isfinite(df_plot["risk_score"].to_numpy(dtype=float)) | np.isfinite(
        df_plot[branch_col].to_numpy(dtype=float)
    )
    df_plot = df_plot.loc[keep].copy()
    if len(df_plot) == 0:
        return df_plot
    return (
        df_plot.sort_values(sort_col, ascending=False).head(int(topN)).iloc[::-1].copy()
    )


def _style_fc_axis(ax):
    ax.grid(True, axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _finite_values_for_xlim(vals):
    """Flatten one or more numeric arrays/Series into finite values for axis scaling."""
    if isinstance(vals, (list, tuple)):
        pieces = []
        for v in vals:
            a = np.asarray(v, float).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                pieces.append(a)
        return np.concatenate(pieces) if pieces else np.asarray([], dtype=float)
    a = np.asarray(vals, float).ravel()
    return a[np.isfinite(a)]


def _positive_xlim_from_values(vals, pad=0.08):
    vals = _finite_values_for_xlim(vals)
    vmax = float(np.max(vals)) if vals.size else 0.0
    return (0.0, max(vmax * (1.0 + float(pad)), 1e-8))


def _signed_xlim_from_values(vals, pad=0.10):
    vals = _finite_values_for_xlim(vals)
    vmax = float(np.max(np.abs(vals))) if vals.size else 0.0
    vmax = max(vmax * (1.0 + float(pad)), 1e-8)
    return (-vmax, vmax)


def _set_positive_xlim(ax, vals, pad=0.08, xlim=None):
    if xlim is None:
        xlim = _positive_xlim_from_values(vals, pad=pad)
    ax.set_xlim(float(xlim[0]), float(xlim[1]))


def _set_signed_xlim(ax, vals, pad=0.10, xlim=None):
    if xlim is None:
        xlim = _signed_xlim_from_values(vals, pad=pad)
    ax.set_xlim(float(xlim[0]), float(xlim[1]))


def _plot_fc_triptych(
    df_plot,
    branch_col,
    delta_col,
    title,
    branch_xlabel,
    outpath,
    branch_color="#355C7D",
    topN=FC_TOPN,
    sort_col=None,
    risk_xlim=None,
    branch_xlim=None,
    delta_xlim=None,
):
    df_plot = _prep_fc_plot_df(
        df_plot,
        branch_col=branch_col,
        delta_col=delta_col,
        sort_col=sort_col,
        topN=topN,
    )
    if len(df_plot) == 0:
        return
    vals_r = np.asarray(df_plot["risk_score"], float)
    vals_b = np.asarray(df_plot[branch_col], float)
    vals_d = np.asarray(df_plot[delta_col], float)
    labs = df_plot["feature"].astype(str).tolist()
    y = np.arange(len(df_plot))
    fig_h = max(5.0, 0.44 * len(df_plot) + 1.8)
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(15.2, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1.2, 1.05]},
    )
    axs[0].hlines(y, 0.0, vals_r, color="#D6DEE6", lw=1.8, zorder=1)
    axs[0].scatter(
        vals_r, y, s=90, color="#4E79A7", edgecolors="white", linewidths=0.9, zorder=2
    )
    axs[0].set_xlabel("Overall FC (Wasserstein)")
    axs[0].set_title(title, fontsize=13, loc="left")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(labs, fontsize=10)
    _style_fc_axis(axs[0])
    _set_positive_xlim(axs[0], vals_r, xlim=risk_xlim)
    colors_b = np.where(np.isfinite(vals_b) & (vals_b >= 0), branch_color, "#9EAAB5")
    axs[1].hlines(y, 0.0, vals_b, color="#D6DEE6", lw=1.8, zorder=1)
    axs[1].scatter(
        vals_b, y, s=90, color=colors_b, edgecolors="white", linewidths=0.9, zorder=2
    )
    axs[1].set_xlabel(branch_xlabel)
    _style_fc_axis(axs[1])
    _set_positive_xlim(axs[1], vals_b, xlim=branch_xlim)
    pos = np.isfinite(vals_d) & (vals_d >= 0)
    neg = np.isfinite(vals_d) & (vals_d < 0)
    axs[2].axvline(0.0, color="#9099A1", lw=1.0, alpha=0.7)
    if np.any(pos):
        axs[2].hlines(y[pos], 0.0, vals_d[pos], color="#F6C7B6", lw=1.8, zorder=1)
        axs[2].scatter(
            vals_d[pos],
            y[pos],
            s=90,
            color="#D55E00",
            edgecolors="white",
            linewidths=0.9,
            zorder=2,
            label="risk up",
        )
    if np.any(neg):
        axs[2].hlines(y[neg], 0.0, vals_d[neg], color="#C7DCEF", lw=1.8, zorder=1)
        axs[2].scatter(
            vals_d[neg],
            y[neg],
            s=90,
            color="#2C7FB8",
            edgecolors="white",
            linewidths=0.9,
            zorder=2,
            label="risk down",
        )
    axs[2].set_xlabel("Δ mean predicted risk after collapse")
    _style_fc_axis(axs[2])
    _set_signed_xlim(axs[2], vals_d, xlim=delta_xlim)
    if np.any(pos) or np.any(neg):
        axs[2].legend(
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=9,
        )
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    savefig(outpath)


def _feature_scope_for_mechanism(feature_name):
    """Return whether a design feature is all-cancer shared, cancer-specific, or covariate.

    This is intentionally feature-name based. The pooled FC scope means the test
    patients were pooled; it does not mean the feature was shared across cancers.
    Shared-across-cancers FC should therefore be filtered on this function, not
    on scope == "pooled".
    """
    s = str(feature_name)
    if s.startswith(SHARED_ACROSS_CANCERS_FEATURE_PREFIXES):
        return "shared_across_cancers"
    if s.startswith(CANCER_SPECIFIC_FEATURE_PREFIXES):
        return "cancer_specific"
    if s.startswith(("mut_shared::", "mut_specific_")):
        return "shared_across_cancers" if s.startswith("mut_shared::") else "cancer_specific"
    if s.startswith("mut::"):
        # Mutation-profile columns currently combine shared-selected and cancer-specific
        # genes under one mut:: prefix in the covariate block, so do not label them
        # as shared unless preprocessing adds a distinct mut_shared:: prefix later.
        return "mutation_covariate_unlabeled_sharedness"
    if s.startswith("tmb::"):
        return "tmb_covariate"
    if s.startswith(("clincont::", "clinbin::")) or s in TREATMENT_FLAG_COLS or s == "TX_EXPLICIT_NO_TREATMENT":
        return "clinical_covariate"
    if re.match(r"^(BRCA|LUAD|COAD)_PC\d+$", s) or re.match(r"^PC\d+$", s):
        return "gene_pc"
    return "other"


def _is_shared_across_cancers_feature(feature_name):
    return _feature_scope_for_mechanism(feature_name) == "shared_across_cancers"


def _feature_family_for_mechanism(feature_name):
    s = str(feature_name)
    if s.startswith(("act_shared::", "act_specific_")) or s.startswith(
        ("pathway::", "pathway_")
    ):
        return "pathway"
    if s.startswith(("mut::", "mut_shared::", "mut_specific_")):
        return "mutation"
    if s.startswith("tmb::"):
        return "tmb"
    if s.startswith("clincont::"):
        return "clinical_continuous"
    if s.startswith("clinbin::"):
        base = _base_treatment_col(s)
        return (
            "treatment"
            if base in TREATMENT_FLAG_COLS or base == "TX_EXPLICIT_NO_TREATMENT"
            else "clinical_binary"
        )
    if re.match(r"^(BRCA|LUAD|COAD)_PC\d+$", s) or re.match(r"^PC\d+$", s):
        return "gene_pc"
    if s in TREATMENT_FLAG_COLS or s == "TX_EXPLICIT_NO_TREATMENT":
        return "treatment"
    if s.startswith(("TX_", "STAGE_", "GRADE_", "AJCC_T_")):
        return "clinical_binary"
    if s in ("AGE", "AGE_Z", "AGE_AT_DIAGNOSIS"):
        return "clinical_continuous"
    return "other"


def _mechanism_feature_mask(feat_names, families=None):
    if families is None:
        families = TREATMENT_CONTEXT_FC_FAMILIES
    allowed = {str(x) for x in families}
    return np.asarray(
        [_feature_family_for_mechanism(f) in allowed for f in feat_names], dtype=bool
    )


def _metadata_treatment_flags(meta_df):
    if meta_df is None or len(meta_df) == 0:
        return []
    return [c for c in TREATMENT_FLAG_COLS if c in meta_df.columns]


def _no_recognized_treatment_mask(meta_df, tx_cols):
    if meta_df is None or len(meta_df) == 0:
        return np.zeros(0, dtype=bool)
    work = meta_df.reset_index(drop=True)
    if "TX_EXPLICIT_NO_TREATMENT" in work.columns:
        vals = (
            pd.to_numeric(work["TX_EXPLICIT_NO_TREATMENT"], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
        return vals > 0.5
    if len(tx_cols) == 0:
        return np.zeros(len(work), dtype=bool)
    A = work[tx_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    return np.sum(A > 0.5, axis=1) == 0


def _make_treatment_context_masks(meta_df):
    """Return non-mutually-exclusive treatment-context masks for test patients."""
    if meta_df is None or len(meta_df) == 0:
        return []
    work = meta_df.reset_index(drop=True).copy()
    tx_cols = _metadata_treatment_flags(work)
    contexts = []
    no_mask = _no_recognized_treatment_mask(work, tx_cols)
    contexts.append(
        {
            "context_id": "NO_RECOGNIZED_TREATMENT",
            "context_label": "No recognized treatment",
            "context_type": "reference_no_recognized_treatment",
            "treatment_flag": "TX_EXPLICIT_NO_TREATMENT",
            "mask": np.asarray(no_mask, bool),
            "nonexclusive_treatment_context": False,
        }
    )
    for col in tx_cols:
        vals = pd.to_numeric(work[col], errors="coerce").fillna(0.0).to_numpy(float)
        contexts.append(
            {
                "context_id": str(col),
                "context_label": _treatment_label(col),
                "context_type": "treatment_positive",
                "treatment_flag": str(col),
                "mask": vals > 0.5,
                "nonexclusive_treatment_context": True,
            }
        )
    return contexts


def _shorten_label_for_plot(label, max_chars=84):
    s = str(label)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= int(max_chars) else s[: int(max_chars) - 1] + "…"


def _plot_treatment_context_fc_lollipop(
    df_plot, title, outpath, score_col="FC_score", topN=TREATMENT_CONTEXT_FC_TOPN
):
    df_plot = df_plot.copy()
    if score_col not in df_plot.columns or len(df_plot) == 0:
        return
    df_plot[score_col] = pd.to_numeric(df_plot[score_col], errors="coerce")
    df_plot = df_plot[np.isfinite(df_plot[score_col].to_numpy(dtype=float))].copy()
    if len(df_plot) == 0:
        return
    top = (
        df_plot.sort_values(score_col, ascending=False)
        .head(int(topN))
        .iloc[::-1]
        .copy()
    )
    vals = top[score_col].to_numpy(dtype=float)
    labels = [_shorten_label_for_plot(x) for x in top["feature"].astype(str).tolist()]
    y = np.arange(len(top))
    fig_h = max(5.2, 0.42 * len(top) + 1.7)
    fig, ax = plt.subplots(figsize=(10.6, fig_h))
    ax.hlines(y, 0.0, vals, color="#D6DEE6", lw=2.2, zorder=1)
    fam = (
        top.get("feature_family", pd.Series(["other"] * len(top)))
        .astype(str)
        .to_numpy()
    )
    color_map = {
        "pathway": "#4E79A7",
        "mutation": "#59A14F",
        "tmb": "#B07AA1",
        "clinical_binary": "#F28E2B",
        "clinical_continuous": "#E15759",
        "treatment": "#76B7B2",
        "gene_pc": "#EDC948",
        "other": "#7F8C8D",
    }
    colors = [color_map.get(str(x), "#7F8C8D") for x in fam]
    ax.scatter(
        vals, y, s=100, color=colors, edgecolors="white", linewidths=1.0, zorder=2
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel(
        "Posterior feature-collapse effect on early-failure risk (Wasserstein)"
    )
    ax.set_title(title, loc="left", fontsize=13)
    ax.grid(True, axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    vmax = np.nanmax(vals) if np.any(np.isfinite(vals)) else 0.0
    ax.set_xlim(0.0, max(vmax * 1.08, 1e-8))
    handles = []
    for fam_name, col in color_map.items():
        if np.any(fam == fam_name):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=fam_name.replace("_", " "),
                    markerfacecolor=col,
                    markersize=8,
                )
            )
    if handles:
        ax.legend(
            handles=handles,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        fig.tight_layout()
    savefig(outpath)

def _plot_treatment_vs_reference_delta(
    df_delta, title, outpath, topN=TREATMENT_CONTEXT_FC_TOPN
):
    df_delta = df_delta.copy()
    if "delta_FC_score_vs_reference" not in df_delta.columns or len(df_delta) == 0:
        return
    df_delta["delta_FC_score_vs_reference"] = pd.to_numeric(
        df_delta["delta_FC_score_vs_reference"], errors="coerce"
    )
    df_delta = df_delta[
        np.isfinite(df_delta["delta_FC_score_vs_reference"].to_numpy(dtype=float))
    ].copy()
    if len(df_delta) == 0:
        return
    df_delta = df_delta.reindex(
        df_delta["delta_FC_score_vs_reference"]
        .abs()
        .sort_values(ascending=False)
        .head(int(topN))
        .index
    ).copy()
    df_delta = df_delta.sort_values("delta_FC_score_vs_reference").copy()
    vals = df_delta["delta_FC_score_vs_reference"].to_numpy(dtype=float)
    labels = [
        _shorten_label_for_plot(x) for x in df_delta["feature"].astype(str).tolist()
    ]
    y = np.arange(len(df_delta))
    colors = np.where(vals >= 0, "#D55E00", "#2C7FB8")
    fig_h = max(5.2, 0.42 * len(df_delta) + 1.7)
    fig, ax = plt.subplots(figsize=(10.4, fig_h))
    ax.axvline(0.0, color="#9099A1", lw=1.0, alpha=0.85)
    ax.hlines(y, 0.0, vals, color="#D6DEE6", lw=2.0, zorder=1)
    ax.scatter(
        vals, y, s=96, color=colors, edgecolors="white", linewidths=1.0, zorder=2
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Treatment-positive FC minus no-recognized-treatment FC")
    ax.set_title(title, loc="left", fontsize=13)
    ax.grid(True, axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    xmax = np.nanmax(np.abs(vals)) if np.any(np.isfinite(vals)) else 0.0
    xmax = max(xmax * 1.10, 1e-8)
    ax.set_xlim(-xmax, xmax)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="larger in treatment-positive",
            markerfacecolor="#D55E00",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="larger in reference",
            markerfacecolor="#2C7FB8",
            markersize=8,
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=[0, 0, 0.83, 1])
    savefig(outpath)


def _plot_treatment_context_heatmap(df_all, H, outpath, topN=TREATMENT_CONTEXT_FC_TOPN):
    if df_all is None or len(df_all) == 0:
        return
    dfh = df_all[
        np.isclose(pd.to_numeric(df_all["H"], errors="coerce"), float(H))
    ].copy()
    if len(dfh) == 0:
        return
    dfh = dfh[dfh["feature_family"].isin([str(x) for x in TREATMENT_CONTEXT_FC_FAMILIES])].copy()
    if len(dfh) == 0:
        return
    top_features = []
    for _, sub in dfh.groupby("context_id", dropna=False):
        top_features.extend(
            sub.sort_values("FC_score", ascending=False)
            .head(int(topN))["feature"]
            .astype(str)
            .tolist()
        )
    top_features = list(dict.fromkeys(top_features))[: max(int(topN) * 3, int(topN))]
    contexts = (
        dfh[["context_id", "context_label", "context_type"]].drop_duplicates().copy()
    )
    contexts["order"] = (
        contexts["context_type"]
        .map({"reference_no_recognized_treatment": 0, "treatment_positive": 1})
        .fillna(2)
    )
    contexts = contexts.sort_values(["order", "context_label"]).reset_index(drop=True)
    mat = np.full((len(top_features), len(contexts)), np.nan, dtype=float)
    for i, feat in enumerate(top_features):
        for j, rr in contexts.iterrows():
            vals = dfh[
                (dfh["feature"].astype(str) == str(feat))
                & (dfh["context_id"].astype(str) == str(rr["context_id"]))
            ]["FC_score"]
            if len(vals):
                mat[i, j] = float(pd.to_numeric(vals.iloc[0], errors="coerce"))
    if mat.size == 0:
        return
    col_max = np.nanmax(mat, axis=0)
    col_max = np.where((~np.isfinite(col_max)) | (col_max <= 0.0), 1.0, col_max)
    mat_scaled = mat / col_max[None, :]
    fig_w = max(8.2, 1.05 * len(contexts) + 4.4)
    fig_h = max(6.0, 0.34 * len(top_features) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat_scaled, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(contexts)))
    ax.set_xticklabels(
        contexts["context_label"].astype(str).tolist(), rotation=35, ha="right"
    )
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels(
        [_shorten_label_for_plot(f, 72) for f in top_features], fontsize=8.5
    )
    ax.set_title(
        f"Treatment-conditioned candidate resistance features (H={int(H)} months)",
        loc="left",
        fontsize=13,
    )
    ax.set_xlabel(
        "Treatment context; patients with multiple treatments appear in each positive context"
    )
    ax.set_ylabel("Top mechanism-focused features")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Within-context scaled FC score")
    fig.tight_layout()
    savefig(outpath)


def _resolve_treatment_conditioning_feature_indices(feat_names):
    """Indices of design-matrix treatment flags used as conditioning covariates."""
    out = []
    for j, nm in enumerate([str(x) for x in feat_names]):
        base = _base_treatment_col(nm)
        if base in TREATMENT_FLAG_COLS or base == "TX_EXPLICIT_NO_TREATMENT":
            out.append(int(j))
    return out


def _treatment_profile_from_metadata_or_design(meta_df, X_s, feat_names):
    """Return a binary [N, Tx] treatment matrix and treatment-column labels.

    Metadata columns are preferred because they are unstandardized.  If metadata
    is unavailable, standardized binary design columns are thresholded at zero;
    for 0/1 features standardized from the training split, zero remains between
    the two encoded values unless the column is constant.
    """
    X_s = np.asarray(X_s, float)
    n = int(X_s.shape[0])
    work = meta_df.reset_index(drop=True).copy() if meta_df is not None else None
    cols = []
    mats = []
    if work is not None and len(work) == n:
        for col in TREATMENT_FLAG_COLS:
            if col in work.columns:
                vals = (
                    pd.to_numeric(work[col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(float)
                )
                cols.append(str(col))
                mats.append((vals > 0.5).astype(int))
    if len(mats) == 0:
        feat_names = [str(x) for x in feat_names]
        f_to_idx = {f: i for i, f in enumerate(feat_names)}
        for base, hit in _resolve_treatment_feature_columns(feat_names):
            if hit in f_to_idx:
                vals = np.asarray(X_s[:, f_to_idx[hit]], float)
                cols.append(str(base))
                mats.append((vals > 0.0).astype(int))
    if len(mats) == 0:
        return np.zeros((n, 0), dtype=int), []
    A = np.vstack(mats).T.astype(int)
    return A, cols




def _treatment_profile_scope_projection(A, tx_cols, scope="pooled", group=None):
    """Project treatment flags to the biologically comparable profile definition.

    Hormone therapy is retained only for Breast-specific profiles.  For pooled /
    shared-across-cancers summaries and for Lung/Colon-specific profiles, hormone
    is excluded from the exact treatment-profile definition so that treatment
    cells are comparable across cancers.
    """
    A = np.asarray(A, dtype=int)
    tx_cols = [str(c) for c in tx_cols]
    scope_s = str(scope).lower()
    group_s = str(group).lower() if group is not None else ""
    allow_hormone = (group_s == "breast") and ("pooled" not in scope_s) and ("shared" not in scope_s) and ("all" not in scope_s)
    keep = []
    for j, c in enumerate(tx_cols):
        base = _base_treatment_col(c)
        if base == "TX_HORMONE" and not allow_hormone:
            continue
        keep.append(int(j))
    if len(keep) == 0:
        return np.zeros((A.shape[0], 0), dtype=int), []
    return A[:, keep].astype(int), [tx_cols[j] for j in keep]


def _treatment_profile_labels_for_scope(A, tx_cols, scope="pooled", group=None):
    A_scope, tx_cols_scope = _treatment_profile_scope_projection(A, tx_cols, scope=scope, group=group)
    return _make_treatment_profile_labels(A_scope, tx_cols_scope), tx_cols_scope

def _make_treatment_profile_labels(A, tx_cols):
    A = np.asarray(A, int)
    if A.ndim != 2:
        raise ValueError(f"Treatment profile matrix must be 2D; got {A.shape}.")
    if A.shape[1] == 0:
        return np.asarray(["TX=none_observed"] * A.shape[0], dtype=object)
    labels = []
    short_cols = [_treatment_label(c) for c in tx_cols]
    for row in A:
        active = [short_cols[j] for j, v in enumerate(row) if int(v) == 1]
        if len(active) == 0:
            labels.append("TX=None")
        else:
            labels.append("TX=" + "+".join(active))
    return np.asarray(labels, dtype=object)


def _treatment_profile_active_count(label):
    """Number of active treatments in an exact treatment-profile label."""
    lab = str(label)
    low = lab.lower()
    if lab in {"", "ALL", "nan", "None"} or "tx=none" in low or "none_observed" in low:
        return 0
    if lab.startswith("TX="):
        body = lab[3:]
    else:
        body = lab
    active = [x.strip() for x in body.split("+") if x.strip()]
    return int(len(active))


def _is_multi_treatment_profile(label):
    return _treatment_profile_active_count(label) >= 2


def _make_treatment_profile_count_table(patient_df):
    """Non-overlapping exact treatment-profile counts by cancer group plus pooled total."""
    if patient_df is None or len(patient_df) == 0:
        return pd.DataFrame()
    d = patient_df.copy()
    d["treatment_profile"] = d["treatment_profile"].astype(str)
    d["group"] = d["group"].astype(str)
    d["treatment_active_count"] = d["treatment_profile"].map(_treatment_profile_active_count).astype(int)
    d["is_multi_treatment_profile"] = d["treatment_active_count"] >= 2
    rows = []
    for group_key, sub in d.groupby("group", dropna=False):
        for tx, ss in sub.groupby("treatment_profile", dropna=False):
            rows.append({
                "scope": "cancer",
                "group": str(group_key),
                "treatment_profile": str(tx),
                "treatment_active_count": int(_treatment_profile_active_count(tx)),
                "is_multi_treatment_profile": bool(_is_multi_treatment_profile(tx)),
                "n_patients": int(len(ss)),
                "PFI_events_total": int(pd.to_numeric(ss["PFI_EVENT"], errors="coerce").fillna(0).astype(int).sum()),
            })
    for tx, ss in d.groupby("treatment_profile", dropna=False):
        rows.append({
            "scope": "pooled",
            "group": "pooled",
            "treatment_profile": str(tx),
            "treatment_active_count": int(_treatment_profile_active_count(tx)),
            "is_multi_treatment_profile": bool(_is_multi_treatment_profile(tx)),
            "n_patients": int(len(ss)),
            "PFI_events_total": int(pd.to_numeric(ss["PFI_EVENT"], errors="coerce").fillna(0).astype(int).sum()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["scope", "group", "is_multi_treatment_profile", "n_patients", "treatment_profile"], ascending=[True, True, True, False, True]).reset_index(drop=True)
    return out


def _exact_profile_fc_min_support(label):
    if _is_multi_treatment_profile(label):
        return int(TREATMENT_EXACT_PROFILE_FC_MULTI_MIN_N), int(TREATMENT_EXACT_PROFILE_FC_MULTI_MIN_EVENTS)
    return int(TREATMENT_EXACT_PROFILE_FC_MIN_N), int(TREATMENT_EXACT_PROFILE_FC_MIN_EVENTS)


def _exact_profile_landscape_min_support(label):
    if _is_multi_treatment_profile(label):
        return int(RESISTANCE_PSEUDOTRAJECTORY_MULTI_TREATMENT_MIN_N), int(RESISTANCE_PSEUDOTRAJECTORY_MULTI_TREATMENT_MIN_EVENTS)
    return int(RESISTANCE_PSEUDOTRAJECTORY_MIN_N), int(RESISTANCE_PSEUDOTRAJECTORY_MIN_EVENTS)


def _make_treatment_adjusted_condition_labels(g_np, groups, treatment_profile_labels):
    g_np = np.asarray(g_np, int).reshape(-1)
    treatment_profile_labels = np.asarray(treatment_profile_labels).astype(str).reshape(-1)
    if treatment_profile_labels.shape[0] != g_np.shape[0]:
        raise ValueError(
            f"treatment_profile_labels length {treatment_profile_labels.shape[0]} does not match group length {g_np.shape[0]}."
        )
    out = []
    for gi, tx_lab in zip(g_np, treatment_profile_labels):
        if TREATMENT_ADJUSTED_FC_STRATIFY_BY_CANCER_GROUP:
            g_lab = str(groups[int(gi)]) if int(gi) < len(groups) else f"group{int(gi)}"
            out.append(f"GROUP={g_lab}|{tx_lab}")
        else:
            out.append(str(tx_lab))
    return np.asarray(out, dtype=object)


def _treatment_adjusted_condition_counts(condition_labels):
    vals, counts = np.unique(np.asarray(condition_labels).astype(str), return_counts=True)
    return pd.DataFrame({"condition_cell": vals, "n": counts}).sort_values(
        ["n", "condition_cell"], ascending=[False, True]
    )


def _add_fc_feature_family_columns(df, feat_names=None):
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df["feature_family"] = [
        _feature_family_for_mechanism(f) for f in df["feature"].astype(str)
    ]
    df["feature_scope"] = [
        _feature_scope_for_mechanism(f) for f in df["feature"].astype(str)
    ]
    df["is_shared_across_cancers_feature"] = df["feature_scope"].astype(str).eq(
        "shared_across_cancers"
    )
    df["is_cancer_specific_feature"] = df["feature_scope"].astype(str).eq(
        "cancer_specific"
    )
    df["is_clinical_feature"] = df["feature_family"].isin(
        ["clinical_binary", "clinical_continuous", "treatment"]
    )
    return df


def _plot_treatment_adjusted_fc(df_plot, title, outpath, topN=TREATMENT_ADJUSTED_FC_TOPN):
    if df_plot is None or len(df_plot) == 0:
        return
    df_plot = df_plot.copy()
    if "FC_conditioning_feature_not_ranked" in df_plot.columns:
        mask_conditioning = df_plot["FC_conditioning_feature_not_ranked"].astype(bool).to_numpy()
        df_plot = df_plot[~mask_conditioning].copy()
    _plot_treatment_context_fc_lollipop(
        df_plot,
        title,
        outpath,
        score_col="FC_score",
        topN=topN,
    )


# LEGACY/INACTIVE: retained only so unrelated historical posthoc helpers remain
# import-compatible. The NPJ August controller never calls this routine. The
# sole executed WFC is run_corrected_wfc_partition from the corrected module.
def run_treatment_adjusted_pooled_and_group_fc_analysis(art, outdir, make_plots=True):
    """Pooled and cancer-specific FC while conditioning on treatment profile.

    Each ranked feature is permuted only inside cells defined by cancer group and
    the complete multi-treatment vector.  This preserves simultaneous treatments
    such as chemo+radiation rather than forcing patients into mutually exclusive
    treatment categories.  Treatment-flag columns are used as conditioning
    covariates and are not perturbed; all other clinical variables remain in the
    FC ranking.
    """
    if not ENABLE_TREATMENT_ADJUSTED_FC:
        return None
    outdir = Path(outdir) / "treatment_adjusted_feature_collapse"
    outdir.mkdir(parents=True, exist_ok=True)

    meta_te = art.get("meta_te")
    Xte_s = art["Xte_t"].detach().cpu().numpy()
    g_te = np.asarray(art["g_te"], int)
    t_te = np.asarray(art["t_te"], float)
    e_te = np.asarray(art["e_te"], int)
    groups = [str(x) for x in art["groups"]]
    feat_names = [str(x) for x in art["feat_names"]]

    if meta_te is not None:
        meta_te = meta_te.reset_index(drop=True).copy()
        if len(meta_te) != Xte_s.shape[0]:
            pd.DataFrame(
                {
                    "message": [
                        f"meta_te length {len(meta_te)} does not match Xte rows {Xte_s.shape[0]}; skipped treatment-conditioned full-profile FC."
                    ]
                }
            ).to_csv(outdir / "treatment_adjusted_fc_skipped.csv", index=False)
            return None

    A_tx, tx_cols = _treatment_profile_from_metadata_or_design(meta_te, Xte_s, feat_names)
    # Pooled/shared exact-profile definitions exclude hormone to keep treatment
    # cells comparable across cancers. Breast-specific profiles below retain hormone.
    tx_profile_labels, tx_cols = _treatment_profile_labels_for_scope(
        A_tx, tx_cols, scope="pooled", group="All_cancers_combined"
    )
    condition_labels = _make_treatment_adjusted_condition_labels(
        g_te, groups, tx_profile_labels
    )
    tx_feature_idxs = _resolve_treatment_conditioning_feature_indices(feat_names)
    skip_feature_idxs = (
        tx_feature_idxs
        if TREATMENT_ADJUSTED_FC_EXCLUDE_TREATMENT_FLAGS_FROM_RANKING
        else []
    )

    readme = (
        "Treatment-adjusted feature-collapse analysis.\n"
        "Each feature is permuted only within cells defined by cancer group and the complete multi-treatment profile.\n"
        "Patients with multiple treatments keep their full treatment vector, e.g. chemo+radiation, and contribute to that exact conditioning cell.\n"
        "Treatment-flag columns are conditioning covariates and are not perturbed in the primary ranking; other clinical variables remain rankable.\n"
        "The scores are posterior predictive feature-collapse sensitivities, not causal treatment effects.\n"
    )
    (outdir / "README_treatment_adjusted_feature_collapse.txt").write_text(
        readme, encoding="utf-8"
    )

    df_cells = _treatment_adjusted_condition_counts(condition_labels)
    df_cells.to_csv(outdir / "treatment_adjusted_condition_cells.csv", index=False)

    tx_count_df = pd.DataFrame({
        "group_idx": g_te,
        "group": [groups[int(g)] if 0 <= int(g) < len(groups) else f"group{int(g)}" for g in g_te],
        "PFI_EVENT": e_te,
        "treatment_profile": tx_profile_labels.astype(str),
    })
    _make_treatment_profile_count_table(tx_count_df).to_csv(
        outdir / "treatment_profile_counts_by_cancer.csv", index=False
    )

    pd.DataFrame(
        {
            "treatment_column_used_for_conditioning": tx_cols,
            "label": [_treatment_label(c) for c in tx_cols],
        }
    ).to_csv(outdir / "treatment_conditioning_columns.csv", index=False)
    pd.DataFrame(
        {
            "feature_index": tx_feature_idxs,
            "feature": [feat_names[j] for j in tx_feature_idxs],
            "role": "conditioning_treatment_flag_not_ranked",
        }
    ).to_csv(outdir / "treatment_conditioning_feature_columns.csv", index=False)

    context_rows = []
    all_rows = []
    for H in [float(h) for h in TREATMENT_ADJUSTED_FC_H_LIST]:
        df_pooled = overall_fc_df(
            art["model"],
            Xte_s,
            g_te,
            feat_names,
            art["alpha_dt"],
            art["dt_edges"],
            H=H,
            S=FC_S,
            R=FC_REPEATS,
            use_crn=FC_USE_CRN,
            condition_labels=condition_labels,
            skip_feature_idxs=skip_feature_idxs,
            fc_context_label="pooled_test_treatment_adjusted",
        )
        df_pooled = _add_fc_feature_family_columns(df_pooled)
        df_pooled.insert(0, "H", float(H))
        df_pooled.insert(1, "scope", "pooled")
        df_pooled.insert(2, "group", "pooled")
        df_pooled.insert(3, "n_scope", int(Xte_s.shape[0]))
        df_pooled.insert(4, "events_scope", int(np.sum(e_te)))
        df_pooled.to_csv(
            outdir / f"treatment_adjusted_pooled_FC_H{int(H)}.csv", index=False
        )
        if make_plots:
            _plot_treatment_adjusted_fc(
                df_pooled,
                f"Treatment-conditioned pooled feature collapsing (H={int(H)} mo)",
                outdir / f"TreatmentAdjustedFC_Pooled_H{int(H)}_top{int(TREATMENT_ADJUSTED_FC_TOPN)}.png",
                topN=TREATMENT_ADJUSTED_FC_TOPN,
            )
        all_rows.append(df_pooled)
        context_rows.append(
            {
                "H": float(H),
                "scope": "pooled",
                "group": "pooled",
                "n": int(Xte_s.shape[0]),
                "events": int(np.sum(e_te)),
                "skipped": False,
                "skip_reason": "",
                "conditioned_on_complete_treatment_profile": True,
                "treatment_columns": ";".join(tx_cols),
            }
        )

        for gi, gname in enumerate(groups):
            mask = g_te == int(gi)
            n_g = int(np.sum(mask))
            e_g = int(np.sum(e_te[mask])) if n_g else 0
            skip_reason = ""
            if n_g < int(TREATMENT_ADJUSTED_FC_MIN_N):
                skip_reason = f"n<{int(TREATMENT_ADJUSTED_FC_MIN_N)}"
            elif e_g < int(TREATMENT_ADJUSTED_FC_MIN_EVENTS):
                skip_reason = f"events<{int(TREATMENT_ADJUSTED_FC_MIN_EVENTS)}"
            context_rows.append(
                {
                    "H": float(H),
                    "scope": "group_specific",
                    "group": str(gname),
                    "n": n_g,
                    "events": e_g,
                    "skipped": bool(skip_reason),
                    "skip_reason": skip_reason,
                    "conditioned_on_complete_treatment_profile": True,
                    "treatment_columns": ";".join(tx_cols),
                }
            )
            if skip_reason:
                continue
            safe_g = _safe_fname(gname)
            tx_profile_labels_g, tx_cols_g = _treatment_profile_labels_for_scope(
                A_tx, tx_cols, scope="group_specific", group=str(gname)
            )
            df_g = overall_fc_df(
                art["model"],
                Xte_s[mask, :],
                g_te[mask],
                feat_names,
                art["alpha_dt"],
                art["dt_edges"],
                H=H,
                S=FC_S,
                R=FC_REPEATS,
                use_crn=FC_USE_CRN,
                condition_labels=tx_profile_labels_g[mask],
                skip_feature_idxs=skip_feature_idxs,
                fc_context_label=f"group_specific_test_treatment_adjusted_{gname}",
            )
            df_g = _add_fc_feature_family_columns(df_g)
            df_g.insert(0, "H", float(H))
            df_g.insert(1, "scope", "group_specific")
            df_g.insert(2, "group", str(gname))
            df_g.insert(3, "n_scope", n_g)
            df_g.insert(4, "events_scope", e_g)
            df_g.to_csv(
                outdir / f"treatment_adjusted_group_FC_H{int(H)}__{safe_g}.csv",
                index=False,
            )
            if make_plots:
                _plot_treatment_adjusted_fc(
                    df_g,
                    f"{gname}: treatment-conditioned feature collapsing (H={int(H)} mo)",
                    outdir / f"TreatmentAdjustedFC_Group_{safe_g}_H{int(H)}_top{int(TREATMENT_ADJUSTED_FC_TOPN)}.png",
                    topN=TREATMENT_ADJUSTED_FC_TOPN,
                )
            all_rows.append(df_g)

        # Exact, non-overlapping treatment-profile FC rankings for landscape endpoint annotation.
        # A patient belongs to exactly one treatment_profile cell; multi-treatment cells are
        # not averaged across single-treatment contexts and require larger support.
        tx_labels_str = tx_profile_labels.astype(str)
        for tx_lab in sorted(pd.Index(tx_labels_str).unique().tolist()):
            tx_mask = tx_labels_str == str(tx_lab)
            n_tx = int(np.sum(tx_mask))
            e_tx = int(np.sum(e_te[tx_mask])) if n_tx else 0
            min_n_tx, min_e_tx = _exact_profile_fc_min_support(tx_lab)
            skip_reason = ""
            if n_tx < min_n_tx:
                skip_reason = f"n<{min_n_tx}"
            elif e_tx < min_e_tx:
                skip_reason = f"events<{min_e_tx}"
            context_rows.append(
                {
                    "H": float(H),
                    "scope": "pooled_exact_treatment_profile",
                    "group": "pooled",
                    "treatment_profile": str(tx_lab),
                    "treatment_active_count": int(_treatment_profile_active_count(tx_lab)),
                    "is_multi_treatment_profile": bool(_is_multi_treatment_profile(tx_lab)),
                    "n": n_tx,
                    "events": e_tx,
                    "skipped": bool(skip_reason),
                    "skip_reason": skip_reason,
                    "conditioned_on_complete_treatment_profile": True,
                    "exact_nonoverlapping_treatment_profile": True,
                    "treatment_columns": ";".join(tx_cols),
                }
            )
            if not skip_reason:
                if TREATMENT_ADJUSTED_FC_STRATIFY_BY_CANCER_GROUP:
                    cond_tx = np.asarray([groups[int(g)] if 0 <= int(g) < len(groups) else f"group{int(g)}" for g in g_te[tx_mask]], dtype=object)
                else:
                    cond_tx = None
                df_tx = overall_fc_df(
                    art["model"],
                    Xte_s[tx_mask, :],
                    g_te[tx_mask],
                    feat_names,
                    art["alpha_dt"],
                    art["dt_edges"],
                    H=H,
                    S=FC_S,
                    R=FC_REPEATS,
                    use_crn=FC_USE_CRN,
                    condition_labels=cond_tx,
                    skip_feature_idxs=skip_feature_idxs,
                    fc_context_label=f"pooled_exact_treatment_profile_{tx_lab}",
                )
                df_tx = _add_fc_feature_family_columns(df_tx)
                df_tx.insert(0, "H", float(H))
                df_tx.insert(1, "scope", "pooled_exact_treatment_profile")
                df_tx.insert(2, "group", "pooled")
                df_tx.insert(3, "treatment_profile", str(tx_lab))
                df_tx.insert(4, "treatment_active_count", int(_treatment_profile_active_count(tx_lab)))
                df_tx.insert(5, "is_multi_treatment_profile", bool(_is_multi_treatment_profile(tx_lab)))
                df_tx.insert(6, "n_scope", n_tx)
                df_tx.insert(7, "events_scope", e_tx)
                df_tx.to_csv(
                    outdir / f"treatment_adjusted_exact_profile_FC_H{int(H)}__{_safe_fname(tx_lab)}.csv",
                    index=False,
                )
                if make_plots:
                    _plot_treatment_adjusted_fc(
                        df_tx,
                        f"{tx_lab}: exact-profile feature collapsing (H={int(H)} mo)",
                        outdir / f"TreatmentAdjustedFC_ExactProfile_{_safe_fname(tx_lab)}_H{int(H)}_top{int(TREATMENT_ADJUSTED_FC_TOPN)}.png",
                        topN=TREATMENT_ADJUSTED_FC_TOPN,
                    )
                all_rows.append(df_tx)

        # Cancer-specific exact treatment-profile rankings. Breast retains hormone;
        # Lung/Colon exclude hormone so sparse/non-biological hormone cells do not
        # appear in those cancer-specific summaries.
        for gi, gname in enumerate(groups):
            group_mask = g_te == int(gi)
            tx_profile_labels_g, tx_cols_g = _treatment_profile_labels_for_scope(
                A_tx, tx_cols, scope="group_exact_treatment_profile", group=str(gname)
            )
            tx_labels_g = tx_profile_labels_g.astype(str)
            for tx_lab_g in sorted(pd.Index(tx_labels_g[group_mask]).unique().tolist()):
                mask_gtx = group_mask & (tx_labels_g == str(tx_lab_g))
                n_gtx = int(np.sum(mask_gtx))
                e_gtx = int(np.sum(e_te[mask_gtx])) if n_gtx else 0
                min_n_gtx, min_e_gtx = _exact_profile_fc_min_support(tx_lab_g)
                skip_reason_g = ""
                if n_gtx < min_n_gtx:
                    skip_reason_g = f"n<{min_n_gtx}"
                elif e_gtx < min_e_gtx:
                    skip_reason_g = f"events<{min_e_gtx}"
                context_rows.append(
                    {
                        "H": float(H),
                        "scope": "group_exact_treatment_profile",
                        "group": str(gname),
                        "treatment_profile": str(tx_lab_g),
                        "treatment_active_count": int(_treatment_profile_active_count(tx_lab_g)),
                        "is_multi_treatment_profile": bool(_is_multi_treatment_profile(tx_lab_g)),
                        "n": n_gtx,
                        "events": e_gtx,
                        "skipped": bool(skip_reason_g),
                        "skip_reason": skip_reason_g,
                        "conditioned_on_complete_treatment_profile": True,
                        "exact_nonoverlapping_treatment_profile": True,
                        "treatment_columns": ";".join(tx_cols_g),
                    }
                )
                if skip_reason_g:
                    continue
                safe_g = _safe_fname(gname)
                safe_tx = _safe_fname(tx_lab_g)
                df_gtx = overall_fc_df(
                    art["model"],
                    Xte_s[mask_gtx, :],
                    g_te[mask_gtx],
                    feat_names,
                    art["alpha_dt"],
                    art["dt_edges"],
                    H=H,
                    S=FC_S,
                    R=FC_REPEATS,
                    use_crn=FC_USE_CRN,
                    condition_labels=None,
                    skip_feature_idxs=skip_feature_idxs,
                    fc_context_label=f"group_exact_treatment_profile_{gname}_{tx_lab_g}",
                )
                df_gtx = _add_fc_feature_family_columns(df_gtx)
                df_gtx.insert(0, "H", float(H))
                df_gtx.insert(1, "scope", "group_exact_treatment_profile")
                df_gtx.insert(2, "group", str(gname))
                df_gtx.insert(3, "treatment_profile", str(tx_lab_g))
                df_gtx.insert(4, "treatment_active_count", int(_treatment_profile_active_count(tx_lab_g)))
                df_gtx.insert(5, "is_multi_treatment_profile", bool(_is_multi_treatment_profile(tx_lab_g)))
                df_gtx.insert(6, "n_scope", n_gtx)
                df_gtx.insert(7, "events_scope", e_gtx)
                df_gtx.to_csv(
                    outdir / f"treatment_adjusted_exact_profile_group_FC_H{int(H)}__{safe_g}__{safe_tx}.csv",
                    index=False,
                )
                if make_plots:
                    _plot_treatment_adjusted_fc(
                        df_gtx,
                        f"{gname}, {tx_lab_g}: exact-profile feature collapsing (H={int(H)} mo)",
                        outdir / f"TreatmentAdjustedFC_GroupExactProfile_{safe_g}_{safe_tx}_H{int(H)}_top{int(TREATMENT_ADJUSTED_FC_TOPN)}.png",
                        topN=TREATMENT_ADJUSTED_FC_TOPN,
                    )
                all_rows.append(df_gtx)

    pd.DataFrame(context_rows).to_csv(
        outdir / "treatment_adjusted_fc_scope_counts.csv", index=False
    )
    if len(all_rows):
        df_all = pd.concat(all_rows, axis=0, ignore_index=True)
        df_all.to_csv(
            outdir / "treatment_adjusted_feature_collapsing_pooled_and_group.csv",
            index=False,
        )
        df_rankable = df_all.copy()
        if "FC_conditioning_feature_not_ranked" in df_rankable.columns:
            df_rankable = df_rankable[
                ~df_rankable["FC_conditioning_feature_not_ranked"].astype(bool)
            ].copy()
        df_shared = df_rankable[
            df_rankable.get("is_shared_across_cancers_feature", False).astype(bool)
        ].copy() if len(df_rankable) else pd.DataFrame()
        if len(df_shared):
            df_shared.to_csv(
                outdir / "treatment_adjusted_shared_across_cancers_feature_collapsing.csv",
                index=False,
            )
            shared_rank_group_cols = [c for c in ["H", "scope", "group", "treatment_profile"] if c in df_shared.columns]
            shared_sort_cols = shared_rank_group_cols + ["FC_score"]
            top_shared_by_scope = (
                df_shared.sort_values(
                    shared_sort_cols,
                    ascending=[True] * len(shared_rank_group_cols) + [False],
                )
                .groupby(shared_rank_group_cols, dropna=False)
                .head(int(TREATMENT_ADJUSTED_FC_TOPN))
                .reset_index(drop=True)
            )
            top_shared_by_scope.to_csv(
                outdir / "treatment_adjusted_shared_across_cancers_top_by_scope.csv",
                index=False,
            )
        else:
            pd.DataFrame({"message": ["No shared-across-cancers latent features were rankable in this FC output."]}).to_csv(
                outdir / "treatment_adjusted_shared_across_cancers_feature_collapsing_skipped.csv",
                index=False,
            )
        if "scope" in df_all.columns:
            df_exact = df_all[df_all["scope"].astype(str).isin(["pooled_exact_treatment_profile", "group_exact_treatment_profile"])].copy()
            if len(df_exact):
                df_exact.to_csv(
                    outdir / "treatment_adjusted_exact_profile_feature_collapsing.csv",
                    index=False,
                )
        rank_group_cols = [c for c in ["H", "scope", "group", "treatment_profile"] if c in df_rankable.columns]
        sort_cols = rank_group_cols + ["FC_score"]
        top_by_scope = (
            df_rankable.sort_values(
                sort_cols,
                ascending=[True] * len(rank_group_cols) + [False],
            )
            .groupby(rank_group_cols, dropna=False)
            .head(int(TREATMENT_ADJUSTED_FC_TOPN))
            .reset_index(drop=True)
        )
        top_by_scope.to_csv(
            outdir / "treatment_adjusted_feature_collapsing_top_by_scope.csv",
            index=False,
        )
        return df_all
    return pd.DataFrame(context_rows)


def run_treatment_conditioned_fc_analysis(art, outdir):
    """Treatment-conditioned feature-collapse ranking for candidate resistance programs.

    Patients with multiple treatments are deliberately included in every
    corresponding treatment-positive context. The reference context is
    TX_EXPLICIT_NO_TREATMENT when available, otherwise all modeled treatment
    flags equal zero.
    """
    if not ENABLE_TREATMENT_CONDITIONED_FC:
        return None
    outdir = Path(outdir) / "treatment_conditioned_feature_collapse"
    outdir.mkdir(parents=True, exist_ok=True)
    meta_te = art.get("meta_te")
    if meta_te is None:
        pd.DataFrame(
            {
                "message": [
                    "No test metadata available; skipped treatment-conditioned FC."
                ]
            }
        ).to_csv(outdir / "treatment_conditioned_fc_skipped.csv", index=False)
        return None
    meta_te = meta_te.reset_index(drop=True).copy()
    Xte_s = art["Xte_t"].detach().cpu().numpy()
    g_te = np.asarray(art["g_te"], int)
    t_te = np.asarray(art["t_te"], float)
    e_te = np.asarray(art["e_te"], int)
    feat_names = [str(x) for x in art["feat_names"]]
    if len(meta_te) != Xte_s.shape[0]:
        pd.DataFrame(
            {
                "message": [
                    f"meta_te length {len(meta_te)} does not match Xte rows {Xte_s.shape[0]}; skipped treatment-conditioned FC."
                ]
            }
        ).to_csv(outdir / "treatment_conditioned_fc_skipped.csv", index=False)
        return None
    contexts = _make_treatment_context_masks(meta_te)
    if len(contexts) == 0:
        pd.DataFrame(
            {
                "message": [
                    "No treatment context columns available; skipped treatment-conditioned FC."
                ]
            }
        ).to_csv(outdir / "treatment_conditioned_fc_skipped.csv", index=False)
        return None
    note = (
        "Treatment-conditioned feature-collapse contexts are not mutually exclusive. "
        "Patients receiving multiple treatments are included in every corresponding treatment-positive context. "
        "The reference context is explicit no recognized treatment when TX_EXPLICIT_NO_TREATMENT is available; otherwise it is all modeled treatment flags equal zero. "
        "All scores are posterior predictive feature-collapse scores within the specified context and are not causal treatment effects.\n"
    )
    (outdir / "README_treatment_conditioned_feature_collapse.txt").write_text(
        note, encoding="utf-8"
    )
    context_rows = []
    all_fc_rows = []
    mechanism_rows = []
    reference_by_H = {}
    for H in [float(h) for h in TREATMENT_CONTEXT_FC_H_LIST]:
        for ctx in contexts:
            mask = np.asarray(ctx["mask"], bool)
            n_ctx = int(np.sum(mask))
            e_ctx = int(np.sum(e_te[mask])) if n_ctx else 0
            groups_present = (
                sorted(set([str(art["groups"][int(g)]) for g in np.unique(g_te[mask])]))
                if n_ctx
                else []
            )
            context_rows.append(
                {
                    "H": float(H),
                    "context_id": str(ctx["context_id"]),
                    "context_label": str(ctx["context_label"]),
                    "context_type": str(ctx["context_type"]),
                    "treatment_flag": str(ctx["treatment_flag"]),
                    "n": n_ctx,
                    "events": e_ctx,
                    "groups_present": ";".join(groups_present),
                    "nonexclusive_treatment_context": bool(
                        ctx.get("nonexclusive_treatment_context", False)
                    ),
                    "skipped": bool(
                        n_ctx < int(TREATMENT_CONTEXT_FC_MIN_N)
                        or e_ctx < int(TREATMENT_CONTEXT_FC_MIN_EVENTS)
                    ),
                    "skip_reason": (
                        ""
                        if (
                            n_ctx >= int(TREATMENT_CONTEXT_FC_MIN_N)
                            and e_ctx >= int(TREATMENT_CONTEXT_FC_MIN_EVENTS)
                        )
                        else f"n<{int(TREATMENT_CONTEXT_FC_MIN_N)} or events<{int(TREATMENT_CONTEXT_FC_MIN_EVENTS)}"
                    ),
                }
            )
            if n_ctx < int(TREATMENT_CONTEXT_FC_MIN_N) or e_ctx < int(
                TREATMENT_CONTEXT_FC_MIN_EVENTS
            ):
                continue
            df_ctx = overall_fc_df(
                art["model"],
                Xte_s[mask, :],
                g_te[mask],
                feat_names,
                art["alpha_dt"],
                art["dt_edges"],
                H=H,
                S=FC_S,
                R=FC_REPEATS,
                use_crn=FC_USE_CRN,
            )
            df_ctx.insert(0, "H", float(H))
            df_ctx.insert(1, "context_id", str(ctx["context_id"]))
            df_ctx.insert(2, "context_label", str(ctx["context_label"]))
            df_ctx.insert(3, "context_type", str(ctx["context_type"]))
            df_ctx.insert(4, "treatment_flag", str(ctx["treatment_flag"]))
            df_ctx.insert(5, "n_context", n_ctx)
            df_ctx.insert(6, "events_context", e_ctx)
            df_ctx.insert(
                7,
                "nonexclusive_treatment_context",
                bool(ctx.get("nonexclusive_treatment_context", False)),
            )
            df_ctx["feature_family"] = [
                _feature_family_for_mechanism(f) for f in df_ctx["feature"].astype(str)
            ]
            df_ctx.to_csv(
                outdir
                / f"treatment_context_FC_H{int(H)}__{_safe_fname(ctx['context_id'])}.csv",
                index=False,
            )
            all_fc_rows.append(df_ctx)
            _plot_treatment_context_fc_lollipop(
                df_ctx,
                f"{ctx['context_label']}: all features by posterior FC (H={int(H)} mo)",
                outdir
                / f"TreatmentContextFC_AllFeatures_H{int(H)}__{_safe_fname(ctx['context_id'])}.png",
                topN=TREATMENT_CONTEXT_FC_TOPN,
            )
            mech = df_ctx[
                df_ctx["feature_family"].isin(
                    [str(x) for x in TREATMENT_CONTEXT_FC_FAMILIES]
                )
            ].copy()
            if len(mech):
                mechanism_rows.append(mech)
                _plot_treatment_context_fc_lollipop(
                    mech,
                    f"{ctx['context_label']}: non-treatment clinical and gene-expression PC features by posterior FC (H={int(H)} mo)",
                    outdir
                    / f"TreatmentContextMechanismFC_CandidateFamilies_H{int(H)}__{_safe_fname(ctx['context_id'])}.png",
                    topN=TREATMENT_CONTEXT_FC_TOPN,
                )
            if str(ctx["context_type"]) == "reference_no_recognized_treatment":
                reference_by_H[float(H)] = (
                    df_ctx[["feature", "FC_score"]]
                    .copy()
                    .rename(columns={"FC_score": "FC_score_reference"})
                )
    df_contexts = pd.DataFrame(context_rows)
    df_contexts.to_csv(
        outdir / "treatment_conditioned_fc_context_counts.csv", index=False
    )
    if len(all_fc_rows) == 0:
        return df_contexts
    df_all = pd.concat(all_fc_rows, axis=0, ignore_index=True)
    df_all.to_csv(
        outdir / "treatment_conditioned_feature_collapsing_all_contexts.csv",
        index=False,
    )
    df_mech = (
        pd.concat(mechanism_rows, axis=0, ignore_index=True)
        if len(mechanism_rows)
        else pd.DataFrame()
    )
    if len(df_mech):
        df_mech.to_csv(
            outdir / "treatment_conditioned_candidate_resistance_features.csv",
            index=False,
        )
        top_summary = (
            df_mech.sort_values(
                ["H", "context_id", "FC_score"], ascending=[True, True, False]
            )
            .groupby(
                ["H", "context_id", "context_label", "feature_family"], dropna=False
            )
            .head(int(TREATMENT_CONTEXT_FC_TOPN))
            .reset_index(drop=True)
        )
        top_summary.to_csv(
            outdir
            / "treatment_conditioned_candidate_resistance_features_top_by_context.csv",
            index=False,
        )
        for H in sorted(df_mech["H"].dropna().unique().tolist()):
            _plot_treatment_context_heatmap(
                df_mech,
                float(H),
                outdir / f"TreatmentContextMechanismFC_Heatmap_H{int(H)}.png",
                topN=min(int(TREATMENT_CONTEXT_FC_TOPN), 18),
            )
    delta_rows = []
    if TREATMENT_CONTEXT_FC_COMPARE_TO_NO_RECOGNIZED:
        for H, ref in reference_by_H.items():
            sub = (
                df_mech[
                    np.isclose(pd.to_numeric(df_mech["H"], errors="coerce"), float(H))
                ].copy()
                if len(df_mech)
                else pd.DataFrame()
            )
            if len(sub) == 0 or ref is None or len(ref) == 0:
                continue
            sub = sub[sub["context_type"].astype(str) == "treatment_positive"].copy()
            for ctx_id, dctx in sub.groupby("context_id", dropna=False):
                merged = dctx.merge(ref, on="feature", how="left")
                merged["delta_FC_score_vs_reference"] = pd.to_numeric(
                    merged["FC_score"], errors="coerce"
                ) - pd.to_numeric(merged["FC_score_reference"], errors="coerce")
                merged["abs_delta_FC_score_vs_reference"] = merged[
                    "delta_FC_score_vs_reference"
                ].abs()
                delta_rows.append(merged)
                _plot_treatment_vs_reference_delta(
                    merged,
                    f"{dctx['context_label'].iloc[0]} vs no-recognized-treatment: candidate resistance features (H={int(H)} mo)",
                    outdir
                    / f"TreatmentVsReferenceMechanismFC_Delta_H{int(H)}__{_safe_fname(ctx_id)}.png",
                    topN=TREATMENT_CONTEXT_FC_TOPN,
                )
    if len(delta_rows):
        df_delta = pd.concat(delta_rows, axis=0, ignore_index=True)
        df_delta.to_csv(
            outdir
            / "treatment_vs_no_recognized_candidate_resistance_feature_deltas.csv",
            index=False,
        )
    return df_all





def _latent_matrix_for_landscape(model, X_t, g_t, sample=False):
    """Extract the trained DGP shared/private latent representation for posthoc plots."""
    model.eval()
    g_t = g_t.to(device=X_t.device, dtype=torch.long).view(-1)
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(_active_cholesky_jitter()):
        h = model.forward_latents(X_t, g_t, sample=bool(sample))
    H = h.detach().cpu().numpy().astype(float)
    if H.ndim != 2:
        H = np.reshape(H, (H.shape[0], -1))
    return H


def _standardize_latent_matrix_for_projection(H):
    H = np.asarray(H, float)
    if H.ndim != 2 or H.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    mu = np.nanmean(H, axis=0, keepdims=True)
    sd = np.nanstd(H, axis=0, keepdims=True)
    sd = np.where((sd <= 1e-8) | ~np.isfinite(sd), 1.0, sd)
    Z = (H - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0)


def _pca_project_state_matrix_to_2d(H):
    """Deterministic PCA/SVD projection of patient-horizon states."""
    Z = _standardize_latent_matrix_for_projection(H)
    if Z.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), {"projection": "empty"}
    if Z.shape[1] == 0:
        return np.zeros((Z.shape[0], 2), dtype=float), {"projection": "empty_state"}
    if Z.shape[1] == 1:
        coords = np.column_stack([Z[:, 0], np.zeros(Z.shape[0])])
        return coords.astype(float), {
            "projection": "PCA_on_centered_scaled_patient_horizon_states",
            "state_dim": int(Z.shape[1]),
            "var_explained_1": 1.0,
            "var_explained_2": 0.0,
        }
    try:
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        coords = U[:, :2] * S[:2]
        if coords.shape[1] < 2:
            coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
        denom = float(np.sum(S ** 2)) if S.size else 0.0
        ve = (S[:2] ** 2) / denom if denom > 0 else np.array([np.nan, np.nan])
        return coords.astype(float), {
            "projection": "PCA_on_centered_scaled_patient_horizon_states",
            "state_dim": int(Z.shape[1]),
            "var_explained_1": float(ve[0]) if ve.size > 0 and np.isfinite(ve[0]) else np.nan,
            "var_explained_2": float(ve[1]) if ve.size > 1 and np.isfinite(ve[1]) else np.nan,
        }
    except Exception as exc:
        coords = Z[:, :2] if Z.shape[1] >= 2 else np.column_stack([Z[:, 0], np.zeros(Z.shape[0])])
        return coords.astype(float), {
            "projection": "fallback_first_two_standardized_state_dimensions",
            "state_dim": int(Z.shape[1]),
            "fallback_reason": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def _project_state_matrix_to_2d(H, method="pca"):
    """Project patient-horizon states with either PCA or UMAP.

    PCA is deterministic and metrically easier to defend. UMAP is saved as a
    nonlinear visualization from the same standardized state matrix.
    """
    method = str(method).lower().strip()
    H = np.asarray(H, float)
    if H.ndim != 2 or H.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), {"projection": "empty", "requested_method": method}
    if method == "pca":
        coords, info = _pca_project_state_matrix_to_2d(H)
        info["requested_method"] = "pca"
        return coords, info
    if method != "umap":
        coords, info = _pca_project_state_matrix_to_2d(H)
        info["requested_method"] = method
        info["fallback_reason"] = f"unknown_projection_method:{method}"
        return coords, info
    Z = _standardize_latent_matrix_for_projection(H)
    n = int(Z.shape[0])
    if n < 5 or Z.shape[1] < 2:
        coords, info = _pca_project_state_matrix_to_2d(H)
        info["requested_method"] = "umap"
        info["umap_fallback_reason"] = "too_few_rows_or_state_dimensions"
        return coords, info
    try:
        import umap  # type: ignore
        n_neighbors = int(min(max(2, n - 1), max(2, int(RESISTANCE_PSEUDOTRAJECTORY_UMAP_N_NEIGHBORS))))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=float(RESISTANCE_PSEUDOTRAJECTORY_UMAP_MIN_DIST),
            metric="euclidean",
            random_state=int(RESISTANCE_PSEUDOTRAJECTORY_UMAP_RANDOM_STATE),
        )
        coords = reducer.fit_transform(Z)
        return np.asarray(coords, float), {
            "projection": "UMAP_on_centered_scaled_patient_horizon_states",
            "requested_method": "umap",
            "state_dim": int(H.shape[1]),
            "n_neighbors": int(n_neighbors),
            "min_dist": float(RESISTANCE_PSEUDOTRAJECTORY_UMAP_MIN_DIST),
            "random_state": int(RESISTANCE_PSEUDOTRAJECTORY_UMAP_RANDOM_STATE),
        }
    except Exception as exc:
        coords, info = _pca_project_state_matrix_to_2d(H)
        info["requested_method"] = "umap"
        info["umap_fallback_reason"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        return coords, info


def _observed_event_by_horizon(times, events, H):
    t = np.asarray(times, float)
    e = np.asarray(events, int)
    return ((e == 1) & np.isfinite(t) & (t <= float(H))).astype(int)


def _load_posthoc_risk_table_for_landscape(outdir):
    p = Path(outdir) / "posthoc_crossfit_test_native_and_calibrated_risks.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def _landscape_risk_vector(art, outdir, H):
    """Prefer calibrated fixed-horizon risk when available; otherwise use native DGP risk."""
    H_int = int(float(H))
    risk_table = _load_posthoc_risk_table_for_landscape(outdir)
    preferred_col = f"p_calibrated_{RESISTANCE_PSEUDOTRAJECTORY_PRIMARY_CALIBRATION_KIND}_H{H_int}"
    native_col = f"p_native_H{H_int}"
    if RESISTANCE_PSEUDOTRAJECTORY_USE_CALIBRATED_PRIMARY_RISK and risk_table is not None:
        if preferred_col in risk_table.columns and len(risk_table) == len(art["t_te"]):
            return risk_table[preferred_col].to_numpy(float), f"calibrated_{RESISTANCE_PSEUDOTRAJECTORY_PRIMARY_CALIBRATION_KIND}"
        if native_col in risk_table.columns and len(risk_table) == len(art["t_te"]):
            return risk_table[native_col].to_numpy(float), "native_from_posthoc_table"
    p_native = _event_probability_mean_at_H(
        art["model"], art["Xte_t"], art["g_te_t"], float(H), art["alpha_dt"], art["dt_edges"], S=FINAL_EVAL_MC_SAMPLES
    )
    return np.asarray(p_native, float), "native_dgp"


def _load_treatment_conditioned_fc_table(outdir):
    fc_path = Path(outdir) / "treatment_adjusted_feature_collapse" / "treatment_adjusted_feature_collapsing_top_by_scope.csv"
    if not fc_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fc_path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "feature" not in df.columns or "FC_score" not in df.columns:
        return pd.DataFrame()
    if "feature_family" not in df.columns:
        df = _add_fc_feature_family_columns(df)
    if "FC_conditioning_feature_not_ranked" in df.columns:
        try:
            df = df[~df["FC_conditioning_feature_not_ranked"].astype(bool)].copy()
        except Exception:
            pass
    return df


def _fc_scope_filter(df, scope="pooled", group="pooled"):
    sub = df.copy()
    if "scope" in sub.columns:
        sub = sub[sub["scope"].astype(str) == str(scope)].copy()
    if "group" in sub.columns:
        sub = sub[sub["group"].astype(str) == str(group)].copy()
    return sub


def _horizon_averaged_fc_table(outdir, scope="pooled", group="pooled"):
    df = _load_treatment_conditioned_fc_table(outdir)
    if df.empty:
        return pd.DataFrame()
    sub = _fc_scope_filter(df, scope=scope, group=group)
    if sub.empty:
        return pd.DataFrame()
    H_allowed = np.asarray([float(h) for h in RESISTANCE_PSEUDOTRAJECTORY_H_LIST], dtype=float)
    sub["H"] = pd.to_numeric(sub.get("H"), errors="coerce")
    sub = sub[np.isfinite(sub["H"].to_numpy(float))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub[np.asarray([np.any(np.isclose(h, H_allowed)) for h in sub["H"].to_numpy(float)])].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["FC_score"] = pd.to_numeric(sub["FC_score"], errors="coerce")
    sub = sub[np.isfinite(sub["FC_score"].to_numpy(float))].copy()
    if sub.empty:
        return pd.DataFrame()
    gcols = ["feature"]
    if "feature_family" in sub.columns:
        gcols.append("feature_family")
    out = (
        sub.groupby(gcols, dropna=False)
        .agg(
            FC_score_horizon_mean=("FC_score", "mean"),
            FC_score_horizon_max=("FC_score", "max"),
            FC_horizon_count=("H", "nunique"),
            FC_horizons=("H", lambda x: ";".join(str(int(v)) for v in sorted(pd.Series(x).dropna().unique()))),
        )
        .reset_index()
        .sort_values("FC_score_horizon_mean", ascending=False)
    )
    out["scope"] = str(scope)
    out["group"] = str(group)
    return out



def _load_treatment_context_fc_table(outdir):
    """Load non-exclusive treatment-context FC rankings, if they were produced."""
    fc_path = Path(outdir) / "treatment_conditioned_feature_collapse" / "treatment_conditioned_feature_collapsing_all_contexts.csv"
    if not fc_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fc_path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "feature" not in df.columns or "FC_score" not in df.columns:
        return pd.DataFrame()
    if "feature_family" not in df.columns:
        df = _add_fc_feature_family_columns(df)
    return df


def _treatment_context_ids_from_profile_label(treatment_profile):
    """Map exact treatment-profile labels (e.g. TX=Chemo+Radiation) to existing non-exclusive FC contexts."""
    lab = str(treatment_profile)
    if lab in {"", "ALL", "nan", "None"}:
        return []
    low = lab.lower()
    if "tx=none" in low or "none_observed" in low or "no recognized" in low:
        return ["NO_RECOGNIZED_TREATMENT"]
    if lab.startswith("TX="):
        active_labels = [x.strip() for x in lab[3:].split("+") if x.strip()]
    else:
        active_labels = [x.strip() for x in lab.split("+") if x.strip()]
    label_to_col = {_treatment_label(c): c for c in TREATMENT_FLAG_COLS}
    ids = []
    for active in active_labels:
        if active in label_to_col:
            ids.append(label_to_col[active])
        else:
            # Allow raw metadata/design names as a fallback.
            base = _base_treatment_col(active)
            if base in TREATMENT_FLAG_COLS:
                ids.append(base)
    return list(dict.fromkeys(ids))


def _horizon_averaged_treatment_profile_fc_table(outdir, treatment_profile, scope="pooled_exact_treatment_profile", group="pooled"):
    """Use exact, non-overlapping treatment-profile FC rankings.

    Multi-treatment profiles such as TX=Chemo+Radiation are not averaged from
    single-treatment contexts.  They are read from the exact full-profile FC
    outputs generated by run_treatment_adjusted_pooled_and_group_fc_analysis.
    """
    fc_path = Path(outdir) / "treatment_adjusted_feature_collapse" / "treatment_adjusted_feature_collapsing_top_by_scope.csv"
    if not fc_path.exists():
        fc_path = Path(outdir) / "treatment_adjusted_feature_collapse" / "treatment_adjusted_feature_collapsing_pooled_and_group.csv"
    if not fc_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fc_path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "feature" not in df.columns or "FC_score" not in df.columns:
        return pd.DataFrame()
    if "feature_family" not in df.columns:
        df = _add_fc_feature_family_columns(df)
    if "FC_conditioning_feature_not_ranked" in df.columns:
        try:
            df = df[~df["FC_conditioning_feature_not_ranked"].astype(bool)].copy()
        except Exception:
            pass
    sub = df.copy()
    if "scope" in sub.columns:
        sub = sub[sub["scope"].astype(str) == str(scope)].copy()
    if "group" in sub.columns:
        sub = sub[sub["group"].astype(str) == str(group)].copy()
    if "treatment_profile" not in sub.columns:
        return pd.DataFrame()
    sub = sub[sub["treatment_profile"].astype(str) == str(treatment_profile)].copy()
    if sub.empty:
        return pd.DataFrame()
    H_allowed = np.asarray([float(h) for h in RESISTANCE_PSEUDOTRAJECTORY_H_LIST], dtype=float)
    sub["H"] = pd.to_numeric(sub.get("H"), errors="coerce")
    sub = sub[np.isfinite(sub["H"].to_numpy(float))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub[np.asarray([np.any(np.isclose(h, H_allowed)) for h in sub["H"].to_numpy(float)])].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["FC_score"] = pd.to_numeric(sub["FC_score"], errors="coerce")
    sub = sub[np.isfinite(sub["FC_score"].to_numpy(float))].copy()
    if sub.empty:
        return pd.DataFrame()
    gcols = ["feature"]
    if "feature_family" in sub.columns:
        gcols.append("feature_family")
    agg = (
        sub.groupby(gcols, dropna=False)
        .agg(
            FC_score_horizon_mean=("FC_score", "mean"),
            FC_score_horizon_max=("FC_score", "max"),
            FC_horizon_count=("H", "nunique"),
            FC_horizons=("H", lambda x: ";".join(str(int(v)) for v in sorted(pd.Series(x).dropna().unique()))),
        )
        .reset_index()
        .sort_values("FC_score_horizon_mean", ascending=False)
    )
    agg["scope"] = str(scope)
    agg["group"] = str(group)
    agg["treatment_profile"] = str(treatment_profile)
    agg["FC_ranking_source"] = "treatment_adjusted_exact_nonoverlapping_profile_FC"
    agg["FC_matched_context_ids"] = str(treatment_profile)
    agg["treatment_active_count"] = int(_treatment_profile_active_count(treatment_profile))
    agg["is_multi_treatment_profile"] = bool(_is_multi_treatment_profile(treatment_profile))
    return agg

def _top_fc_feature_table_for_scope(
    outdir,
    scope="pooled",
    group="pooled",
    topn=RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_TOP_FC_FEATURES,
    treatment_profile=None,
):
    """Return FC-prioritized non-treatment features for a scope/group or treatment profile."""
    # For exact treatment-profile landscape panels, prefer the already-computed
    # treatment-specific FC rankings. This prevents treatment panels from being
    # annotated with only pooled/cancer-specific FC rankings.
    if treatment_profile is not None and str(treatment_profile) not in {"", "ALL", "nan", "None"}:
        tab = _horizon_averaged_treatment_profile_fc_table(
            outdir, treatment_profile=treatment_profile, scope=scope, group=group
        )
        if tab is not None and not tab.empty:
            tab = tab.copy()
            tab["FC_annotation_scope"] = "exact_nonoverlapping_treatment_profile_specific"
        else:
            tab = _horizon_averaged_fc_table(outdir, scope=scope, group=group)
            if tab is not None and not tab.empty:
                tab = tab.copy()
                tab["FC_annotation_scope"] = "fallback_scope_group_treatment_adjusted"
                tab["requested_treatment_profile"] = str(treatment_profile)
    else:
        tab = _horizon_averaged_fc_table(outdir, scope=scope, group=group)
        if tab is not None and not tab.empty:
            tab = tab.copy()
            tab["FC_annotation_scope"] = "scope_group_treatment_adjusted"

    if tab is None or tab.empty:
        return pd.DataFrame()
    tab = tab.copy()
    if "feature_family" in tab.columns:
        tab = tab[tab["feature_family"].astype(str) != "treatment"].copy()
    if tab.empty or "feature" not in tab.columns:
        return pd.DataFrame()
    tab["FC_score_horizon_mean"] = pd.to_numeric(tab.get("FC_score_horizon_mean"), errors="coerce")
    tab = tab[np.isfinite(tab["FC_score_horizon_mean"].to_numpy(float))].copy()
    if tab.empty:
        return pd.DataFrame()
    tab = tab.sort_values("FC_score_horizon_mean", ascending=False).head(int(topn))
    max_fc = float(np.nanmax(tab["FC_score_horizon_mean"].to_numpy(float)))
    if not np.isfinite(max_fc) or max_fc <= 0:
        max_fc = 1.0
    tab["FC_weight"] = np.clip(tab["FC_score_horizon_mean"].to_numpy(float) / max_fc, 0.0, np.inf)
    return tab.reset_index(drop=True)

def _landscape_fc_avg_top_text(
    outdir,
    scope="pooled",
    group="pooled",
    topn=RESISTANCE_PSEUDOTRAJECTORY_FC_ANNOTATION_TOPN,
    treatment_profile=None,
):
    tab = _top_fc_feature_table_for_scope(
        outdir,
        scope=scope,
        group=group,
        topn=topn,
        treatment_profile=treatment_profile,
    )
    if tab.empty:
        return ""
    source = str(tab.get("FC_annotation_scope", pd.Series([""])).iloc[0]) if "FC_annotation_scope" in tab.columns else ""
    if source == "exact_nonoverlapping_treatment_profile_specific":
        lines = ["Top exact-profile FC features:"]
    elif source.startswith("fallback"):
        lines = ["Top FC features (fallback ranking):"]
    else:
        lines = ["Top FC-prioritized features:"]
    for _, r in tab.iterrows():
        lines.append(f"• {_shorten_label_for_plot(str(r['feature']), 34)}")
    return "\n".join(lines)

def _fc_feature_signature_matrix(X_raw, feat_names, fc_tab):
    """Build a standardized, FC-weighted feature signature matrix."""
    X = np.asarray(X_raw, float)
    feat_names = [str(f) for f in feat_names]
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return np.zeros((X.shape[0] if X.ndim >= 1 else 0, 0), dtype=float), [], np.asarray([], float)
    if fc_tab is None or len(fc_tab) == 0 or "feature" not in fc_tab.columns:
        return np.zeros((X.shape[0], 0), dtype=float), [], np.asarray([], float)
    fmap = {str(f): j for j, f in enumerate(feat_names)}
    idxs, used_features, weights = [], [], []
    for _, r in fc_tab.iterrows():
        f = str(r.get("feature"))
        if f not in fmap:
            continue
        base = _base_treatment_col(f)
        if base in set(TREATMENT_FLAG_COLS):
            continue
        j = int(fmap[f])
        if j < 0 or j >= X.shape[1]:
            continue
        idxs.append(j)
        used_features.append(f)
        w = pd.to_numeric(pd.Series([r.get("FC_weight", r.get("FC_score_horizon_mean", 1.0))]), errors="coerce").iloc[0]
        weights.append(float(w) if np.isfinite(w) else 1.0)
    if len(idxs) == 0:
        return np.zeros((X.shape[0], 0), dtype=float), [], np.asarray([], float)
    V = X[:, idxs].astype(float)
    mu = np.nanmean(V, axis=0, keepdims=True)
    sd = np.nanstd(V, axis=0, keepdims=True)
    sd = np.where((sd <= 1e-8) | ~np.isfinite(sd), 1.0, sd)
    Vz = (V - mu) / sd
    Vz = np.where(np.isfinite(Vz), Vz, 0.0)
    weights = np.asarray(weights, dtype=float)
    if weights.size:
        wmax = float(np.nanmax(np.abs(weights)))
        if not np.isfinite(wmax) or wmax <= 0:
            wmax = 1.0
        weights = weights / wmax
        weights = np.where(np.isfinite(weights), weights, 1.0)
        Vz = Vz * weights.reshape(1, -1)
    return Vz.astype(float), used_features, weights.astype(float)


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR adjustment for a one-dimensional p-value vector."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if int(np.sum(finite)) == 0:
        return q
    pv = p[finite]
    m = float(len(pv))
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(len(ranked), dtype=float) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    qv = np.empty_like(adj)
    qv[order] = adj
    q[finite] = qv
    return q


def _bootstrap_mean_delta_ci(xin, xout, B=500, q=(0.05, 0.95), seed=0):
    """Bootstrap CI for mean(xin) - mean(xout)."""
    xin = np.asarray(xin, dtype=float)
    xout = np.asarray(xout, dtype=float)
    xin = xin[np.isfinite(xin)]
    xout = xout[np.isfinite(xout)]
    if xin.size < 3 or xout.size < 3 or int(B) <= 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        xb = xin[rng.integers(0, xin.size, size=xin.size)]
        yb = xout[rng.integers(0, xout.size, size=xout.size)]
        vals[b] = float(np.nanmean(xb) - np.nanmean(yb))
    lo, hi = np.nanquantile(vals, [float(q[0]), float(q[1])])
    return float(lo), float(hi), float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan


def _permutation_mean_delta_pvalue(xin, xout, B=500, seed=0):
    """Two-sided permutation p-value for a two-sample mean difference."""
    xin = np.asarray(xin, dtype=float)
    xout = np.asarray(xout, dtype=float)
    xin = xin[np.isfinite(xin)]
    xout = xout[np.isfinite(xout)]
    if xin.size < 3 or xout.size < 3 or int(B) <= 0:
        return np.nan
    obs = float(np.nanmean(xin) - np.nanmean(xout))
    pooled = np.concatenate([xin, xout])
    n_in = int(xin.size)
    rng = np.random.default_rng(int(seed))
    count = 0
    for _ in range(int(B)):
        perm = rng.permutation(pooled)
        delta = float(np.nanmean(perm[:n_in]) - np.nanmean(perm[n_in:]))
        if abs(delta) >= abs(obs) - 1e-15:
            count += 1
    return float((count + 1.0) / (float(B) + 1.0))


def _risk_anchor_values(patient_df):
    """Risk used for low/medium/high cohort assignment.

    Default is mean predicted risk over 18 and 24 months, because 3/6 months can
    be too sparse for stable trajectory stratification.
    """
    anchor_hs = [float(h) for h in RESISTANCE_PSEUDOTRAJECTORY_STRATIFY_H_LIST]
    cols = [f"risk_H{int(h)}" for h in anchor_hs if f"risk_H{int(h)}" in patient_df.columns]
    if not cols:
        fallback = f"risk_H{int(RESISTANCE_PSEUDOTRAJECTORY_PRIMARY_RISK_H)}"
        cols = [fallback] if fallback in patient_df.columns else []
    if not cols:
        risk_cols = [c for c in patient_df.columns if str(c).startswith("risk_H")]
        cols = risk_cols[:1]
    if not cols:
        return np.full(len(patient_df), np.nan, dtype=float), []
    vals = patient_df.loc[:, cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy(float)
    return vals, cols


def _build_patient_horizon_landscape_dataframe(patient_df, H_lat, X_raw, feat_names, outdir):
    """Create one row per patient per prediction horizon.

    The displayed x/y landscape is latent-only: PCA/UMAP is fit to the trained
    DGP shared/private latent representation H_lat. Risk is used for color and
    risk-cohort assignment only; prediction horizon is used only as the z-axis.
    Endpoint-neighborhood distances are also latent-only.
    """
    horizons = [float(h) for h in RESISTANCE_PSEUDOTRAJECTORY_H_LIST]
    n = int(len(patient_df))
    if n == 0 or len(horizons) == 0:
        return pd.DataFrame(), np.zeros((0, 0), dtype=float), {"projection_methods": []}

    rows = []
    H_lat = np.asarray(H_lat, float)
    if H_lat.ndim != 2 or H_lat.shape[0] != n:
        H_lat = np.zeros((n, 0), dtype=float)

    # Latent-only state matrix for projection and endpoint-neighborhood distance.
    # No risk, cumulative hazard, horizon, or FC features enter x/y geometry.
    latent_state = np.where(np.isfinite(H_lat), H_lat, 0.0)
    latent_state_std = _standardize_latent_matrix_for_projection(latent_state)

    anchor_vals, anchor_cols = _risk_anchor_values(patient_df)
    anchor_label = ",".join([c.replace("risk_H", "H") for c in anchor_cols]) if anchor_cols else "unavailable"

    # Project unique patient latent states once, then reuse x/y for every horizon.
    projection_info = {}
    patient_coords = {}
    methods = [str(m).lower() for m in RESISTANCE_PSEUDOTRAJECTORY_PROJECTION_METHODS]
    methods = list(dict.fromkeys([m for m in methods if m in {"pca", "umap"}])) or ["pca"]
    for method in methods:
        coords, info = _project_state_matrix_to_2d(
            latent_state_std,
            method=method,
            random_state=int(RESISTANCE_PSEUDOTRAJECTORY_UMAP_RANDOM_STATE),
        )
        projection_info[method] = info
        patient_coords[method] = coords

    for i in range(n):
        for H in horizons:
            H_int = int(H)
            risk_col = f"risk_H{H_int}"
            risk_val = pd.to_numeric(pd.Series([patient_df.iloc[i].get(risk_col, np.nan)]), errors="coerce").iloc[0]
            risk_val = float(risk_val) if np.isfinite(risk_val) else np.nan
            survival_val = float(1.0 - np.clip(risk_val, 0.0, 1.0)) if np.isfinite(risk_val) else np.nan
            cumhaz_val = float(-np.log(max(survival_val, 1e-12))) if np.isfinite(survival_val) else np.nan
            observed_event = int(
                pd.to_numeric(
                    pd.Series([patient_df.iloc[i].get(f"observed_event_by_H{H_int}", 0)]),
                    errors="coerce",
                ).fillna(0).iloc[0]
            )

            base = patient_df.iloc[i].to_dict()
            base.update(
                {
                    "horizon_months": float(H),
                    "risk_H": risk_val,
                    "survival_H": survival_val,
                    "cumhaz_H": cumhaz_val,
                    "risk_anchor_value": float(anchor_vals[i]) if np.isfinite(anchor_vals[i]) else np.nan,
                    "risk_anchor_columns": anchor_label,
                    "observed_event_by_horizon": observed_event,
                }
            )
            for k in range(H_lat.shape[1]):
                base[f"latent_dim_{k+1:02d}"] = float(H_lat[i, k])
            for k in range(latent_state_std.shape[1]):
                base[f"state_for_distance_{k+1:03d}"] = float(latent_state_std[i, k])
            for method, coords in patient_coords.items():
                base[f"landscape_1_{method}"] = float(coords[i, 0])
                base[f"landscape_2_{method}"] = float(coords[i, 1])
            rows.append(base)

    horizon_df = pd.DataFrame(rows)
    if len(horizon_df) == 0:
        return horizon_df, latent_state_std, projection_info

    # Backward-compatible default plotting columns use PCA if available, otherwise
    # the first requested projection.
    default_method = "pca" if "pca" in patient_coords else list(patient_coords.keys())[0]
    horizon_df["landscape_1"] = pd.to_numeric(horizon_df[f"landscape_1_{default_method}"], errors="coerce")
    horizon_df["landscape_2"] = pd.to_numeric(horizon_df[f"landscape_2_{default_method}"], errors="coerce")
    horizon_df["projection_method_default"] = default_method

    projection_info["projection_state"] = "trained_DGP_latent_only"
    projection_info["risk_in_projection"] = False
    projection_info["horizon_in_projection"] = False
    projection_info["cumhaz_in_projection"] = False
    projection_info["fc_features_in_projection"] = False
    projection_info["endpoint_distance_state"] = "standardized_trained_DGP_latent_only"
    projection_info["n_patient_horizon_rows"] = int(len(horizon_df))
    projection_info["n_unique_patients"] = int(n)
    projection_info["n_latent_dimensions"] = int(H_lat.shape[1])
    projection_info["horizons_months"] = horizons
    return horizon_df, latent_state_std, projection_info

def _assign_scope_risk_cohorts(dscope):
    """Assign low/medium/high risk cohorts within the plotted scope."""
    if dscope is None or len(dscope) == 0:
        return dscope
    out = dscope.copy()
    patient = out.drop_duplicates("test_index").copy()
    vals = pd.to_numeric(patient.get("risk_anchor_value"), errors="coerce").to_numpy(float)
    finite = np.isfinite(vals)
    if int(np.sum(finite)) < 3:
        out["risk_cohort"] = "All patients"
        out["risk_cohort_order"] = 0
        return out
    q1, q2 = np.nanquantile(vals[finite], [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite(q1) or not np.isfinite(q2) or q2 <= q1:
        q1, q2 = np.nanquantile(vals[finite], [0.5, 0.5])
    labels = []
    orders = []
    for v in vals:
        if not np.isfinite(v):
            labels.append("Unknown risk")
            orders.append(-1)
        elif v <= q1:
            labels.append("Low risk")
            orders.append(0)
        elif v <= q2:
            labels.append("Medium risk")
            orders.append(1)
        else:
            labels.append("High risk")
            orders.append(2)
    patient["risk_cohort"] = labels
    patient["risk_cohort_order"] = orders
    cmap = patient.set_index("test_index")["risk_cohort"].to_dict()
    omap = patient.set_index("test_index")["risk_cohort_order"].to_dict()
    out["risk_cohort"] = out["test_index"].map(cmap).fillna("Unknown risk")
    out["risk_cohort_order"] = out["test_index"].map(omap).fillna(-1).astype(int)
    return out


def _compute_risk_cohort_horizon_centroids(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    req = ["landscape_1", "landscape_2", "horizon_months", "risk_H", "risk_cohort", "risk_cohort_order"]
    if any(c not in df.columns for c in req):
        return pd.DataFrame()
    rows = []
    qlo, qhi = RESISTANCE_PSEUDOTRAJECTORY_BOOTSTRAP_Q
    B = int(RESISTANCE_PSEUDOTRAJECTORY_CENTROID_BOOTSTRAP_N)
    for (ord_val, cohort, H), d in df.groupby(["risk_cohort_order", "risk_cohort", "horizon_months"], dropna=False):
        try:
            ord_i = int(ord_val)
        except Exception:
            ord_i = -1
        if ord_i < 0:
            continue
        n = int(len(d))
        if n < int(RESISTANCE_PSEUDOTRAJECTORY_MIN_BIN_N):
            continue
        x = pd.to_numeric(d["landscape_1"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(d["landscape_2"], errors="coerce").to_numpy(float)
        r = pd.to_numeric(d["risk_H"], errors="coerce").to_numpy(float)
        ch = pd.to_numeric(d.get("cumhaz_H"), errors="coerce").to_numpy(float)
        valid_xy = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(valid_xy)) < int(RESISTANCE_PSEUDOTRAJECTORY_MIN_BIN_N):
            continue

        row = {
            "risk_cohort_order": ord_i,
            "risk_cohort": str(cohort),
            "horizon_months": float(H),
            "n": n,
            "events_by_horizon": int(pd.to_numeric(d.get("observed_event_by_horizon"), errors="coerce").fillna(0).sum()),
            "landscape_1": float(np.nanmedian(x)),
            "landscape_2": float(np.nanmedian(y)),
            "risk_H": float(np.nanmean(r)),
            "cumhaz_H": float(np.nanmean(ch)),
            "centroid_bootstrap_n": int(B) if B > 0 and n >= 3 else 0,
        }

        if B > 0 and int(np.sum(valid_xy)) >= 3:
            rng_seed = int(RESISTANCE_PSEUDOTRAJECTORY_INFERENCE_SEED + 100000 * max(ord_i, 0) + int(round(float(H) * 100)))
            rng = np.random.default_rng(rng_seed)
            idx_valid = np.where(valid_xy)[0]
            bx = np.empty(B, dtype=float)
            by = np.empty(B, dtype=float)
            br = np.empty(B, dtype=float)
            bch = np.empty(B, dtype=float)
            for b in range(B):
                ii = rng.choice(idx_valid, size=idx_valid.size, replace=True)
                bx[b] = float(np.nanmedian(x[ii]))
                by[b] = float(np.nanmedian(y[ii]))
                br[b] = float(np.nanmean(r[ii]))
                bch[b] = float(np.nanmean(ch[ii]))
            row.update(
                {
                    "landscape_1_boot_q05": float(np.nanquantile(bx, qlo)),
                    "landscape_1_boot_q95": float(np.nanquantile(bx, qhi)),
                    "landscape_2_boot_q05": float(np.nanquantile(by, qlo)),
                    "landscape_2_boot_q95": float(np.nanquantile(by, qhi)),
                    "risk_H_boot_q05": float(np.nanquantile(br, qlo)),
                    "risk_H_boot_q95": float(np.nanquantile(br, qhi)),
                    "cumhaz_H_boot_q05": float(np.nanquantile(bch, qlo)),
                    "cumhaz_H_boot_q95": float(np.nanquantile(bch, qhi)),
                }
            )
        else:
            row.update(
                {
                    "landscape_1_boot_q05": np.nan,
                    "landscape_1_boot_q95": np.nan,
                    "landscape_2_boot_q05": np.nan,
                    "landscape_2_boot_q95": np.nan,
                    "risk_H_boot_q05": np.nan,
                    "risk_H_boot_q95": np.nan,
                    "cumhaz_H_boot_q05": np.nan,
                    "cumhaz_H_boot_q95": np.nan,
                }
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["risk_cohort_order", "horizon_months"]).reset_index(drop=True)


def _compute_late_endpoint_fc_feature_signatures(dscope, X_raw, feat_names, outdir, fc_scope, fc_group, projection_method, treatment_profile=None):
    """Annotate late-horizon trajectory endpoints with FC-feature signatures.

    Endpoint neighborhoods are selected in the full standardized patient-horizon
    state space saved in `state_for_distance_*` columns, not in the displayed
    PCA/UMAP coordinates. The displayed projection is used only for plotting and
    for reporting the endpoint coordinate in the figure.
    """
    if dscope is None or len(dscope) == 0:
        return pd.DataFrame(), pd.DataFrame(), "empty_scope"
    d = _assign_scope_risk_cohorts(dscope).copy()
    endpoint_H = float(RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_H)
    hvals = pd.to_numeric(d["horizon_months"], errors="coerce").to_numpy(float)
    d_end = d[np.isclose(hvals, endpoint_H)].copy()
    if len(d_end) < 6:
        return pd.DataFrame(), pd.DataFrame(), "too_few_late_horizon_patients"

    state_cols = [c for c in d_end.columns if str(c).startswith("state_for_distance_")]
    if len(state_cols) == 0:
        return pd.DataFrame(), pd.DataFrame(), "missing_full_state_distance_columns"
    state_mat = d_end[state_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pid_vec = pd.to_numeric(d_end["test_index"], errors="coerce").to_numpy(float)
    ok_state = np.isfinite(pid_vec) & np.all(np.isfinite(state_mat), axis=1)
    d_end = d_end.loc[ok_state].copy()
    state_mat = state_mat[ok_state, :]
    if len(d_end) < 6 or state_mat.shape[0] < 6:
        return pd.DataFrame(), pd.DataFrame(), "too_few_finite_late_horizon_state_vectors"

    cent = _compute_risk_cohort_horizon_centroids(d)
    cent = cent[np.isclose(pd.to_numeric(cent.get("horizon_months"), errors="coerce").to_numpy(float), endpoint_H)].copy() if len(cent) else pd.DataFrame()
    if cent.empty:
        return pd.DataFrame(), pd.DataFrame(), "no_late_horizon_trajectory_endpoints"

    fc_tab = _top_fc_feature_table_for_scope(
        outdir,
        scope=fc_scope,
        group=fc_group,
        topn=RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_TOP_FC_FEATURES,
        treatment_profile=treatment_profile,
    )
    if fc_tab.empty:
        return pd.DataFrame(), pd.DataFrame(), "no_FC_features_available"
    _, used_features, _ = _fc_feature_signature_matrix(X_raw, feat_names, fc_tab)
    if len(used_features) == 0:
        return pd.DataFrame(), pd.DataFrame(), "no_matched_FC_features_in_design"
    fc_lookup = fc_tab.set_index(fc_tab["feature"].astype(str)).to_dict(orient="index")
    fc_source = str(fc_tab.get("FC_annotation_scope", pd.Series(["unknown"])).iloc[0]) if "FC_annotation_scope" in fc_tab.columns and len(fc_tab) else "unknown"
    fc_rank_source = str(fc_tab.get("FC_ranking_source", pd.Series([""])).iloc[0]) if "FC_ranking_source" in fc_tab.columns and len(fc_tab) else ""
    fc_matched_context_ids = str(fc_tab.get("FC_matched_context_ids", pd.Series([""])).iloc[0]) if "FC_matched_context_ids" in fc_tab.columns and len(fc_tab) else ""

    X = np.asarray(X_raw, float)
    feat_names = [str(f) for f in feat_names]
    fmap = {str(f): j for j, f in enumerate(feat_names)}
    all_terminal_pids = pd.to_numeric(d_end["test_index"], errors="coerce").dropna().astype(int).unique()
    all_terminal_pids = all_terminal_pids[(all_terminal_pids >= 0) & (all_terminal_pids < X.shape[0])]
    if all_terminal_pids.size < 6:
        return pd.DataFrame(), pd.DataFrame(), "too_few_valid_late_horizon_patients"

    summary_rows = []
    feature_rows = []
    n_end = int(len(d_end))
    base_n_neighbors = int(np.ceil(float(RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_NEIGHBOR_FRAC) * n_end))
    base_n_neighbors = max(int(RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_MIN_NEIGHBORS), base_n_neighbors)
    base_n_neighbors = min(max(3, base_n_neighbors), n_end)
    qlo, qhi = RESISTANCE_PSEUDOTRAJECTORY_BOOTSTRAP_Q
    B_boot = int(RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_ENRICH_BOOTSTRAP_N)
    B_perm = int(RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_ENRICH_PERMUTATIONS)

    for _, c in cent.sort_values("risk_cohort_order").iterrows():
        cohort = str(c["risk_cohort"])
        ord_i = int(c["risk_cohort_order"])
        cohort_mask = pd.to_numeric(d_end["risk_cohort_order"], errors="coerce").fillna(-1).astype(int).to_numpy() == ord_i
        if int(np.sum(cohort_mask)) < 3:
            continue
        center_state = np.nanmedian(state_mat[cohort_mask, :], axis=0)
        if not np.all(np.isfinite(center_state)):
            continue
        dist = np.sqrt(np.sum((state_mat - center_state.reshape(1, -1)) ** 2, axis=1))
        dd = d_end.copy()
        dd["endpoint_distance"] = dist
        dd = dd[np.isfinite(dd["endpoint_distance"].to_numpy(float))].sort_values("endpoint_distance")
        if len(dd) < 3:
            continue
        n_neighbors = min(base_n_neighbors, len(dd))
        near = dd.head(n_neighbors).copy()
        pids_in = pd.to_numeric(near["test_index"], errors="coerce").dropna().astype(int).unique()
        pids_out = np.asarray([pid for pid in all_terminal_pids if int(pid) not in set(map(int, pids_in))], dtype=int)
        if pids_in.size < 3 or pids_out.size < 3:
            continue

        cx = float(pd.to_numeric(d_end.loc[cohort_mask, "landscape_1"], errors="coerce").median()) if "landscape_1" in d_end.columns else np.nan
        cy = float(pd.to_numeric(d_end.loc[cohort_mask, "landscape_2"], errors="coerce").median()) if "landscape_2" in d_end.columns else np.nan
        mean_risk = float(pd.to_numeric(near["risk_H"], errors="coerce").mean())
        event_frac = float(pd.to_numeric(near.get("observed_event_by_horizon"), errors="coerce").fillna(0).mean())
        radius = float(pd.to_numeric(near["endpoint_distance"], errors="coerce").max())
        summary_rows.append(
            {
                "projection_method": str(projection_method),
                "scope": str(fc_scope),
                "group": str(fc_group),
                "treatment_profile_for_FC_ranking": str(treatment_profile) if treatment_profile is not None else "ALL",
                "FC_annotation_scope": fc_source,
                "FC_ranking_source": fc_rank_source,
                "FC_matched_context_ids": fc_matched_context_ids,
                "risk_cohort_order": ord_i,
                "risk_cohort": cohort,
                "endpoint_horizon_months": float(endpoint_H),
                "n_near_endpoint": int(pids_in.size),
                "n_other_late_horizon": int(pids_out.size),
                "endpoint_landscape_1": cx,
                "endpoint_landscape_2": cy,
                "neighborhood_radius_full_state_space": radius,
                "neighborhood_distance_space": "full_standardized_patient_horizon_state",
                "mean_endpoint_neighborhood_risk": mean_risk,
                "event_fraction_by_endpoint_horizon": event_frac,
            }
        )
        endpoint_seed_base = int(
            RESISTANCE_PSEUDOTRAJECTORY_INFERENCE_SEED
            + 1000000 * max(ord_i, 0)
            + 1000 * int(round(endpoint_H))
            + (0 if str(projection_method).lower() == "pca" else 503)
        )
        for ff, f in enumerate(used_features):
            j = fmap.get(str(f), None)
            if j is None or j >= X.shape[1]:
                continue
            xin = X[pids_in, j]
            xout = X[pids_out, j]
            if np.sum(np.isfinite(xin)) < 3 or np.sum(np.isfinite(xout)) < 3:
                continue
            sd_all = float(np.nanstd(X[all_terminal_pids, j]))
            if not np.isfinite(sd_all) or sd_all <= 1e-8:
                sd_all = 1.0
            mu_in = float(np.nanmean(xin))
            mu_out = float(np.nanmean(xout))
            delta = mu_in - mu_out
            std_delta = delta / sd_all
            seed = endpoint_seed_base + 17 * int(ff)
            ci_lo, ci_hi, boot_sd = _bootstrap_mean_delta_ci(
                xin,
                xout,
                B=B_boot,
                q=(qlo, qhi),
                seed=seed,
            )
            p_perm = _permutation_mean_delta_pvalue(
                xin,
                xout,
                B=B_perm,
                seed=seed + 7919,
            )
            feature_rows.append(
                {
                    "projection_method": str(projection_method),
                    "scope": str(fc_scope),
                    "group": str(fc_group),
                    "risk_cohort_order": ord_i,
                    "risk_cohort": cohort,
                    "endpoint_horizon_months": float(endpoint_H),
                    "feature": str(f),
                    "feature_family": _feature_family_for_mechanism(f),
                    "treatment_profile_for_FC_ranking": str(treatment_profile) if treatment_profile is not None else "ALL",
                    "FC_annotation_scope": fc_source,
                    "FC_ranking_source": fc_rank_source,
                    "FC_matched_context_ids": fc_matched_context_ids,
                    "FC_score_horizon_mean": pd.to_numeric(pd.Series([fc_lookup.get(str(f), {}).get("FC_score_horizon_mean", np.nan)]), errors="coerce").iloc[0],
                    "FC_score_horizon_max": pd.to_numeric(pd.Series([fc_lookup.get(str(f), {}).get("FC_score_horizon_max", np.nan)]), errors="coerce").iloc[0],
                    "FC_horizons": str(fc_lookup.get(str(f), {}).get("FC_horizons", "")),
                    "FC_context_ids": str(fc_lookup.get(str(f), {}).get("FC_context_ids", "")),
                    "n_near_endpoint": int(pids_in.size),
                    "n_other_late_horizon": int(pids_out.size),
                    "mean_near_endpoint": mu_in,
                    "mean_other_late_horizon": mu_out,
                    "delta_near_endpoint_minus_other": delta,
                    "delta_boot_q05": ci_lo,
                    "delta_boot_q95": ci_hi,
                    "delta_boot_sd": boot_sd,
                    "permutation_p_two_sided": p_perm,
                    "standardized_delta_near_endpoint_minus_other": std_delta,
                    "abs_standardized_delta_near_endpoint_minus_other": abs(std_delta),
                    "neighborhood_distance_space": "full_standardized_patient_horizon_state",
                }
            )

    summary = pd.DataFrame(summary_rows)
    features = pd.DataFrame(feature_rows)
    if len(features):
        features["permutation_q_BH_within_endpoint"] = np.nan
        group_cols = ["projection_method", "scope", "group", "risk_cohort_order", "risk_cohort", "endpoint_horizon_months"]
        for _, idx in features.groupby(group_cols, dropna=False).groups.items():
            idx = list(idx)
            qv = _bh_fdr(pd.to_numeric(features.loc[idx, "permutation_p_two_sided"], errors="coerce").to_numpy(float))
            features.loc[idx, "permutation_q_BH_within_endpoint"] = qv
        features = features.sort_values(
            ["projection_method", "risk_cohort_order", "abs_standardized_delta_near_endpoint_minus_other"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        if len(summary):
            top_rows = []
            for keys, dg in features.groupby(["projection_method", "risk_cohort_order", "risk_cohort"], dropna=False):
                pos = dg.sort_values("standardized_delta_near_endpoint_minus_other", ascending=False).head(3)
                neg = dg.sort_values("standardized_delta_near_endpoint_minus_other", ascending=True).head(3)
                top_rows.append(
                    {
                        "projection_method": keys[0],
                        "risk_cohort_order": int(keys[1]),
                        "risk_cohort": str(keys[2]),
                        "top_enriched_FC_features": ";".join(pos["feature"].astype(str).tolist()),
                        "top_depleted_FC_features": ";".join(neg["feature"].astype(str).tolist()),
                    }
                )
            summary = summary.merge(pd.DataFrame(top_rows), on=["projection_method", "risk_cohort_order", "risk_cohort"], how="left")
    return summary, features, ""


def _plot_resistance_pseudotrajectory_3d(df, outpath, title, projection_label, annotation_text=""):
    """Plot a 3D horizon-ordered resistance trajectory landscape.

    x/y are a shared latent-only projection of the trained DGP shared/private
    latent representation. z is the
    prediction horizon. Lines connect low/medium/high risk cohort centroids
    across horizons.
    """
    if df is None or len(df) == 0:
        return
    d = _assign_scope_risk_cohorts(df)
    x = pd.to_numeric(d["landscape_1"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(d["landscape_2"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(d["horizon_months"], errors="coerce").to_numpy(float)
    r = pd.to_numeric(d["risk_H"], errors="coerce").to_numpy(float)
    evt = pd.to_numeric(d.get("observed_event_by_horizon", 0), errors="coerce").fillna(0).to_numpy(int) > 0
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(r)
    if int(np.sum(ok)) < 3:
        return
    d = d.loc[ok].copy()
    x, y, z, r, evt = x[ok], y[ok], z[ok], r[ok], evt[ok]
    vmin = max(0.0, float(np.nanquantile(r, 0.02))) if np.any(np.isfinite(r)) else 0.0
    vmax = min(1.0, float(np.nanquantile(r, 0.98))) if np.any(np.isfinite(r)) else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    fig = plt.figure(figsize=(9.6, 7.3))
    ax = fig.add_subplot(111, projection="3d")
    m0 = ~evt
    m1 = evt
    sc = ax.scatter(
        x[m0], y[m0], z[m0], c=r[m0], s=22, alpha=0.45, cmap="viridis", vmin=vmin, vmax=vmax,
        edgecolors="none", depthshade=True, label="No observed PFI event by horizon"
    )
    if np.any(m1):
        ax.scatter(
            x[m1], y[m1], z[m1], c=r[m1], s=48, alpha=0.85, cmap="viridis", vmin=vmin, vmax=vmax,
            marker="^", edgecolors="black", linewidths=0.45, depthshade=True, label="Observed PFI event by horizon"
        )

    cent = _compute_risk_cohort_horizon_centroids(d)
    if len(cent):
        for _, gc in cent.groupby("risk_cohort_order", dropna=False):
            gc = gc.sort_values("horizon_months")
            cohort = str(gc["risk_cohort"].iloc[0])
            if len(gc) < 2:
                continue
            ax.plot(
                gc["landscape_1"].to_numpy(float),
                gc["landscape_2"].to_numpy(float),
                gc["horizon_months"].to_numpy(float),
                linewidth=2.3,
                alpha=0.90,
                label=f"{cohort} trajectory",
            )
            ax.scatter(
                gc["landscape_1"].to_numpy(float),
                gc["landscape_2"].to_numpy(float),
                gc["horizon_months"].to_numpy(float),
                s=34,
                alpha=0.90,
                depthshade=False,
            )
            last = gc.iloc[-1]
            ax.text(
                float(last["landscape_1"]),
                float(last["landscape_2"]),
                float(last["horizon_months"]),
                cohort.replace(" risk", ""),
                fontsize=7,
            )

    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.08)
    cb.set_label("Predicted PFI-event probability at horizon")
    ax.set_title(title, loc="left", fontsize=13)
    ax.set_xlabel(f"DGP latent-risk landscape 1 ({projection_label})")
    ax.set_ylabel(f"DGP latent-risk landscape 2 ({projection_label})")
    ax.set_zlabel("Prediction horizon (months)")
    ax.view_init(elev=24, azim=42)
    ax.grid(alpha=0.20)
    ax.legend(frameon=True, fontsize=7.6, loc="upper left")
    if annotation_text:
        fig.text(
            0.78, 0.88, annotation_text, va="top", ha="left", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.95)
        )
        fig.tight_layout(rect=[0, 0, 0.76, 1])
    else:
        fig.tight_layout()
    savefig(outpath)


def _join_endpoint_features_with_fc(outdir, endpoint_feature_df):
    if endpoint_feature_df is None or endpoint_feature_df.empty:
        return pd.DataFrame()
    if "FC_score_horizon_mean" in endpoint_feature_df.columns:
        out = endpoint_feature_df.copy()
        fc_score = pd.to_numeric(out.get("FC_score_horizon_mean"), errors="coerce").fillna(0.0)
        enrich_score = pd.to_numeric(out.get("abs_standardized_delta_near_endpoint_minus_other"), errors="coerce").fillna(0.0)
        out["endpoint_FC_joint_score"] = fc_score * enrich_score
        return out
    fc_all = []
    scope_pairs = endpoint_feature_df[["scope", "group"]].drop_duplicates()
    for _, r in scope_pairs.iterrows():
        scope = str(r["scope"])
        group = str(r["group"])
        tab = _horizon_averaged_fc_table(outdir, scope=scope, group=group)
        if len(tab):
            tab = tab.copy()
            tab["scope"] = scope
            tab["group"] = group
            fc_all.append(tab)
    if not fc_all:
        out = endpoint_feature_df.copy()
        out["FC_score_horizon_mean"] = np.nan
        out["endpoint_FC_joint_score"] = np.nan
        return out
    fc = pd.concat(fc_all, axis=0, ignore_index=True)
    keep_cols = [c for c in ["scope", "group", "feature", "FC_score_horizon_mean", "FC_score_horizon_max", "FC_horizon_count", "FC_horizons", "feature_family"] if c in fc.columns]
    fc = fc.loc[:, keep_cols].copy()
    if "feature_family" in fc.columns and "feature_family" in endpoint_feature_df.columns:
        fc = fc.drop(columns=["feature_family"])
    merged = endpoint_feature_df.merge(fc, on=["scope", "group", "feature"], how="left")
    fc_score = pd.to_numeric(merged.get("FC_score_horizon_mean"), errors="coerce").fillna(0.0)
    enrich_score = pd.to_numeric(merged.get("abs_standardized_delta_near_endpoint_minus_other"), errors="coerce").fillna(0.0)
    merged["endpoint_FC_joint_score"] = fc_score * enrich_score
    return merged


def run_resistance_landscape_analysis(art, outdir):
    """Posthoc 3D resistance trajectory landscapes from the trained DGP.

    This block uses a latent-only projection of the trained DGP shared/private
    representation. Low/medium/high risk
    cohort trajectories are horizon-ordered centroid paths. Late endpoints are
    annotated with FC-prioritized feature signatures from nearby patients.
    """
    if not ENABLE_RESISTANCE_LANDSCAPES or not ENABLE_RESISTANCE_PSEUDOTRAJECTORY_3D:
        return None
    outdir = Path(outdir)
    land_dir = outdir / "resistance_trajectory_landscape_3d"
    land_dir.mkdir(parents=True, exist_ok=True)

    model = art["model"]
    Xte_t = art["Xte_t"]
    g_te_t = art["g_te_t"]
    g_te = np.asarray(art["g_te"], int)
    t_te = np.asarray(art["t_te"], float)
    e_te = np.asarray(art["e_te"], int)
    groups = [str(x) for x in art["groups"]]
    feat_names = [str(x) for x in art["feat_names"]]
    X_raw = np.asarray(art.get("Xte_np_sub"), float)
    if X_raw.ndim != 2 or X_raw.shape[0] != len(t_te):
        X_raw = Xte_t.detach().cpu().numpy()
    meta_te = art.get("meta_te")
    X_s = Xte_t.detach().cpu().numpy()

    H_lat = _latent_matrix_for_landscape(
        model, Xte_t, g_te_t, sample=bool(RESISTANCE_PSEUDOTRAJECTORY_LATENT_SAMPLE)
    )
    A_tx, tx_cols = _treatment_profile_from_metadata_or_design(meta_te, X_s, feat_names)
    tx_labels = _make_treatment_profile_labels(A_tx, tx_cols)

    patient_df = pd.DataFrame(
        {
            "test_index": np.arange(len(t_te), dtype=int),
            "PFI_TIME_MONTHS": t_te,
            "PFI_EVENT": e_te,
            "group_idx": g_te,
            "group": [groups[int(g)] if int(g) < len(groups) else f"group{int(g)}" for g in g_te],
            "treatment_profile": tx_labels,
        }
    )
    if meta_te is not None and len(meta_te) == len(patient_df):
        meta_work = meta_te.reset_index(drop=True).copy()
        for col in ["SAMPLE_KEY", "PATIENT_ID", "CANCER_CODE"]:
            if col in meta_work.columns:
                patient_df[col] = meta_work[col].astype(str).to_numpy()

    tx_count_table = _make_treatment_profile_count_table(patient_df)
    if len(tx_count_table):
        tx_count_table.to_csv(land_dir / "treatment_profile_counts_by_cancer.csv", index=False)

    risk_versions = {}
    risk_cols = []
    for H in [float(h) for h in RESISTANCE_PSEUDOTRAJECTORY_H_LIST]:
        H_int = int(H)
        risk, risk_version = _landscape_risk_vector(art, outdir, H)
        col = f"risk_H{H_int}"
        patient_df[col] = np.asarray(risk, float)
        patient_df[f"risk_version_H{H_int}"] = str(risk_version)
        patient_df[f"observed_event_by_H{H_int}"] = _observed_event_by_horizon(t_te, e_te, H)
        risk_versions[col] = str(risk_version)
        risk_cols.append(col)
    if risk_cols:
        patient_df["risk_mean_across_horizons"] = patient_df[risk_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        patient_df["risk_max_across_horizons"] = patient_df[risk_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    else:
        patient_df["risk_mean_across_horizons"] = np.nan
        patient_df["risk_max_across_horizons"] = np.nan
    anchor_vals, anchor_cols = _risk_anchor_values(patient_df)
    patient_df["risk_anchor_value"] = anchor_vals
    patient_df["risk_anchor_columns"] = ";".join(anchor_cols)

    horizon_df, state_matrix, projection_info = _build_patient_horizon_landscape_dataframe(
        patient_df=patient_df,
        H_lat=H_lat,
        X_raw=X_raw,
        feat_names=feat_names,
        outdir=outdir,
    )

    readme = {
        "analysis": "DGP resistance trajectory landscape with late-horizon FC feature annotation",
        "coordinate_source": "latent-only PCA/UMAP projection of the trained DGP shared/private latent representation; risk is used only for color/stratification and prediction horizon is used only as the z-axis",
        "projection": "both PCA and UMAP versions are saved from the same trained-DGP latent matrix; UMAP falls back to PCA if unavailable",
        "z_axis": "prediction horizon in months",
        "trajectory_definition": "low/medium/high risk trajectories are within-scope cohort centroid paths ordered across prediction horizons",
        "risk_cohort_definition": "within each plotted scope, low/medium/high cohorts are tertiles of mean predicted PFI-event risk over the anchor horizons",
        "risk_group_anchor_horizons_months": [float(h) for h in RESISTANCE_PSEUDOTRAJECTORY_STRATIFY_H_LIST],
        "endpoint_feature_annotation": "late-horizon trajectory endpoints are annotated by FC-prioritized features enriched among nearby patients selected by standardized trained-DGP latent distance, not by risk, cumulative hazard, horizon, or 2D PCA/UMAP display distance",
        "risk_color": "horizon-specific predicted PFI-event probability; calibrated risks are used when available",
        "treatment_conditioning": "exact treatment-profile panels are non-overlapping: each patient belongs to exactly one complete treatment-profile cell. Multi-treatment profiles are not averaged from single-treatment contexts; endpoint annotations use exact-profile treatment-adjusted FC rankings when available. Multi-treatment panels require larger sample/event support.",
        "min_n": int(RESISTANCE_PSEUDOTRAJECTORY_MIN_N),
        "min_events": int(RESISTANCE_PSEUDOTRAJECTORY_MIN_EVENTS),
        "min_n_per_cohort_horizon_centroid": int(RESISTANCE_PSEUDOTRAJECTORY_MIN_BIN_N),
        "horizons_months": [float(h) for h in RESISTANCE_PSEUDOTRAJECTORY_H_LIST],
        "endpoint_horizon_months": float(RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_H),
        "projection_methods": [str(m).lower() for m in RESISTANCE_PSEUDOTRAJECTORY_PROJECTION_METHODS],
        "risk_versions": risk_versions,
        "centroid_bootstrap_n": int(RESISTANCE_PSEUDOTRAJECTORY_CENTROID_BOOTSTRAP_N),
        "endpoint_feature_bootstrap_n": int(RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_ENRICH_BOOTSTRAP_N),
        "endpoint_feature_permutations": int(RESISTANCE_PSEUDOTRAJECTORY_ENDPOINT_ENRICH_PERMUTATIONS),
        "endpoint_feature_fdr": "Benjamini-Hochberg within each endpoint/cohort feature set",
        "projection_info": projection_info,
    }
    with open(land_dir / "README_resistance_trajectory_landscape_3d.json", "w", encoding="utf-8") as f:
        json.dump(readme, f, indent=2)

    scope_rows = []
    centroid_rows = []
    endpoint_summary_rows = []
    endpoint_feature_rows = []
    endpoint_audit_rows = []

    methods = [str(m).lower() for m in RESISTANCE_PSEUDOTRAJECTORY_PROJECTION_METHODS]
    methods = list(dict.fromkeys([m for m in methods if m in {"pca", "umap"}])) or ["pca"]

    def _handle_scope(mask_patient, scope, group, treatment_profile, filename_stub, title_base, fc_scope="pooled", fc_group="pooled", fc_treatment_profile=None):
        mask_patient = np.asarray(mask_patient, bool)
        n = int(np.sum(mask_patient))
        ev = int(np.sum(e_te[mask_patient])) if n else 0
        if "exact_treatment_profile" in str(scope):
            min_n_scope, min_events_scope = _exact_profile_landscape_min_support(treatment_profile)
        else:
            min_n_scope, min_events_scope = int(RESISTANCE_PSEUDOTRAJECTORY_MIN_N), int(RESISTANCE_PSEUDOTRAJECTORY_MIN_EVENTS)
        skip_reason = ""
        if n < int(min_n_scope):
            skip_reason = f"n<{int(min_n_scope)}"
        elif ev < int(min_events_scope):
            skip_reason = f"events<{int(min_events_scope)}"
        scope_rows.append(
            {
                "scope": str(scope),
                "group": str(group),
                "treatment_profile": str(treatment_profile),
                "n_patients": n,
                "PFI_events_total": ev,
                "min_n_required": int(min_n_scope),
                "min_events_required": int(min_events_scope),
                "treatment_active_count": int(_treatment_profile_active_count(treatment_profile)),
                "is_multi_treatment_profile": bool(_is_multi_treatment_profile(treatment_profile)),
                "skipped": bool(skip_reason),
                "skip_reason": skip_reason,
                "projection_methods": ";".join(methods),
                "FC_ranking_treatment_profile": str(fc_treatment_profile) if fc_treatment_profile is not None else "ALL",
                "uses_treatment_specific_FC_annotation": bool(fc_treatment_profile is not None),
            }
        )
        if skip_reason:
            return
        patient_ids = np.where(mask_patient)[0]
        dscope_base = horizon_df[horizon_df["test_index"].isin(patient_ids)].copy()
        for method in methods:
            lx = f"landscape_1_{method}"
            ly = f"landscape_2_{method}"
            if lx not in dscope_base.columns or ly not in dscope_base.columns:
                continue
            dscope = dscope_base.copy()
            dscope["landscape_1"] = pd.to_numeric(dscope[lx], errors="coerce")
            dscope["landscape_2"] = pd.to_numeric(dscope[ly], errors="coerce")
            dscope = _assign_scope_risk_cohorts(dscope)
            proj_label = method.upper()
            title = f"{title_base} — {proj_label}"
            _plot_resistance_pseudotrajectory_3d(
                dscope,
                land_dir / f"ResistanceTrajectoryLandscape3D_{filename_stub}_{proj_label}.png",
                title,
                projection_label=proj_label,
                annotation_text=_landscape_fc_avg_top_text(
                    outdir,
                    scope=fc_scope,
                    group=fc_group,
                    treatment_profile=fc_treatment_profile,
                ),
            )
            cent = _compute_risk_cohort_horizon_centroids(dscope)
            if len(cent):
                cent.insert(0, "projection_method", method)
                cent.insert(1, "scope", str(scope))
                cent.insert(2, "group", str(group))
                cent.insert(3, "treatment_profile", str(treatment_profile))
                centroid_rows.append(cent)
            endpoint_summary, endpoint_features, endpoint_skip = _compute_late_endpoint_fc_feature_signatures(
                dscope=dscope,
                X_raw=X_raw,
                feat_names=feat_names,
                outdir=outdir,
                fc_scope=fc_scope,
                fc_group=fc_group,
                projection_method=method,
                treatment_profile=fc_treatment_profile,
            )
            endpoint_audit_rows.append(
                {
                    "scope": str(scope),
                    "group": str(group),
                    "treatment_profile": str(treatment_profile),
                    "projection_method": method,
                    "endpoint_annotation_skipped": bool(endpoint_skip),
                    "endpoint_annotation_skip_reason": str(endpoint_skip),
                }
            )
            if len(endpoint_summary):
                endpoint_summary.insert(0, "plot_scope", str(scope))
                endpoint_summary.insert(1, "plot_group", str(group))
                endpoint_summary.insert(2, "treatment_profile", str(treatment_profile))
                endpoint_summary_rows.append(endpoint_summary)
            if len(endpoint_features):
                endpoint_features.insert(0, "plot_scope", str(scope))
                endpoint_features.insert(1, "plot_group", str(group))
                endpoint_features.insert(2, "treatment_profile", str(treatment_profile))
                endpoint_feature_rows.append(endpoint_features)

    # Pooled and cancer-specific trajectory panels.
    m_all = np.ones(len(patient_df), dtype=bool)
    _handle_scope(
        m_all,
        scope="pooled",
        group="pooled",
        treatment_profile="ALL",
        filename_stub="Pooled",
        title_base="Pooled DGP resistance trajectory landscape",
        fc_scope="pooled",
        fc_group="pooled",
    )
    for gi, gname in enumerate(groups):
        m = g_te == int(gi)
        _handle_scope(
            m,
            scope="group",
            group=str(gname),
            treatment_profile="ALL",
            filename_stub=f"{_safe_fname(gname)}",
            title_base=f"{gname}: DGP resistance trajectory landscape",
            fc_scope="group_specific",
            fc_group=str(gname),
        )

    # Exact treatment-combination extensions: non-overlapping, only if adequate.
    for tx_lab in sorted(pd.Index(patient_df["treatment_profile"].astype(str)).unique().tolist()):
        mtx = patient_df["treatment_profile"].astype(str).to_numpy() == str(tx_lab)
        _handle_scope(
            mtx,
            scope="pooled_exact_treatment_profile",
            group="pooled",
            treatment_profile=str(tx_lab),
            filename_stub=f"Pooled_{_safe_fname(tx_lab)}",
            title_base=f"Pooled DGP resistance trajectory landscape: {tx_lab}",
            fc_scope="pooled_exact_treatment_profile",
            fc_group="pooled",
            fc_treatment_profile=str(tx_lab),
        )
        for gi, gname in enumerate(groups):
            mg = g_te == int(gi)
            mgtx = mtx & mg
            _handle_scope(
                mgtx,
                scope="group_exact_treatment_profile",
                group=str(gname),
                treatment_profile=str(tx_lab),
                filename_stub=f"{_safe_fname(gname)}_{_safe_fname(tx_lab)}",
                title_base=f"{gname} DGP resistance trajectory landscape: {tx_lab}",
                fc_scope="group_exact_treatment_profile",
                fc_group=str(gname),
                fc_treatment_profile=str(tx_lab),
            )

    # Save all coordinate and interpretation artifacts.
    patient_df.to_csv(land_dir / "resistance_trajectory_patient_risk_table.csv", index=False)
    horizon_df.to_csv(land_dir / "resistance_trajectory_patient_horizon_coordinates.csv", index=False)
    pd.DataFrame(scope_rows).to_csv(land_dir / "resistance_trajectory_scope_counts.csv", index=False)
    if endpoint_audit_rows:
        pd.DataFrame(endpoint_audit_rows).to_csv(land_dir / "resistance_trajectory_endpoint_annotation_audit.csv", index=False)
    if centroid_rows:
        pd.concat(centroid_rows, axis=0, ignore_index=True).to_csv(
            land_dir / "resistance_trajectory_risk_cohort_horizon_centroids.csv", index=False
        )
    if endpoint_summary_rows:
        pd.concat(endpoint_summary_rows, axis=0, ignore_index=True).to_csv(
            land_dir / "resistance_trajectory_late_endpoint_FC_signature_summary.csv", index=False
        )
    if endpoint_feature_rows:
        endpoint_feat = pd.concat(endpoint_feature_rows, axis=0, ignore_index=True)
        endpoint_feat.to_csv(land_dir / "resistance_trajectory_late_endpoint_FC_feature_enrichment.csv", index=False)
        joined = _join_endpoint_features_with_fc(outdir, endpoint_feat)
        joined.to_csv(land_dir / "resistance_trajectory_late_endpoint_features_with_FC.csv", index=False)
        top_joined = joined.copy()
        sort_col = "endpoint_FC_joint_score" if "endpoint_FC_joint_score" in top_joined.columns else "abs_standardized_delta_near_endpoint_minus_other"
        top_joined = top_joined.sort_values(
            ["plot_scope", "plot_group", "treatment_profile", "projection_method", "risk_cohort_order", sort_col],
            ascending=[True, True, True, True, True, False],
        )
        top_joined.groupby(
            ["plot_scope", "plot_group", "treatment_profile", "projection_method", "risk_cohort"],
            dropna=False,
        ).head(int(RESISTANCE_PSEUDOTRAJECTORY_TOP_FEATURES)).to_csv(
            land_dir / "resistance_trajectory_top_late_endpoint_features.csv",
            index=False,
        )

    # Save horizon-averaged FC summaries for pooled and cancer-specific scopes.
    fc_avg_rows = []
    tab = _horizon_averaged_fc_table(outdir, scope="pooled", group="pooled")
    if len(tab):
        fc_avg_rows.append(tab)
    for gname in groups:
        tab = _horizon_averaged_fc_table(outdir, scope="group_specific", group=str(gname))
        if len(tab):
            fc_avg_rows.append(tab)
    if fc_avg_rows:
        pd.concat(fc_avg_rows, axis=0, ignore_index=True).to_csv(
            land_dir / "resistance_trajectory_horizon_averaged_treatment_conditioned_FC.csv",
            index=False,
        )
    return horizon_df


def plot_risk_uq_violin(
    model, Xte_t, g_te_t, groups, alpha_dt, dt_edges, outdir, H_list=RISK_UQ_H_LIST
):
    for H in H_list:
        SH = posterior_survival_at_H(
            model, Xte_t, g_te_t, float(H), alpha_dt, dt_edges, S=RISK_UQ_MC
        )
        R = -np.log(np.clip(SH, 1e-12, 1.0))
        data = []
        labels = []
        g_np = g_te_t.detach().cpu().numpy().astype(int)
        for gi, gname in enumerate(groups):
            m = g_np == gi
            if not np.any(m):
                continue
            vals = R[:, m].ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(str(gname))
        if len(data) == 0:
            continue
        fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(data)), 5.3))
        parts = ax.violinplot(
            data, showmeans=False, showmedians=True, showextrema=False
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.22)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(r"Posterior risk  $-\log S(H)$")
        ax.set_title(f"Test-set posterior risk distributions at H={int(H)} months")
        ax.grid(True, axis="y", alpha=0.18)
        savefig(Path(outdir) / f"risk_uq_violin_test_H{int(H)}.png")


def plot_representative_patient_risk_trajectories(
    model,
    Xte_t,
    g_te_t,
    t_te,
    e_te,
    groups,
    alpha_dt,
    dt_edges,
    outdir,
    group_cindex_map,
    n_per_group=REPRESENTATIVE_PATIENTS_N,
):
    g_np = g_te_t.detach().cpu().numpy().astype(int)
    rows = []
    for gi, gname in enumerate(groups):
        m = np.where(g_np == gi)[0]
        if m.size == 0:
            continue
        sel = m[: min(int(n_per_group), m.size)]
        for idx in sel:
            rows.append(
                {
                    "group": str(gname),
                    "test_index": int(idx),
                    "group_c_index": float(group_cindex_map.get(str(gname), np.nan)),
                    "event_time_months": float(t_te[idx]),
                    "event": int(e_te[idx]),
                }
            )
    if len(rows) == 0:
        return
    rep = pd.DataFrame(rows)
    H_max = float(REPRESENTATIVE_TRAJ_H_MAX)
    H_grid = np.arange(0.0, H_max + 1e-9, float(REPRESENTATIVE_TRAJ_STEP))
    H_grid = np.unique(np.clip(H_grid, 0.0, float(H_max)))
    if H_grid.size == 0 or H_grid[0] != 0.0:
        H_grid = np.r_[0.0, H_grid]
    fig, axes = plt.subplots(
        nrows=len(rep), ncols=1, figsize=(9.0, max(3.2 * len(rep), 4.4)), sharex=True
    )
    if len(rep) == 1:
        axes = [axes]
    long_rows = []
    for ax, (_, rr) in zip(axes, rep.iterrows()):
        idx = int(rr["test_index"])
        curve_rows = []
        for H in H_grid:
            SH = posterior_survival_at_H(
                model,
                Xte_t[idx : idx + 1],
                g_te_t[idx : idx + 1],
                float(H),
                alpha_dt,
                dt_edges,
                S=int(REPRESENTATIVE_TRAJ_MC),
            )
            risk = -np.log(np.clip(SH[:, 0], 1e-12, 1.0))
            curve_rows.append(
                {
                    "H_months": float(H),
                    "risk_mean": float(np.mean(risk)),
                    "risk_q_lo": float(np.quantile(risk, 0.10)),
                    "risk_q_hi": float(np.quantile(risk, 0.90)),
                }
            )
        curve = pd.DataFrame(curve_rows)
        for _, crow in curve.iterrows():
            rec = rr.to_dict()
            rec.update(crow.to_dict())
            long_rows.append(rec)
        h_vals = _np1d(curve["H_months"])
        mu = _np1d(curve["risk_mean"])
        lo = _np1d(curve["risk_q_lo"])
        hi = _np1d(curve["risk_q_hi"])
        ax.plot(h_vals, mu, lw=2.2, color="#2F4B7C")
        ax.fill_between(h_vals, lo, hi, color="#4C78A8", alpha=0.22)
        if float(H_max) >= 60.0:
            ax.axvline(60.0, ls="--", lw=1.0, color="gray", alpha=0.75)
        if float(H_max) >= 120.0:
            ax.axvline(120.0, ls=":", lw=1.0, color="gray", alpha=0.75)
        ax.set_title(
            f"{rr['group']} | test C-index={rr['group_c_index']:.3f} | representative test patient #{idx}"
        )
        ax.set_ylabel(r"Risk  $-\log S(H)$")
        ax.grid(True, alpha=0.20)
    axes[-1].set_xlabel("Horizon (months)")
    fig.suptitle(
        f"Representative patient posterior risk trajectories (test set, H <= {int(round(H_max))} months)",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(Path(outdir) / "representative_patient_risk_trajectories_test.png")
    pd.DataFrame(long_rows).to_csv(
        Path(outdir) / "representative_patient_risk_trajectories_test.csv", index=False
    )


def posterior_risk_samples_at_H(
    model, X, g_idx, H, alpha_dt, dt_edges, S=FINAL_EVAL_MC_SAMPLES
):
    SH = posterior_survival_at_H(
        model, X, g_idx, float(H), alpha_dt, dt_edges, S=int(S)
    )
    return -np.log(np.clip(SH, 1e-12, 1.0))


def posterior_event_prob_samples_at_H(
    model, X, g_idx, H, alpha_dt, dt_edges, S=FINAL_EVAL_MC_SAMPLES
):
    SH = posterior_survival_at_H(
        model, X, g_idx, float(H), alpha_dt, dt_edges, S=int(S)
    )
    return 1.0 - np.clip(SH, 0.0, 1.0)


def _event_probability_mean_at_H(
    model, X, g_idx, H, alpha_dt, dt_edges, S=FINAL_EVAL_MC_SAMPLES
):
    """Posterior mean event probability P(T <= H).

    This wrapper is used by the post-hoc recalibration block. It returns one
    mean probability per patient, averaging over posterior predictive draws.
    """
    P = posterior_event_prob_samples_at_H(
        model, X, g_idx, float(H), alpha_dt, dt_edges, S=int(S)
    )
    return np.mean(np.clip(P, 0.0, 1.0), axis=0)


def export_posterior_risk_summary(
    model,
    Xte_t,
    g_te_t,
    t_te,
    e_te,
    groups,
    alpha_dt,
    dt_edges,
    outdir,
    H_list=RISK_UQ_H_LIST,
):
    g_np = g_te_t.detach().cpu().numpy().astype(int)
    rows = []
    for H in [float(h) for h in H_list]:
        SH = posterior_survival_at_H(
            model, Xte_t, g_te_t, H, alpha_dt, dt_edges, S=RISK_UQ_MC
        )
        R = -np.log(np.clip(SH, 1e-12, 1.0))
        P = 1.0 - np.clip(SH, 0.0, 1.0)
        risk_mean = np.mean(R, axis=0)
        risk_sd = np.std(R, axis=0)
        risk_q_lo = np.quantile(R, UQ_INTERVAL_Q[0], axis=0)
        risk_q_hi = np.quantile(R, UQ_INTERVAL_Q[1], axis=0)
        prob_mean = np.mean(P, axis=0)
        prob_sd = np.std(P, axis=0)
        prob_q_lo = np.quantile(P, UQ_INTERVAL_Q[0], axis=0)
        prob_q_hi = np.quantile(P, UQ_INTERVAL_Q[1], axis=0)
        for i in range(len(t_te)):
            rows.append(
                {
                    "test_index": int(i),
                    "group_idx": int(g_np[i]),
                    "group": (
                        str(groups[g_np[i]])
                        if 0 <= int(g_np[i]) < len(groups)
                        else str(g_np[i])
                    ),
                    "event_time_months": float(t_te[i]),
                    "event": int(e_te[i]),
                    "H_months": float(H),
                    "risk_mean": float(risk_mean[i]),
                    "risk_sd": float(risk_sd[i]),
                    "risk_q_lo": float(risk_q_lo[i]),
                    "risk_q_hi": float(risk_q_hi[i]),
                    "risk_ci_width": float(risk_q_hi[i] - risk_q_lo[i]),
                    "event_prob_mean": float(prob_mean[i]),
                    "event_prob_sd": float(prob_sd[i]),
                    "event_prob_q_lo": float(prob_q_lo[i]),
                    "event_prob_q_hi": float(prob_q_hi[i]),
                    "event_prob_ci_width": float(prob_q_hi[i] - prob_q_lo[i]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(Path(outdir) / "test_posterior_risk_summary.csv", index=False)
    return df


def plot_representative_patient_posterior_risk_distributions(
    model,
    Xte_t,
    g_te_t,
    t_te,
    e_te,
    groups,
    alpha_dt,
    dt_edges,
    outdir,
    n_per_group=UQ_REPRESENTATIVE_DENSITIES_PER_GROUP,
    H_list=RISK_UQ_H_LIST,
    S=RISK_UQ_MC,
):
    """Plot representative posterior risk densities with standardized axes.

    For each horizon, every representative-patient histogram uses the same
    x-axis limits, the same bin edges, and the same density y-axis limit.
    The standardization is performed within each horizon, not across all
    horizons, because posterior risk naturally increases with H.
    """
    chosen = []
    g_np = g_te_t.detach().cpu().numpy().astype(int)
    for gi, gname in enumerate(groups):
        idxs = np.where(g_np == gi)[0]
        for idx in idxs[: int(n_per_group)]:
            chosen.append((str(gname), int(idx)))
    if len(chosen) == 0:
        return
    draw_rows = []
    axis_rows = []
    for H in [float(h) for h in H_list]:
        R = posterior_risk_samples_at_H(
            model, Xte_t, g_te_t, H, alpha_dt, dt_edges, S=S
        )
        if R.ndim != 2 or R.shape[1] == 0:
            continue

        draws_by_idx = {}
        all_draws = []
        for gname, idx in chosen:
            if idx >= R.shape[1]:
                continue
            draws = np.asarray(R[:, idx], float)
            draws = draws[np.isfinite(draws)]
            if draws.size >= 3:
                draws_by_idx[int(idx)] = draws
                all_draws.append(draws)
        if len(all_draws) == 0:
            continue

        all_draws_flat = np.concatenate(all_draws)
        x_min = float(np.min(all_draws_flat))
        x_max = float(np.max(all_draws_flat))
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            continue
        width = max(x_max - x_min, 1e-6)
        x_lo = max(0.0, x_min - 0.04 * width)
        x_hi = x_max + 0.08 * width
        if x_hi <= x_lo:
            x_hi = x_lo + 1e-6

        n_bins = int(min(40, max(18, np.sqrt(int(S)))))
        bins = np.linspace(x_lo, x_hi, n_bins)

        # Use the same density y-axis within a horizon so histogram heights are
        # comparable across representative patients. Compute the maximum bin
        # density before plotting rather than letting Matplotlib rescale panels.
        y_max = 0.0
        for draws in draws_by_idx.values():
            hist_density, _ = np.histogram(draws, bins=bins, density=True)
            finite_density = hist_density[np.isfinite(hist_density)]
            if finite_density.size:
                y_max = max(y_max, float(np.max(finite_density)))
        y_hi = 1.08 * y_max if np.isfinite(y_max) and y_max > 0.0 else 1.0

        axis_rows.append(
            {
                "H_months": float(H),
                "x_lo": float(x_lo),
                "x_hi": float(x_hi),
                "y_lo": 0.0,
                "y_hi": float(y_hi),
                "n_representative_patients": int(len(draws_by_idx)),
                "S": int(S),
                "density_normalized": True,
                "axis_standardization": "shared x and shared density y within horizon",
            }
        )

        fig, axes = plt.subplots(
            len(chosen),
            1,
            figsize=(8.8, max(3.0 * len(chosen), 4.2)),
            sharex=True,
            sharey=True,
        )
        if len(chosen) == 1:
            axes = [axes]
        for ax, (gname, idx) in zip(axes, chosen):
            draws = draws_by_idx.get(int(idx))
            if draws is None:
                ax.axis("off")
                continue
            ax.hist(
                draws,
                bins=bins,
                density=True,
                alpha=0.30,
                edgecolor="white",
                linewidth=0.6,
            )
            mu = float(np.mean(draws))
            sd = float(np.std(draws))
            lo = float(np.quantile(draws, UQ_INTERVAL_Q[0]))
            hi = float(np.quantile(draws, UQ_INTERVAL_Q[1]))
            med = float(np.quantile(draws, 0.5))
            ax.axvline(mu, lw=1.4, color="black", ls="--", label="posterior mean")
            ax.axvline(
                med, lw=1.2, color="tab:orange", ls="-", label="posterior median"
            )
            ax.axvspan(
                lo,
                hi,
                alpha=0.14,
                color="tab:blue",
                label=f"{int(round((UQ_INTERVAL_Q[1]-UQ_INTERVAL_Q[0])*100))}% interval",
            )
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(0.0, y_hi)
            ax.set_ylabel("Posterior density")
            ax.set_title(
                f"{gname} | test patient #{idx} | mean={mu:.3f}, median={med:.3f}, sd={sd:.3f}, "
                f"{int(round((UQ_INTERVAL_Q[1]-UQ_INTERVAL_Q[0])*100))}% interval=[{lo:.3f}, {hi:.3f}]"
            )
            ax.grid(True, alpha=0.18)
            # Legend for this multi-panel figure is placed once outside the plotting area below.
            for d in draws:
                draw_rows.append(
                    {
                        "H_months": float(H),
                        "group": str(gname),
                        "test_index": int(idx),
                        "event_time_months": float(t_te[idx]),
                        "event": int(e_te[idx]),
                        "risk_draw": float(d),
                    }
                )
        axes[-1].set_xlabel(r"Posterior risk  $-\log S(H)$")
        handles, labels = axes[0].get_legend_handles_labels() if len(axes) else ([], [])
        if handles:
            fig.legend(
                handles,
                labels,
                frameon=True,
                loc="upper left",
                bbox_to_anchor=(1.01, 0.98),
                borderaxespad=0.0,
            )
        fig.suptitle(
            f"Representative patient posterior risk distributions @ H={int(H)} months",
            y=0.995,
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 0.84, 0.98])
        savefig(
            Path(outdir)
            / f"representative_patient_risk_distributions_test_H{int(H)}.png"
        )
    if len(draw_rows):
        pd.DataFrame(draw_rows).to_csv(
            Path(outdir) / "representative_patient_risk_distributions_test.csv",
            index=False,
        )
    if len(axis_rows):
        pd.DataFrame(axis_rows).to_csv(
            Path(outdir) / "representative_patient_risk_distribution_axis_limits.csv",
            index=False,
        )


def _plot_group_horizon_heatmap(
    df, value_col, title, outpath, fmt="%.3f", cmap="viridis"
):
    if not ENABLE_HEATMAPS:
        return
    if df is None or len(df) == 0 or value_col not in df.columns:
        return
    mat = (
        df.pivot(index="group", columns="H_months", values=value_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if mat.empty:
        return
    vals = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(6.0, 1.5 * mat.shape[1]), max(3.6, 0.8 * mat.shape[0] + 1.8))
    )
    im = ax.imshow(vals, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([str(int(round(float(x)))) for x in mat.columns], rotation=0)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([str(x) for x in mat.index])
    ax.set_xlabel("Horizon (months)")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = vals[i, j]
            label = "nan" if not np.isfinite(v) else (fmt % float(v))
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=(
                    "white"
                    if np.isfinite(v) and v > np.nanmedian(vals[np.isfinite(vals)])
                    else "black"
                ),
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(outpath)


def export_group_uncertainty_summary(risk_summary_df, outdir):
    if risk_summary_df is None or len(risk_summary_df) == 0:
        return None
    grp = (
        risk_summary_df.groupby(["group", "H_months"], dropna=False)
        .agg(
            n=("test_index", "count"),
            mean_risk=("risk_mean", "mean"),
            mean_risk_sd=("risk_sd", "mean"),
            median_risk_sd=("risk_sd", "median"),
            mean_risk_ci_width=("risk_ci_width", "mean"),
            median_risk_ci_width=("risk_ci_width", "median"),
            mean_event_prob=("event_prob_mean", "mean"),
            mean_event_prob_sd=("event_prob_sd", "mean"),
            mean_event_prob_ci_width=("event_prob_ci_width", "mean"),
        )
        .reset_index()
    )
    grp.to_csv(Path(outdir) / "test_group_uncertainty_summary.csv", index=False)
    _plot_group_horizon_heatmap(
        grp,
        "mean_risk_sd",
        "Mean posterior risk SD by group and horizon",
        Path(outdir) / "test_group_uncertainty_mean_risk_sd_heatmap.png",
    )
    _plot_group_horizon_heatmap(
        grp,
        "mean_risk_ci_width",
        f"Mean posterior risk {int(round((UQ_INTERVAL_Q[1]-UQ_INTERVAL_Q[0])*100))}% posterior interval width by group and horizon",
        Path(outdir) / "test_group_uncertainty_mean_risk_ci_width_heatmap.png",
    )
    return grp


def _quantile_bin_ids(prob, n_bins=UQ_CALIBRATION_BINS):
    prob = np.asarray(prob, float)
    prob = np.clip(prob, 0.0, 1.0)
    if prob.size == 0:
        return np.zeros((0,), dtype=int), np.array([0.0, 1.0], dtype=float)
    q = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    edges = np.quantile(prob, q)
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    bin_ids = np.digitize(prob, edges[1:-1], right=True)
    return bin_ids.astype(int), edges.astype(float)


def export_horizon_calibration_summary(
    model,
    Xte_t,
    g_te_t,
    t_te,
    e_te,
    groups,
    alpha_dt,
    dt_edges,
    outdir,
    H_list=POSTERIOR_H_LIST,
):
    g_np = g_te_t.detach().cpu().numpy().astype(int)
    rows = []
    plot_specs = []
    for H in [float(h) for h in H_list]:
        P = posterior_event_prob_samples_at_H(
            model, Xte_t, g_te_t, H, alpha_dt, dt_edges, S=RISK_UQ_MC
        )
        p_mean_all = np.mean(P, axis=0)
        valid, y_all, w_all = _ipcw_binary_event_targets(t_te, e_te, float(H))
        valid = np.asarray(valid, bool) & np.isfinite(p_mean_all)
        if int(np.sum(valid & (w_all > 0))) < 10:
            continue
        scope_specs = [("ALL", np.ones_like(g_np, dtype=bool))] + [
            (str(gname), g_np == gi) for gi, gname in enumerate(groups)
        ]
        for scope_name, mask_scope in scope_specs:
            m = valid & mask_scope & np.isfinite(w_all) & (w_all > 0)
            n = int(np.sum(m))
            if n < 10 or float(np.sum(w_all[m] * y_all[m])) <= 1e-8:
                continue
            p_mean = np.asarray(p_mean_all[m], float)
            y_use = np.asarray(y_all[m], int)
            w_use = np.asarray(w_all[m], float)
            bin_ids, edges = _quantile_bin_ids(p_mean, n_bins=UQ_CALIBRATION_BINS)
            for b in np.unique(bin_ids):
                mb = bin_ids == b
                if int(np.sum(mb)) == 0:
                    continue
                pred_mean = _weighted_mean_safe(p_mean[mb], w_use[mb])
                obs_rate = _weighted_mean_safe(y_use[mb], w_use[mb])
                rows.append(
                    {
                        "H_months": float(H),
                        "scope": str(scope_name),
                        "bin_id": int(b),
                        "n": int(np.sum(mb)),
                        "weight_sum": float(np.sum(w_use[mb])),
                        "pred_mean": (
                            float(pred_mean) if np.isfinite(pred_mean) else np.nan
                        ),
                        "obs_rate_ipcw": (
                            float(obs_rate) if np.isfinite(obs_rate) else np.nan
                        ),
                        "abs_calibration_error": (
                            float(abs(pred_mean - obs_rate))
                            if np.isfinite(pred_mean) and np.isfinite(obs_rate)
                            else np.nan
                        ),
                        "bin_left": float(edges[min(int(b), len(edges) - 2)]),
                        "bin_right": float(edges[min(int(b) + 1, len(edges) - 1)]),
                        "estimation_method": "IPCW fixed-horizon calibration",
                    }
                )
            plot_specs.append((float(H), str(scope_name)))
    df = pd.DataFrame(rows)
    df.to_csv(Path(outdir) / "test_horizon_calibration_summary.csv", index=False)
    if len(df):
        for H, scope_name in sorted(set(plot_specs)):
            sub = df[
                (np.isclose(pd.to_numeric(df["H_months"], errors="coerce"), float(H)))
                & (df["scope"].astype(str) == str(scope_name))
            ].copy()
            if len(sub) == 0:
                continue
            fig, ax = plt.subplots(figsize=(6.2, 5.0))
            x = sub["pred_mean"].to_numpy(dtype=float)
            yv = sub["obs_rate_ipcw"].to_numpy(dtype=float)
            n = sub["weight_sum"].to_numpy(dtype=float)
            ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="gray")
            ax.scatter(
                x,
                yv,
                s=np.clip(10.0 * np.sqrt(np.maximum(n, 1.0)), 40.0, 220.0),
                alpha=0.85,
            )
            ax.plot(x, yv, lw=1.5, alpha=0.8)
            ax.set_xlabel(f"Predicted event probability by H={int(H)} mo")
            ax.set_ylabel("Observed event probability (IPCW)")
            title_scope = "Overall" if str(scope_name) == "ALL" else str(scope_name)
            ax.set_title(f"IPCW calibration curve — {title_scope} — H={int(H)} mo")
            ax.grid(True, alpha=0.18)
            fig.tight_layout()
            safe_scope = _safe_fname(scope_name)
            savefig(Path(outdir) / f"calibration_curve_H{int(H)}__{safe_scope}.png")
    return df


def export_groupwise_coverage_summary(
    model,
    Xte_t,
    g_te_t,
    t_te,
    e_te,
    groups,
    alpha_dt,
    dt_edges,
    outdir,
    H_list=POSTERIOR_H_LIST,
):
    g_np = g_te_t.detach().cpu().numpy().astype(int)
    rows = []
    for H in [float(h) for h in H_list]:
        P = posterior_event_prob_samples_at_H(
            model, Xte_t, g_te_t, H, alpha_dt, dt_edges, S=RISK_UQ_MC
        )
        valid, y, w = _ipcw_binary_event_targets(t_te, e_te, float(H))
        valid = np.asarray(valid, bool) & np.isfinite(w)
        if int(np.sum(valid & (w > 0))) == 0:
            continue
        for scope_name, mask_scope in [("ALL", np.ones_like(g_np, dtype=bool))] + [
            (str(gname), g_np == gi) for gi, gname in enumerate(groups)
        ]:
            m_obs = valid & mask_scope & (w > 0)
            n = int(np.sum(mask_scope))
            n_obs = int(np.sum(m_obs))
            if n < 10 or n_obs < 10:
                continue
            obs_rate = _weighted_mean_safe(y[m_obs], w[m_obs])
            group_draws = np.mean(P[:, mask_scope], axis=1)
            q_lo = float(np.quantile(group_draws, UQ_INTERVAL_Q[0]))
            q_hi = float(np.quantile(group_draws, UQ_INTERVAL_Q[1]))
            rows.append(
                {
                    "H_months": float(H),
                    "scope": str(scope_name),
                    "n_group": n,
                    "n_observed_for_ipcw": n_obs,
                    "observed_event_rate_ipcw": (
                        float(obs_rate) if np.isfinite(obs_rate) else np.nan
                    ),
                    "posterior_group_mean_q_lo": q_lo,
                    "posterior_group_mean_q_hi": q_hi,
                    "posterior_group_mean_interval_width": float(q_hi - q_lo),
                    "covered": bool(
                        np.isfinite(obs_rate)
                        and (obs_rate >= q_lo)
                        and (obs_rate <= q_hi)
                    ),
                    "estimation_method": "diagnostic comparison of posterior group mean with IPCW observed event probability; not formal frequentist coverage",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(Path(outdir) / "test_groupwise_coverage_summary.csv", index=False)
    if len(df):
        _plot_group_horizon_heatmap(
            df[df["scope"].astype(str) != "ALL"].rename(columns={"scope": "group"}),
            "posterior_group_mean_interval_width",
            f"Groupwise posterior interval width ({int(round((UQ_INTERVAL_Q[1]-UQ_INTERVAL_Q[0])*100))}% interval)",
            Path(outdir) / "groupwise_coverage_interval_width_heatmap.png",
        )
    return df


def _private_latent_draw(model, x, g_idx, sample=True):
    g_idx = g_idx.to(device=x.device, dtype=torch.long).view(-1)
    h_private = torch.zeros(
        x.size(0), model.private_dim, device=x.device, dtype=x.dtype
    )
    for g in range(model.num_groups):
        idx = (g_idx == g).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        xg = x.index_select(0, idx)
        dist = model.private[g](xg)
        vals = dist.rsample() if sample else dist.mean
        pg = vals.transpose(-1, -2).contiguous()
        h_private.index_copy_(0, idx, pg)
    return h_private


def _output_score_from_latents(model, h, g_idx, sample=True):
    g_idx = g_idx.to(device=h.device, dtype=torch.long).view(-1)
    if isinstance(getattr(model, "output", None), torch.nn.ModuleList):
        s = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        for g in range(model.num_groups):
            idx = (g_idx == g).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            hg = h.index_select(0, idx)
            dist = model.output[g](hg)
            vals = dist.rsample() if sample else dist.mean
            vals = vals if vals.dim() == 1 else vals.squeeze(-1)
            s.index_copy_(0, idx, vals)
        return s
    f_all = model.output(h)
    vals = f_all.rsample() if sample else f_all.mean
    s = (
        vals
        if vals.dim() == 1
        else _select_task(vals, g_idx, num_tasks=model.num_groups)
    )
    if s.dim() == 2:
        s = s.mean(dim=0)
    return s


def _risk_from_score_tensor(score_t, dt_edges, alpha_dt, H):
    score_np = score_t.detach().cpu().numpy().astype(float)
    return risk_from_score_at_H(
        score_np, dt_edges, alpha_dt.detach().cpu().numpy(), float(H)
    )


def export_branch_uncertainty_decomposition(
    model,
    Xte_t,
    g_te_t,
    t_te,
    e_te,
    groups,
    alpha_dt,
    dt_edges,
    outdir,
    H_list=BRANCH_DECOMP_H_LIST,
    S=BRANCH_DECOMP_MC,
):
    g_np = g_te_t.detach().cpu().numpy().astype(int)
    h_shared_mean = model.shared_latent_mt(Xte_t, g_te_t, sample=False)
    h_private_mean = _private_latent_draw(model, Xte_t, g_te_t, sample=False)
    h_mean = torch.cat([h_shared_mean, h_private_mean], dim=-1).contiguous()
    total_rows = []
    note = (
        "Approximate conditional branch-attribution summary. "
        "Shared, private, and output components are varied one at a time while the remaining components are held fixed. "
        "These attributed variances do not generally sum to the full posterior variance because covariance and interaction terms are omitted."
    )
    for H in [float(h) for h in H_list]:
        total_R = posterior_risk_samples_at_H(
            model, Xte_t, g_te_t, H, alpha_dt, dt_edges, S=S
        )
        shared_samples = []
        private_samples = []
        output_samples = []
        with torch.no_grad():
            for _ in range(int(S)):
                hs = model.shared_latent_mt(Xte_t, g_te_t, sample=True)
                h = torch.cat([hs, h_private_mean], dim=-1).contiguous()
                score = _output_score_from_latents(model, h, g_te_t, sample=False)
                shared_samples.append(
                    _risk_from_score_tensor(score, dt_edges, alpha_dt, H)
                )

                hp = _private_latent_draw(model, Xte_t, g_te_t, sample=True)
                h = torch.cat([h_shared_mean, hp], dim=-1).contiguous()
                score = _output_score_from_latents(model, h, g_te_t, sample=False)
                private_samples.append(
                    _risk_from_score_tensor(score, dt_edges, alpha_dt, H)
                )

                score = _output_score_from_latents(model, h_mean, g_te_t, sample=True)
                output_samples.append(
                    _risk_from_score_tensor(score, dt_edges, alpha_dt, H)
                )
        shared_arr = np.asarray(shared_samples, float)
        private_arr = np.asarray(private_samples, float)
        output_arr = np.asarray(output_samples, float)
        total_var = np.var(total_R, axis=0)
        total_sd = np.sqrt(np.maximum(total_var, 0.0))
        shared_var = np.var(shared_arr, axis=0)
        private_var = np.var(private_arr, axis=0)
        output_var = np.var(output_arr, axis=0)
        component_sum_var = shared_var + private_var + output_var
        comp_sum_safe = np.where(component_sum_var <= 1e-12, np.nan, component_sum_var)
        total_var_safe = np.where(total_var <= 1e-12, np.nan, total_var)
        residual_var = total_var - component_sum_var
        shared_share_attr = shared_var / comp_sum_safe
        private_share_attr = private_var / comp_sum_safe
        output_share_attr = output_var / comp_sum_safe
        shared_share_total = shared_var / total_var_safe
        private_share_total = private_var / total_var_safe
        output_share_total = output_var / total_var_safe
        residual_share_total = residual_var / total_var_safe
        for i in range(len(t_te)):
            total_rows.append(
                {
                    "test_index": int(i),
                    "group_idx": int(g_np[i]),
                    "group": (
                        str(groups[g_np[i]])
                        if 0 <= int(g_np[i]) < len(groups)
                        else str(g_np[i])
                    ),
                    "event_time_months": float(t_te[i]),
                    "event": int(e_te[i]),
                    "H_months": float(H),
                    "total_posterior_risk_sd": float(total_sd[i]),
                    "shared_branch_conditional_sd": float(
                        np.sqrt(max(shared_var[i], 0.0))
                    ),
                    "private_branch_conditional_sd": float(
                        np.sqrt(max(private_var[i], 0.0))
                    ),
                    "output_layer_conditional_sd": float(
                        np.sqrt(max(output_var[i], 0.0))
                    ),
                    "shared_attribution_share_within_attributed_variance": (
                        float(shared_share_attr[i])
                        if np.isfinite(shared_share_attr[i])
                        else np.nan
                    ),
                    "private_attribution_share_within_attributed_variance": (
                        float(private_share_attr[i])
                        if np.isfinite(private_share_attr[i])
                        else np.nan
                    ),
                    "output_attribution_share_within_attributed_variance": (
                        float(output_share_attr[i])
                        if np.isfinite(output_share_attr[i])
                        else np.nan
                    ),
                    "shared_attribution_share_of_total_variance": (
                        float(shared_share_total[i])
                        if np.isfinite(shared_share_total[i])
                        else np.nan
                    ),
                    "private_attribution_share_of_total_variance": (
                        float(private_share_total[i])
                        if np.isfinite(private_share_total[i])
                        else np.nan
                    ),
                    "output_attribution_share_of_total_variance": (
                        float(output_share_total[i])
                        if np.isfinite(output_share_total[i])
                        else np.nan
                    ),
                    "residual_interaction_share_of_total_variance": (
                        float(residual_share_total[i])
                        if np.isfinite(residual_share_total[i])
                        else np.nan
                    ),
                    "attribution_note": note,
                }
            )
    df = pd.DataFrame(total_rows)
    df.to_csv(Path(outdir) / "test_branch_uncertainty_decomposition.csv", index=False)
    (Path(outdir) / "test_branch_uncertainty_decomposition_notes.txt").write_text(
        note + "\n", encoding="utf-8"
    )
    if len(df):
        grp = (
            df.groupby(["group", "H_months"], dropna=False)
            .agg(
                n=("test_index", "count"),
                mean_total_posterior_risk_sd=("total_posterior_risk_sd", "mean"),
                mean_shared_attr_within_attributed=(
                    "shared_attribution_share_within_attributed_variance",
                    "mean",
                ),
                mean_private_attr_within_attributed=(
                    "private_attribution_share_within_attributed_variance",
                    "mean",
                ),
                mean_output_attr_within_attributed=(
                    "output_attribution_share_within_attributed_variance",
                    "mean",
                ),
                mean_residual_share_of_total=(
                    "residual_interaction_share_of_total_variance",
                    "mean",
                ),
            )
            .reset_index()
        )
        grp.to_csv(
            Path(outdir) / "test_branch_uncertainty_group_summary.csv", index=False
        )
        _plot_group_horizon_heatmap(
            grp,
            "mean_total_posterior_risk_sd",
            "Mean total posterior risk SD by group and horizon",
            Path(outdir) / "test_branch_total_risk_sd_heatmap.png",
        )
        for H in sorted(grp["H_months"].dropna().unique().tolist()):
            sub = grp[
                np.isclose(pd.to_numeric(grp["H_months"], errors="coerce"), float(H))
            ].copy()
            if len(sub) == 0:
                continue
            fig, ax = plt.subplots(figsize=(7.6, max(4.0, 0.7 * len(sub) + 1.8)))
            y = np.arange(len(sub))
            v1 = sub["mean_shared_attr_within_attributed"].to_numpy(dtype=float)
            v2 = sub["mean_private_attr_within_attributed"].to_numpy(dtype=float)
            v3 = sub["mean_output_attr_within_attributed"].to_numpy(dtype=float)
            ax.barh(y, v1, label="shared branch", alpha=0.85)
            ax.barh(
                y,
                v2,
                left=np.nan_to_num(v1, nan=0.0),
                label="private branch",
                alpha=0.85,
            )
            ax.barh(
                y,
                v3,
                left=np.nan_to_num(v1, nan=0.0) + np.nan_to_num(v2, nan=0.0),
                label="output layer",
                alpha=0.85,
            )
            ax.set_yticks(y)
            ax.set_yticklabels(sub["group"].astype(str).tolist())
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("Normalized share within attributed variance")
            ax.set_title(
                f"Approximate branch-attributed variance composition @ H={int(H)} months"
            )
            ax.legend(
                frameon=True,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
            ax.grid(True, axis="x", alpha=0.18)
            fig.tight_layout(rect=[0, 0, 0.84, 1])
            savefig(Path(outdir) / f"branch_uncertainty_variance_share_H{int(H)}.png")

            fig2, ax2 = plt.subplots(figsize=(7.2, max(4.0, 0.7 * len(sub) + 1.8)))
            vr = sub["mean_residual_share_of_total"].to_numpy(dtype=float)
            ax2.barh(y, vr, alpha=0.85)
            ax2.set_yticks(y)
            ax2.set_yticklabels(sub["group"].astype(str).tolist())
            ax2.axvline(0.0, color="gray", lw=1.0)
            ax2.set_xlabel("Residual share of total posterior variance")
            ax2.set_title(f"Unattributed residual variance gap @ H={int(H)} months")
            ax2.grid(True, axis="x", alpha=0.18)
            fig2.tight_layout()
            savefig(Path(outdir) / f"branch_uncertainty_residual_share_H{int(H)}.png")
        return grp
    return None


def _plot_signed_lollipop(
    df_plot, value_col, label_col, title, outpath, xlabel, topN=24
):
    df_plot = df_plot.copy()
    df_plot[value_col] = pd.to_numeric(df_plot[value_col], errors="coerce")
    df_plot = df_plot[np.isfinite(df_plot[value_col].to_numpy(dtype=float))].copy()
    if len(df_plot) == 0:
        return
    if len(df_plot) > int(topN):
        df_plot = df_plot.reindex(
            df_plot[value_col].abs().sort_values(ascending=False).head(int(topN)).index
        ).copy()
    df_plot = df_plot.sort_values(value_col).copy()
    y = np.arange(len(df_plot))
    vals = df_plot[value_col].to_numpy(dtype=float)
    labs = df_plot[label_col].astype(str).tolist()
    colors = np.where(vals >= 0, "#D55E00", "#2C7FB8")
    fig_h = max(5.0, 0.38 * len(df_plot) + 1.6)
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    ax.axvline(0.0, color="#9099A1", lw=1.0, alpha=0.8)
    ax.hlines(y, 0.0, vals, color="#D6DEE6", lw=1.8, zorder=1)
    ax.scatter(
        vals, y, s=90, color=colors, edgecolors="white", linewidths=0.9, zorder=2
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(True, axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(outpath)


def plot_pc_gene_backprojection_lollipops(prep_dir, outdir, top_genes=24):
    if prep_dir is None:
        return
    prep_dir = Path(prep_dir)
    out_csv_dir = Path(outdir) / "pc_gene_backprojections"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for cancer_code in ALL_CANCER_CODES:
        load_path = (
            prep_dir / f"tcga_{str(cancer_code).lower()}_selected_pc_loadings.csv"
        )
        if not load_path.exists():
            continue
        df = pd.read_csv(load_path, index_col=0)
        if df.shape[1] == 0:
            continue
        for pc in df.columns:
            s = pd.to_numeric(df[pc], errors="coerce")
            s = s[np.isfinite(s)]
            if s.empty:
                continue
            top = s.abs().sort_values(ascending=False).head(int(top_genes))
            sub = pd.DataFrame(
                {
                    "gene": top.index.astype(str),
                    "loading": s.loc[top.index].to_numpy(dtype=float),
                    "cancer_code": str(cancer_code),
                    "pc": str(pc),
                }
            )
            sub.to_csv(
                out_csv_dir
                / f"{_safe_fname(cancer_code)}__{_safe_fname(pc)}_top_gene_loadings.csv",
                index=False,
            )
            _plot_signed_lollipop(
                sub,
                value_col="loading",
                label_col="gene",
                title=f"{cancer_code} {pc} gene back-projection (top |loading| genes)",
                outpath=out_csv_dir
                / f"{_safe_fname(cancer_code)}__{_safe_fname(pc)}_gene_backprojection_lollipop.png",
                xlabel="PC loading",
                topN=int(top_genes),
            )
            summary_rows.append(
                {
                    "cancer_code": str(cancer_code),
                    "pc": str(pc),
                    "n_genes_plotted": int(len(sub)),
                    "max_abs_loading": float(
                        np.max(np.abs(sub["loading"].to_numpy(dtype=float)))
                    ),
                }
            )
    if len(summary_rows):
        pd.DataFrame(summary_rows).to_csv(
            out_csv_dir / "pc_gene_backprojection_manifest.csv", index=False
        )


def run_fc_driven_pc_backprojection_multiH(
    prep_dir,
    outdir,
    fc_df,
    feat_names,
    X_ref_s,
    nlS_vec,
    H_fc,
    top_pc_features=10,
    top_genes=24,
    scope_tag="test",
):
    if prep_dir is None or fc_df is None:
        return [], []
    prep_dir = Path(prep_dir)
    outdir = Path(outdir)
    feat_names = list(map(str, feat_names))
    if not _has_pc_features(feat_names):
        print(
            f"[PC_BACKPROJECTION_SKIP] No gene-expression PC features are present for H={int(H_fc)}; "
            "no gene-expression PC columns were detected, so PC-to-gene backprojection is skipped."
        )
        return [], []
    feat_to_idx = {str(f): i for i, f in enumerate(feat_names)}
    X_ref_s = np.asarray(X_ref_s, float)
    nlS_vec = np.asarray(nlS_vec, float)

    loadings_by_cancer = {}
    for cancer_code in ALL_CANCER_CODES:
        load_path = (
            prep_dir / f"tcga_{str(cancer_code).lower()}_selected_pc_loadings.csv"
        )
        if not load_path.exists():
            continue
        try:
            loadings_by_cancer[str(cancer_code)] = pd.read_csv(load_path, index_col=0)
        except Exception:
            continue
    if len(loadings_by_cancer) == 0:
        return [], []

    df_rank = fc_df.copy()
    if "feature" not in df_rank.columns or "FC_score" not in df_rank.columns:
        return [], []
    df_rank["FC_score"] = pd.to_numeric(df_rank["FC_score"], errors="coerce")
    df_rank = df_rank[np.isfinite(df_rank["FC_score"])].copy()
    df_rank = df_rank.sort_values("FC_score", ascending=False).reset_index(drop=True)

    out_pc_dir = outdir / "pc_gene_backprojections_fc" / f"H{int(H_fc)}"
    out_pc_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    gene_rows = []

    pc_candidates = []
    for rank0, row in df_rank.iterrows():
        fname = str(row["feature"])
        cancer_code, pc_name = _parse_pc_feature_name(fname)
        if pc_name is None or fname not in feat_to_idx:
            continue
        resolved_cancer = None
        if cancer_code is not None and cancer_code in loadings_by_cancer:
            if pc_name in loadings_by_cancer[cancer_code].columns:
                resolved_cancer = str(cancer_code)
        else:
            hits = [
                cc
                for cc, df_load in loadings_by_cancer.items()
                if pc_name in df_load.columns
            ]
            if len(hits) == 1:
                resolved_cancer = str(hits[0])
        if resolved_cancer is None:
            continue
        pc_candidates.append(
            {
                "feature": fname,
                "feature_rank": int(rank0 + 1),
                "feat_idx": int(feat_to_idx[fname]),
                "cancer_code": resolved_cancer,
                "pc_name": str(pc_name),
                "fc_score": float(row["FC_score"]),
            }
        )
        if len(pc_candidates) >= int(top_pc_features):
            break

    for pc_rank, info in enumerate(pc_candidates, start=1):
        df_load = loadings_by_cancer[info["cancer_code"]]
        s = pd.to_numeric(df_load[info["pc_name"]], errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            continue
        pc_scores = np.asarray(X_ref_s[:, int(info["feat_idx"])], float)
        rho = _raw_spearman(pc_scores, nlS_vec)
        orient_sign = 1.0 if (np.isfinite(rho) and rho >= 0.0) else -1.0
        s_oriented = orient_sign * s
        top = s_oriented.abs().sort_values(ascending=False).head(int(top_genes))
        sub = pd.DataFrame(
            {
                "gene": top.index.astype(str),
                "loading_oriented": s_oriented.loc[top.index].to_numpy(dtype=float),
                "abs_loading": np.abs(s_oriented.loc[top.index].to_numpy(dtype=float)),
                "feature": str(info["feature"]),
                "cancer_code": str(info["cancer_code"]),
                "pc": str(info["pc_name"]),
                "scope": str(scope_tag),
                "H": int(H_fc),
                "fc_score": float(info["fc_score"]),
                "raw_spearman_with_neglogS": float(rho),
            }
        )
        csv_path = (
            out_pc_dir
            / f"{_safe_fname(info['feature'])}_top_gene_loadings_FCtop_H{int(H_fc)}.csv"
        )
        png_path = (
            out_pc_dir
            / f"{_safe_fname(info['feature'])}_gene_backprojection_lollipop_FCtop_H{int(H_fc)}.png"
        )
        sub.to_csv(csv_path, index=False)
        _plot_signed_lollipop(
            sub,
            value_col="loading_oriented",
            label_col="gene",
            title=f"{info['feature']} (FC-top) gene back-projection @ H={int(H_fc)} mo",
            outpath=png_path,
            xlabel="PC loading (oriented)",
            topN=int(top_genes),
        )
        summary_rows.append(
            {
                "scope": str(scope_tag),
                "H": int(H_fc),
                "feature": str(info["feature"]),
                "cancer_code": str(info["cancer_code"]),
                "pc": str(info["pc_name"]),
                "feature_rank": int(info["feature_rank"]),
                "pc_rank_within_fc_top": int(pc_rank),
                "fc_score": float(info["fc_score"]),
                "raw_spearman_with_neglogS": float(rho),
                "n_genes_plotted": int(len(sub)),
                "max_abs_loading": float(
                    np.max(np.abs(sub["loading_oriented"].to_numpy(dtype=float)))
                ),
                "csv_path": str(csv_path),
                "png_path": str(png_path),
            }
        )
        gene_rows.append(sub.copy())

    return summary_rows, gene_rows


def _binary_event_by_horizon(times, events, H):
    valid, y, w = _ipcw_binary_event_targets(times, events, H)
    keep = np.asarray(valid, bool) & np.isfinite(w) & (w > 0)
    return keep, y


def _auroc_auprc_summary(risk_scores, times, events, H):
    risk_scores = np.asarray(risk_scores, float)
    valid, y, w = _ipcw_binary_event_targets(times, events, H)
    keep = np.asarray(valid, bool) & np.isfinite(risk_scores) & np.isfinite(w) & (w > 0)
    y = np.asarray(y, int)[keep]
    s = np.asarray(risk_scores, float)[keep]
    sw = np.asarray(w, float)[keep]
    row = {
        "H": int(H),
        "n_eval": int(len(y)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "effective_weight_sum": float(np.sum(sw)) if sw.size else 0.0,
        "weighted_n_pos": float(np.sum(sw[y == 1])) if sw.size else 0.0,
        "weighted_n_neg": float(np.sum(sw[y == 0])) if sw.size else 0.0,
        "auroc": np.nan,
        "auprc": np.nan,
        "skipped": False,
        "estimation_method": "IPCW fixed-horizon discrimination",
    }
    if (
        len(y) < 3
        or int(np.sum(y == 1)) < 2
        or int(np.sum(y == 0)) < 2
        or float(np.sum(sw)) <= 0.0
    ):
        row["skipped"] = True
        return row
    row["auroc"] = float(roc_auc_score(y, s, sample_weight=sw))
    row["auprc"] = float(average_precision_score(y, s, sample_weight=sw))
    return row


def plot_auroc_auprc_curves(
    risk_scores,
    times,
    events,
    H,
    outdir,
    prefix="test",
    scope="pooled",
    group="pooled",
    title_prefix=None,
):
    """Save IPCW ROC/PR curves and return the fixed-horizon metrics.

    The metrics are calculated from the raw IPCW-weighted empirical curves and
    displayed in a side annotation panel, not inside the plotting area.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    risk_scores = np.asarray(risk_scores, float)
    valid, y, w = _ipcw_binary_event_targets(times, events, H)
    keep = np.asarray(valid, bool) & np.isfinite(risk_scores) & np.isfinite(w) & (w > 0)
    y = np.asarray(y, int)[keep]
    s = risk_scores[keep]
    sw = np.asarray(w, float)[keep]
    row = {
        "prefix": str(prefix),
        "scope": str(scope),
        "group": str(group),
        "H": int(H),
        "n_eval": int(len(y)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "effective_weight_sum": float(np.sum(sw)) if sw.size else 0.0,
        "weighted_prevalence": np.nan,
        "AUROC": np.nan,
        "AUPRC": np.nan,
        "skipped": False,
        "estimation_method": "IPCW fixed-horizon discrimination",
    }
    if sw.size and float(np.sum(sw)) > 0.0:
        row["weighted_prevalence"] = float(np.sum(sw * y) / max(np.sum(sw), 1e-8))
    if (
        len(y) < 3
        or int(np.sum(y == 1)) < 2
        or int(np.sum(y == 0)) < 2
        or float(np.sum(sw)) <= 0.0
    ):
        row["skipped"] = True
        return row

    auroc = float(roc_auc_score(y, s, sample_weight=sw))
    auprc = float(average_precision_score(y, s, sample_weight=sw))
    row["AUROC"] = auroc
    row["AUPRC"] = auprc

    fpr, tpr, _ = roc_curve(y, s, sample_weight=sw)
    prec, rec, _ = precision_recall_curve(y, s, sample_weight=sw)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        outdir / f"{prefix}_roc_curve_H{int(H)}.csv", index=False
    )
    pd.DataFrame({"recall": rec, "precision": prec}).to_csv(
        outdir / f"{prefix}_pr_curve_H{int(H)}.csv", index=False
    )

    plot_title_prefix = str(title_prefix) if title_prefix is not None else str(scope).replace("_", " ").title()
    fig, axs = plt.subplots(1, 2, figsize=(14.2, 5.0))
    axs[0].step(fpr, tpr, where="post", lw=2.0)
    axs[0].plot([0, 1], [0, 1], ls="--", lw=1.0, color="gray")
    axs[0].set_xlabel("False positive rate")
    axs[0].set_ylabel("True positive rate")
    axs[0].set_title(f"{plot_title_prefix} ROC curve, H={int(H)} months")
    axs[0].grid(True, alpha=0.18)

    baseline = float(row["weighted_prevalence"])
    axs[1].step(rec, prec, where="post", lw=2.0)
    if np.isfinite(baseline):
        axs[1].axhline(baseline, ls="--", lw=1.0, color="gray")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_title(f"{plot_title_prefix} precision-recall curve, H={int(H)} months")
    axs[1].grid(True, alpha=0.18)

    metric_text = (
        f"AUROC = {auroc:.4f}\n"
        f"AUPRC = {auprc:.4f}\n"
        f"Weighted prevalence = {baseline:.4f}\n"
        f"n+ = {int(row['n_pos'])}, n− = {int(row['n_neg'])}"
    )
    fig.text(
        0.885,
        0.72,
        metric_text,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#BFC7D0", alpha=0.95, pad=6.0),
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
    outpath = Path(outdir) / f"{prefix}_ROC_PR_curves_H{int(H)}.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_with_source(fig, outpath)
    plt.close(fig)
    return row

def _posthoc_log(outdir, msg):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(msg, flush=True)
    with open(outdir / "posthoc_progress.log", "a", encoding="utf-8") as f:
        f.write(str(msg).rstrip() + "\n")


def _write_posthoc_traceback(outdir, *args):
    """Write a posthoc traceback.

    Accepts either (outdir, exc) or (outdir, label, exc) for compatibility with
    older patched blocks that wanted a named traceback file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if len(args) == 1:
        label = "posthoc_traceback"
        exc = args[0]
    elif len(args) >= 2:
        label = str(args[0])
        exc = args[1]
    else:
        label = "posthoc_traceback"
        exc = RuntimeError("unknown posthoc exception")
    label = _safe_fname(label) if "_safe_fname" in globals() else re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    tb = traceback.format_exc()
    with open(outdir / f"{label}.txt", "w", encoding="utf-8") as f:
        f.write(tb)
    if label != "posthoc_traceback":
        with open(outdir / "posthoc_traceback.txt", "w", encoding="utf-8") as f:
            f.write(tb)
    _posthoc_log(outdir, f"[POSTHOC][ERROR] {type(exc).__name__}: {exc}")


def _km_predict_step(kmf, times):
    return np.asarray(kmf.predict(np.asarray(times, float)), float)


def _fit_censoring_km(times, events):
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    m = np.isfinite(times) & np.isfinite(events)
    times = times[m]
    events = events[m]
    km_c = KaplanMeierFitter()
    km_c.fit(times, event_observed=(events == 0).astype(int))
    return km_c


def _parse_repeated_split_metric_col(col, groups=None):
    """Parse repeated-split metric columns into group / metric / split / horizon."""
    groups = [] if groups is None else [str(g) for g in groups]
    metric_names = ["somers_dxy", "c", "brier", "ibs", "auroc", "auprc"]
    m = re.match(
        r"^(?P<body>.+?)_(?P<split>train|test)_H(?P<H>\d+(?:\.\d+)?)$", str(col)
    )
    if m is None:
        return None
    body = m.group("body")
    split = m.group("split")
    H = float(m.group("H"))
    group = "pooled"
    metric = body
    for g in groups:
        prefix = f"{g}_"
        if body.startswith(prefix):
            group = g
            metric = body[len(prefix) :]
            break
    if metric not in metric_names:
        return None
    return {"metric": metric, "split": split, "H": H, "group": group}


def _metric_summary_row(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "median": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n_non_nan": 0,
        }
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(np.mean(vals)),
        "sd": sd,
        "sem": float(sd / np.sqrt(n)) if n > 1 else 0.0,
        "median": float(np.median(vals)),
        "q05": float(np.quantile(vals, 0.05)),
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
        "q95": float(np.quantile(vals, 0.95)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n_non_nan": n,
    }


def _fmt_mean_sd(mean, sd, digits=4):
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(sd):
        return f"{mean:.{digits}f} ± nan"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def export_repeated_split_metric_summaries(ok_df, cfg_dir, cfg_tag, groups=None):
    """Export mean±SD across outer splits for all main performance metrics."""
    cfg_dir = Path(cfg_dir)
    groups = [] if groups is None else [str(g) for g in groups]
    if ok_df is None or len(ok_df) == 0:
        empty = pd.DataFrame()
        empty.to_csv(cfg_dir / f"{cfg_tag}_summary_all_metrics.csv", index=False)
        empty.to_csv(cfg_dir / f"{cfg_tag}_summary_all_metrics_wide.csv", index=False)
        return empty
    rows = []
    for col in ok_df.columns:
        parsed = _parse_repeated_split_metric_col(col, groups=groups)
        if parsed is None:
            continue
        summ = _metric_summary_row(ok_df[col])
        rows.append(
            {
                "column": str(col),
                "metric": parsed["metric"],
                "split": parsed["split"],
                "H": parsed["H"],
                "H_months": (
                    int(parsed["H"]) if float(parsed["H"]).is_integer() else parsed["H"]
                ),
                "group": parsed["group"],
                **summ,
                "mean_sd": _fmt_mean_sd(summ["mean"], summ["sd"]),
                "n_seeds_total": (
                    int(ok_df["seed"].nunique())
                    if "seed" in ok_df.columns
                    else int(len(ok_df))
                ),
            }
        )
    out_df = pd.DataFrame(rows)
    if len(out_df):
        metric_order = {
            m: i
            for i, m in enumerate(["c", "somers_dxy", "brier", "ibs", "auroc", "auprc"])
        }
        split_order = {"test": 0, "train": 1}
        group_order = {"pooled": 0, **{g: i + 1 for i, g in enumerate(groups)}}
        out_df["metric_order"] = (
            out_df["metric"].map(metric_order).fillna(999).astype(int)
        )
        out_df["split_order"] = out_df["split"].map(split_order).fillna(999).astype(int)
        out_df["group_order"] = out_df["group"].map(group_order).fillna(999).astype(int)
        out_df = out_df.sort_values(
            ["split_order", "metric_order", "H", "group_order", "group"]
        ).drop(columns=["metric_order", "split_order", "group_order"])
    out_df.to_csv(cfg_dir / f"{cfg_tag}_summary_all_metrics.csv", index=False)

    if len(out_df):
        wide = out_df.pivot_table(
            index=["metric", "split", "H", "H_months"],
            columns="group",
            values="mean_sd",
            aggfunc="first",
        ).reset_index()
        preferred_cols = ["metric", "split", "H", "H_months", "pooled"] + [
            g for g in groups if g in wide.columns
        ]
        other_cols = [c for c in wide.columns if c not in preferred_cols]
        wide = wide[[c for c in preferred_cols if c in wide.columns] + other_cols]
    else:
        wide = pd.DataFrame()
    wide.to_csv(cfg_dir / f"{cfg_tag}_summary_all_metrics_wide.csv", index=False)
    return out_df


def export_repeated_split_event_count_summaries(count_rows, cfg_dir, cfg_tag):
    """Export fixed-horizon event counts/prevalence for every repeated split and mean±SD summaries."""
    cfg_dir = Path(cfg_dir)
    df_counts = pd.DataFrame(count_rows)
    df_counts.to_csv(
        cfg_dir / f"{cfg_tag}_early_failure_event_counts_by_split.csv", index=False
    )
    if len(df_counts) == 0:
        empty = pd.DataFrame()
        empty.to_csv(
            cfg_dir / f"{cfg_tag}_early_failure_event_counts_summary.csv", index=False
        )
        return df_counts, empty
    numeric_cols = [
        "n_total",
        "n_known_by_H",
        "n_early_events",
        "n_event_free_at_H",
        "n_censored_before_H",
        "known_fraction",
        "censored_before_H_fraction",
        "early_event_rate_known",
        "early_event_rate_total_naive",
        "event_free_rate_known",
        "ipcw_weight_sum",
        "weighted_n_early_events",
        "baseline_ipcw_event_rate",
    ]
    rows = []
    for (split, group, H), dfg in df_counts.groupby(
        ["split", "group", "H"], dropna=False
    ):
        base = {
            "split": split,
            "group": group,
            "H": H,
            "H_months": int(H) if float(H).is_integer() else H,
        }
        base["n_splits"] = (
            int(dfg["seed"].nunique()) if "seed" in dfg.columns else int(len(dfg))
        )
        for col in numeric_cols:
            if col not in dfg.columns:
                continue
            summ = _metric_summary_row(dfg[col])
            for key, val in summ.items():
                base[f"{col}_{key}"] = val
            base[f"{col}_mean_sd"] = _fmt_mean_sd(summ["mean"], summ["sd"])
        rows.append(base)
    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(["split", "H", "group"]).reset_index(drop=True)
    summary.to_csv(
        cfg_dir / f"{cfg_tag}_early_failure_event_counts_summary.csv", index=False
    )
    return df_counts, summary


def _ipcw_binary_event_targets(times, events, H):
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    H = float(H)
    y = np.zeros(times.shape[0], dtype=int)
    w = np.zeros(times.shape[0], dtype=float)
    valid = np.isfinite(times) & np.isfinite(events)
    if int(np.sum(valid)) == 0:
        return valid, y, w
    t = times[valid]
    e = events[valid]
    pos = (e == 1) & (t <= H)
    neg = t > H
    try:
        km_c = _fit_censoring_km(t, e)
        t_left = np.maximum(np.nextafter(t, -np.inf), 0.0)
        G_tminus = np.clip(_km_predict_step(km_c, t_left), 1e-8, None)
        G_H = float(
            np.clip(_km_predict_step(km_c, np.array([H], dtype=float))[0], 1e-8, None)
        )
    except Exception:
        G_tminus = np.ones(t.shape[0], dtype=float)
        G_H = 1.0
    y_valid = pos.astype(int)
    w_valid = np.zeros(t.shape[0], dtype=float)
    w_valid[pos] = 1.0 / G_tminus[pos]
    w_valid[neg] = 1.0 / G_H
    y[valid] = y_valid
    w[valid] = w_valid
    return valid, y, w


def _weighted_mean_safe(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if int(np.sum(m)) == 0:
        return np.nan
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def brier_score_ipcw_at_H(surv_prob, times, events, H):
    surv_prob = np.asarray(surv_prob, float)
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    m = np.isfinite(surv_prob) & np.isfinite(times) & np.isfinite(events)
    surv_prob = np.clip(surv_prob[m], 1e-12, 1.0)
    times = times[m]
    events = events[m]
    if surv_prob.size == 0:
        return np.nan
    censor_event = (events == 0).astype(int)
    try:
        km_c = KaplanMeierFitter()
        km_c.fit(times, event_observed=censor_event)
        G_t = np.clip(
            _km_predict_step(km_c, np.full(times.shape[0], float(H))), 1e-8, None
        )
        G_tm = np.clip(
            _km_predict_step(km_c, np.maximum(times - 1e-8, 0.0)), 1e-8, None
        )
    except Exception:
        G_t = np.ones(times.shape[0], dtype=float)
        G_tm = np.ones(times.shape[0], dtype=float)
    term_event = (
        ((times <= float(H)) & (events == 1)).astype(float) * (surv_prob**2) / G_tm
    )
    term_alive = (times > float(H)).astype(float) * ((1.0 - surv_prob) ** 2) / G_t
    return float(np.mean(term_event + term_alive))


def _ibs_from_brier_curve(time_grid, brier_vals, H):
    time_grid = np.asarray(time_grid, float)
    brier_vals = np.asarray(brier_vals, float)
    m = (
        np.isfinite(time_grid)
        & np.isfinite(brier_vals)
        & (time_grid > 0.0)
        & (time_grid <= float(H))
    )
    if int(np.sum(m)) == 0:
        return np.nan
    tg = time_grid[m]
    bv = brier_vals[m]
    if tg.size == 1:
        return float(bv[0])
    return float(np.trapz(bv, tg) / float(H))


def _has_treatment_flags(meta_df):
    if meta_df is None or not isinstance(meta_df, pd.DataFrame):
        return False
    return all(c in meta_df.columns for c in TREATMENT_FLAG_COLS)


def _treatment_combo_label_from_row(row):
    active = []
    for col in TREATMENT_FLAG_COLS:
        try:
            v = float(row.get(col, 0.0))
        except Exception:
            v = 0.0
        if np.isfinite(v) and v > 0.5:
            active.append(TREATMENT_FLAG_LABELS[col])
    if len(active) == 0:
        return "No flagged treatment"
    return " + ".join(active)


def build_treatment_combo_series(meta_df):
    if not _has_treatment_flags(meta_df):
        return pd.Series(["Treatment unavailable"] * len(meta_df), index=meta_df.index)
    ser = meta_df.apply(_treatment_combo_label_from_row, axis=1)
    return ser.astype(str)


def summarize_treatment_combos(meta_df):
    if meta_df is None or not isinstance(meta_df, pd.DataFrame) or len(meta_df) == 0:
        return pd.DataFrame()
    work = meta_df.copy()
    work["TREATMENT_COMBO"] = build_treatment_combo_series(work)
    grp_cols = ["GROUP_LABEL", "TREATMENT_COMBO"]
    if "PFI_EVENT" in work.columns:
        out = (
            work.groupby(grp_cols, dropna=False)
            .agg(
                n=("TREATMENT_COMBO", "size"),
                events=(
                    "PFI_EVENT",
                    lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum()),
                ),
            )
            .reset_index()
            .sort_values(
                ["GROUP_LABEL", "n", "events", "TREATMENT_COMBO"],
                ascending=[True, False, False, True],
            )
            .reset_index(drop=True)
        )
    else:
        out = (
            work.groupby(grp_cols, dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(
                ["GROUP_LABEL", "n", "TREATMENT_COMBO"], ascending=[True, False, True]
            )
            .reset_index(drop=True)
        )
        out["events"] = np.nan
    return out


def _fit_high_vs_low_hr(t, e, gL, gH):
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    gL = np.asarray(gL, bool)
    gH = np.asarray(gH, bool)
    keep = gL | gH
    if (
        int(np.sum(keep)) < 10
        or int(np.sum(gL)) < TREATMENT_RISK_STRATA_MIN_COUNT
        or int(np.sum(gH)) < TREATMENT_RISK_STRATA_MIN_COUNT
    ):
        return {
            "hr_high_vs_low": np.nan,
            "hr_low95": np.nan,
            "hr_high95": np.nan,
            "n_low_high": int(np.sum(keep)),
            "events_low_high": int(np.sum(e[keep])) if int(np.sum(keep)) else 0,
        }
    if int(np.sum(e[keep])) < 6:
        return {
            "hr_high_vs_low": np.nan,
            "hr_low95": np.nan,
            "hr_high95": np.nan,
            "n_low_high": int(np.sum(keep)),
            "events_low_high": int(np.sum(e[keep])),
        }
    df = pd.DataFrame(
        {
            "time": t[keep],
            "event": e[keep].astype(int),
            "high_risk": gH[keep].astype(int),
        }
    )
    try:
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(df, duration_col="time", event_col="event", show_progress=False)
        s = (
            cph.summary.loc["high_risk"]
            if "high_risk" in cph.summary.index
            else cph.summary.iloc[0]
        )
        return {
            "hr_high_vs_low": float(np.exp(s["coef"])),
            "hr_low95": float(np.exp(s["coef lower 95%"])),
            "hr_high95": float(np.exp(s["coef upper 95%"])),
            "n_low_high": int(np.sum(keep)),
            "events_low_high": int(np.sum(e[keep])),
        }
    except Exception:
        return {
            "hr_high_vs_low": np.nan,
            "hr_low95": np.nan,
            "hr_high95": np.nan,
            "n_low_high": int(np.sum(keep)),
            "events_low_high": int(np.sum(e[keep])),
        }


def plot_treatment_hr_forest(summary_df, H_post, outpath, title=None):
    if summary_df is None or len(summary_df) == 0:
        return
    df = summary_df.copy()
    for c in ["hr_high_vs_low", "hr_low95", "hr_high95", "n", "events"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = (
        np.isfinite(df["hr_high_vs_low"].to_numpy(dtype=float))
        & np.isfinite(df["hr_low95"].to_numpy(dtype=float))
        & np.isfinite(df["hr_high95"].to_numpy(dtype=float))
    )
    df = df.loc[keep].copy()
    if len(df) == 0:
        return
    df["label"] = df.apply(
        lambda r: f"{r['group']} | {r['treatment_combo']} (n={int(r['n'])}, ev={int(r['events'])})",
        axis=1,
    )
    df = df.sort_values(
        ["group", "hr_high_vs_low", "treatment_combo"], ascending=[True, False, True]
    ).reset_index(drop=True)
    y = np.arange(len(df))
    lo = df["hr_low95"].to_numpy(dtype=float)
    mid = df["hr_high_vs_low"].to_numpy(dtype=float)
    hi = df["hr_high95"].to_numpy(dtype=float)
    err = np.vstack([np.clip(mid - lo, 0.0, None), np.clip(hi - mid, 0.0, None)])
    fig_h = max(5.0, 0.42 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(9.8, fig_h))
    ax.errorbar(mid, y, xerr=err, fmt="o", capsize=3, lw=1.4)
    ax.axvline(1.0, ls="--", lw=1.0, color="gray", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    ax.set_xlabel("Hazard ratio: high vs low posterior risk")
    ax.set_title(
        title
        or f"High-vs-low posterior risk HR by cancer and treatment subgroup (H={int(H_post)} mo)"
    )
    savefig(outpath)


def fit_one_config_with_artifacts(
    Xtr_np,
    Xte_np,
    t_tr,
    t_te,
    e_tr,
    e_te,
    g_tr,
    g_te,
    groups,
    feat_names,
    blocks,
    surv_top_k,
    shared_dim,
    private_dim,
    mt_latent_rank,
    output_rank,
    meta_tr=None,
    meta_te=None,
    prep_dir=None,
):
    Xtr_sub_np, Xte_sub_np, feat_names_sub = subset_feature_matrices(
        Xtr_np, Xte_np, feat_names, surv_top_k
    )
    blocks_sub = infer_blocks_from_feat_names(feat_names_sub)
    Xfit_np = Xtr_sub_np
    t_fit, e_fit, g_fit = t_tr, e_tr, g_tr
    Xfit_t, Xte_t = standardize_from_train(Xfit_np, Xte_sub_np)
    g_fit_t = torch.tensor(g_fit, dtype=torch.long, device=device)
    g_te_t = torch.tensor(g_te, dtype=torch.long, device=device)
    dt_edges = make_discrete_time_edges(t_fit, DT_BINS)
    event_bin_fit = discretize_times(t_fit, dt_edges)
    event_bin_fit_t = torch.tensor(event_bin_fit, dtype=torch.long, device=device)
    e_fit_t = torch.tensor(e_fit, dtype=torch.float32, device=device)
    model = DGP(
        in_dim=Xfit_t.shape[1],
        Xref=Xfit_t,
        num_groups=len(groups),
        shared_dim=shared_dim,
        private_dim=private_dim,
        mt_shared_rank=mt_latent_rank,
        output_rank=output_rank,
        blocks=blocks_sub,
    ).to(device)
    verify_architecture(model)
    alpha_dt = torch.nn.Parameter(
        torch.zeros(len(dt_edges) - 1, device=device, dtype=torch.float32)
    )
    opt = build_optimizer(model, alpha_dt)
    AR_fit, EV_fit = make_dt_masks(t_fit, e_fit, dt_edges)
    bin_weights_t = torch.tensor(
        make_dt_bin_weights(dt_edges), dtype=torch.float32, device=device
    )
    best_train_obj = np.inf
    best_state = None
    for it in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        ell = expected_dt_ll_mc(
            model,
            Xfit_t,
            g_fit_t,
            AR_fit,
            EV_fit,
            alpha_dt,
            S=MC_TRAIN,
            bin_weights=bin_weights_t,
        )
        kl_s, kl_p, kl_o = total_kl_details(model)
        kl_total = kl_s + kl_p + kl_o
        kl_beta = beta_schedule(it)
        elbo = ell - kl_beta * kl_total
        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [alpha_dt], GRAD_CLIP)
        opt.step()

        train_obj = float(loss.detach().cpu())
        if it % PRINT_EVERY == 0 or it == 1:
            n_ev = max(1, int(np.sum(e_fit)))
            print(
                f"[{it:04d}] ELL/evt={float(ell.detach().cpu())/n_ev:.4f} | "
                f"KL/evt={(float(kl_total.detach().cpu())/n_ev):.4f} | "
                f"KLsh={(float(kl_s.detach().cpu())/n_ev):.4f} | "
                f"KLpr={(float(kl_p.detach().cpu())/n_ev):.4f} | "
                f"KLout={(float(kl_o.detach().cpu())/n_ev):.4f} | "
                f"beta={kl_beta:.3f} | ELBO={float(elbo.detach().cpu()):.4f}"
            )

        if train_obj < best_train_obj:
            best_train_obj = train_obj
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "alpha_dt": alpha_dt.detach().cpu().clone(),
            }
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        alpha_dt.data.copy_(
            best_state["alpha_dt"].to(device=device, dtype=torch.float32)
        )
    model.eval()
    out = {}
    out.update(
        evaluate_model_mc(
            model,
            Xfit_t,
            Xte_t,
            g_fit_t,
            g_te_t,
            t_fit,
            t_te,
            e_fit,
            e_te,
            g_fit,
            g_te,
            groups,
            dt_edges,
            alpha_dt,
            eval_label="test",
            include_fit=True,
            S_eval=FINAL_EVAL_MC_SAMPLES,
            eval_horizons=get_eval_horizons_with_median(t_fit, e_fit),
        )
    )
    out["train_n"] = int(len(t_fit))
    out["test_n"] = int(len(t_te))
    out["inner_train_n"] = int(len(t_fit))
    out["inner_val_n"] = 0
    out["n_features_used"] = int(Xfit_t.shape[1])
    out["pc_features_per_group"] = int(surv_top_k)
    out["pc_features_used_total"] = int(
        sum(
            1
            for f in feat_names_sub
            if re.match(r"^(BRCA|LUAD|COAD)_PC\d+$", str(f))
            or re.match(r"^PC\d+$", str(f))
        )
    )
    out["extra_nonpc_features_used"] = int(
        Xfit_t.shape[1] - out["pc_features_used_total"]
    )
    out["best_train_objective"] = float(best_train_obj)
    artifacts = {
        "model": model,
        "alpha_dt": alpha_dt,
        "dt_edges": dt_edges,
        "Xtr_t": Xfit_t,
        "Xte_t": Xte_t,
        "Xtr_np_sub": Xtr_sub_np,
        "Xte_np_sub": Xte_sub_np,
        "standardize_mu": Xfit_np.mean(axis=0, keepdims=True),
        "standardize_sd": np.where(
            (Xfit_np.std(axis=0, keepdims=True) <= 0.0)
            | ~np.isfinite(Xfit_np.std(axis=0, keepdims=True)),
            1.0,
            Xfit_np.std(axis=0, keepdims=True),
        ),
        "g_tr_t": g_fit_t,
        "g_te_t": g_te_t,
        "t_tr": np.asarray(t_fit, float),
        "t_te": np.asarray(t_te, float),
        "e_tr": np.asarray(e_fit, int),
        "e_te": np.asarray(e_te, int),
        "g_tr": np.asarray(g_fit, int),
        "g_te": np.asarray(g_te, int),
        "groups": list(groups),
        "feat_names": list(feat_names_sub),
        "meta_tr": meta_tr.copy() if meta_tr is not None else None,
        "meta_te": meta_te.copy() if meta_te is not None else None,
        "score_fit_mean": np.mean(
            predict_score_samples(model, Xfit_t, g_fit_t, S=FINAL_EVAL_MC_SAMPLES),
            axis=0,
        ),
        "score_te_mean": np.mean(
            predict_score_samples(model, Xte_t, g_te_t, S=FINAL_EVAL_MC_SAMPLES), axis=0
        ),
        "prep_dir": prep_dir,
        # Needed by posthoc cross-fitted recalibration when it refits fold-specific DGPs.
        # Keep both artifact metadata and report metadata aligned with the active config.
        "surv_top_k": int(surv_top_k),
        "shared_dim": int(shared_dim),
        "private_dim": int(private_dim),
        "mt_latent_rank": int(mt_latent_rank),
        "output_rank": int(output_rank),
    }
    return out, artifacts


def _as_torch_from_raw_standardized(X_raw, mu, sd):
    Xs = (np.asarray(X_raw, float) - np.asarray(mu, float)) / np.asarray(sd, float)
    return torch.tensor(Xs, dtype=torch.float32, device=device)


def _feature_index_map(feat_names):
    return {str(nm): int(i) for i, nm in enumerate(feat_names)}


def _counterfactual_treatment_tensor(art, treatment_col, value):
    feat_names = [str(x) for x in art["feat_names"]]
    fmap = _feature_index_map(feat_names)
    if str(treatment_col) not in fmap:
        return None
    Xraw = np.asarray(art["Xte_np_sub"], float).copy()
    Xraw[:, fmap[str(treatment_col)]] = float(value)
    return _as_torch_from_raw_standardized(
        Xraw, art["standardize_mu"], art["standardize_sd"]
    )


def _treatment_profile_label_from_values(values_by_base_col):
    active = []
    for base_col in TREATMENT_FLAG_COLS:
        try:
            v = float(values_by_base_col.get(base_col, 0.0))
        except Exception:
            v = 0.0
        if np.isfinite(v) and v > 0.5:
            active.append(TREATMENT_FLAG_LABELS.get(base_col, base_col))
    if len(active) == 0:
        return "No flagged treatment"
    return " + ".join(active)


def _counterfactual_treatment_profile_tensor(art, profile_values_by_base_col, available_flags=None):
    feat_names = [str(x) for x in art["feat_names"]]
    fmap = _feature_index_map(feat_names)
    if available_flags is None:
        available_flags = _resolve_treatment_feature_columns(feat_names)
    if len(available_flags) == 0:
        return None
    Xraw = np.asarray(art["Xte_np_sub"], float).copy()
    for base_col, design_col in available_flags:
        if str(design_col) not in fmap:
            return None
        Xraw[:, fmap[str(design_col)]] = float(profile_values_by_base_col.get(base_col, 0.0))
    return _as_torch_from_raw_standardized(
        Xraw, art["standardize_mu"], art["standardize_sd"]
    )


def _observed_treatment_profiles_from_design(art, available_flags):
    feat_names = [str(x) for x in art["feat_names"]]
    fmap = _feature_index_map(feat_names)
    Xraw = np.asarray(art["Xte_np_sub"], float)
    if Xraw.ndim != 2 or len(available_flags) == 0:
        return []
    cols = []
    bases = []
    for base_col, design_col in available_flags:
        if str(design_col) in fmap:
            bases.append(str(base_col))
            cols.append(int(fmap[str(design_col)]))
    if len(cols) == 0:
        return []
    A = (Xraw[:, cols] > 0.5).astype(int)
    profiles = []
    for vec in sorted({tuple(row.tolist()) for row in A}, key=lambda z: (sum(z), z)):
        values = {base: float(v) for base, v in zip(bases, vec)}
        label = _treatment_profile_label_from_values(values)
        profiles.append({
            "profile_values": values,
            "treatment_profile": label,
            "treatment_profile_vector": ";".join([f"{base}={int(values.get(base, 0.0))}" for base in bases]),
            "treatment_active_count": int(sum(vec)),
            "is_multi_treatment_profile": bool(sum(vec) > 1),
        })
    return profiles


def _rmst_from_survival_edges(t_edges, S_edges, tau):
    t_edges = np.asarray(t_edges, float)
    S_edges = np.asarray(S_edges, float)
    tau = float(tau)
    if S_edges.ndim != 3 or t_edges.size != S_edges.shape[2]:
        return np.full(S_edges.shape[:2], np.nan, dtype=float)
    total = np.zeros(S_edges.shape[:2], dtype=float)
    for j in range(len(t_edges) - 1):
        left = float(t_edges[j])
        right = float(t_edges[j + 1])
        if left >= tau:
            break
        width = max(0.0, min(right, tau) - left)
        if width > 0.0:
            total += S_edges[:, :, j] * width
    return total


def run_standardized_treatment_effect_analysis(art, outdir):
    """Publication-grade treatment summaries using model-standardized contrasts.

    The analysis avoids tiny observed treatment-specific KM strata.  For each
    treatment flag, predictions are generated twice for the same test-set
    patients, toggling only that treatment indicator to 0 or 1 while preserving
    all other covariates.  Reported effects are model-implied associations, not
    causal effects.
    """
    outdir = Path(outdir)
    model = art["model"]
    alpha_dt = art["alpha_dt"]
    dt_edges = art["dt_edges"]
    g_te = np.asarray(art["g_te"], int)
    g_te_t = art["g_te_t"]
    groups = list(art["groups"])
    feat_names = [str(x) for x in art["feat_names"]]
    available_flags = _resolve_treatment_feature_columns(feat_names)
    rows = []
    rmst_rows = []
    if len(available_flags) == 0:
        pd.DataFrame(
            {
                "message": [
                    "No treatment flag columns found in selected feature matrix."
                ],
                "expected_base_columns": [",".join(TREATMENT_FLAG_COLS)],
                "expected_design_columns": [
                    ",".join([f"clinbin::{c}" for c in TREATMENT_FLAG_COLS])
                ],
            }
        ).to_csv(outdir / "treatment_standardized_effects_skipped.csv", index=False)
        return
    t_edges_ref = None
    for tx_base_col, tx_col in available_flags:
        tx_label = _treatment_label(tx_base_col)
        X0_t = _counterfactual_treatment_tensor(art, tx_col, 0.0)
        X1_t = _counterfactual_treatment_tensor(art, tx_col, 1.0)
        if X0_t is None or X1_t is None:
            continue
        t_edges0, S0_edges = posterior_survival_curve_edges(
            model, X0_t, g_te_t, alpha_dt, dt_edges, S=TREATMENT_EFFECT_MC
        )
        t_edges1, S1_edges = posterior_survival_curve_edges(
            model, X1_t, g_te_t, alpha_dt, dt_edges, S=TREATMENT_EFFECT_MC
        )
        t_edges_ref = t_edges0
        for gi, gname in enumerate(groups):
            mg = g_te == gi
            n_g = int(np.sum(mg))
            if n_g < 3:
                continue
            safe_tx = _safe_fname(tx_label)
            safe_g = _safe_fname(gname)
            avg0 = np.mean(S0_edges[:, mg, :], axis=1)
            avg1 = np.mean(S1_edges[:, mg, :], axis=1)
            m0 = np.mean(avg0, axis=0)
            m1 = np.mean(avg1, axis=0)
            q0 = np.quantile(avg0, [0.05, 0.95], axis=0)
            q1 = np.quantile(avg1, [0.05, 0.95], axis=0)
            fig, ax = plt.subplots(figsize=(7.8, 5.2))
            ax.plot(t_edges0, m0, lw=2.0, label=f"{tx_label}=0")
            ax.fill_between(t_edges0, q0[0], q0[1], alpha=0.18)
            ax.plot(t_edges1, m1, lw=2.0, label=f"{tx_label}=1")
            ax.fill_between(t_edges1, q1[0], q1[1], alpha=0.18)
            ax.set_xlabel("Time since baseline (months)")
            ax.set_ylabel("Mean posterior predictive PFI survival")
            ax.set_title(
                f"{gname}: model-standardized survival contrast for {tx_label}"
            )
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, alpha=0.18)
            ax.legend(
                frameon=True,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=[0, 0, 0.82, 1])
            savefig(outdir / f"TreatmentStandardizedSurvival__{safe_g}__{safe_tx}.png")

            for H in [float(h) for h in TREATMENT_EFFECT_H_LIST]:
                SH0 = posterior_survival_at_H(
                    model, X0_t, g_te_t, H, alpha_dt, dt_edges, S=TREATMENT_EFFECT_MC
                )
                SH1 = posterior_survival_at_H(
                    model, X1_t, g_te_t, H, alpha_dt, dt_edges, S=TREATMENT_EFFECT_MC
                )
                diff_s = np.mean((1.0 - SH1[:, mg]) - (1.0 - SH0[:, mg]), axis=1)
                rows.append(
                    {
                        "H": float(H),
                        "group": str(gname),
                        "treatment_flag": str(tx_base_col),
                        "treatment_feature_column": str(tx_col),
                        "treatment_label": str(tx_label),
                        "n_standardized": int(n_g),
                        "estimand": "mean counterfactual risk difference P(T<=H | TX=1) - P(T<=H | TX=0)",
                        "effect_mean": float(np.mean(diff_s)),
                        "effect_median": float(np.median(diff_s)),
                        "effect_q05": float(np.quantile(diff_s, 0.05)),
                        "effect_q95": float(np.quantile(diff_s, 0.95)),
                        "posterior_mc_samples": int(TREATMENT_EFFECT_MC),
                        "interpretation": "model-implied association; not a causal estimand without exchangeability/positivity assumptions",
                    }
                )
            for tau in [float(x) for x in TREATMENT_EFFECT_RMST_TAU_LIST]:
                rmst0 = _rmst_from_survival_edges(t_edges0, S0_edges[:, mg, :], tau)
                rmst1 = _rmst_from_survival_edges(t_edges1, S1_edges[:, mg, :], tau)
                diff_s = np.mean(rmst1 - rmst0, axis=1)
                rmst_rows.append(
                    {
                        "tau": float(tau),
                        "group": str(gname),
                        "treatment_flag": str(tx_base_col),
                        "treatment_feature_column": str(tx_col),
                        "treatment_label": str(tx_label),
                        "n_standardized": int(n_g),
                        "estimand": "mean counterfactual RMST(TX=1) - RMST(TX=0)",
                        "effect_mean_months": float(np.mean(diff_s)),
                        "effect_median_months": float(np.median(diff_s)),
                        "effect_q05_months": float(np.quantile(diff_s, 0.05)),
                        "effect_q95_months": float(np.quantile(diff_s, 0.95)),
                        "posterior_mc_samples": int(TREATMENT_EFFECT_MC),
                        "interpretation": "model-implied association; not a causal estimand without exchangeability/positivity assumptions",
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(
        outdir / "treatment_standardized_horizon_risk_differences.csv", index=False
    )
    if len(df):
        for H, dfh in df.groupby("H"):
            dfh = dfh.copy()
            dfh["label"] = (
                dfh["group"].astype(str) + " | " + dfh["treatment_label"].astype(str)
            )
            dfh = dfh.sort_values("effect_mean").reset_index(drop=True)
            y = np.arange(len(dfh))
            x = dfh["effect_mean"].to_numpy(float)
            lo = dfh["effect_q05"].to_numpy(float)
            hi = dfh["effect_q95"].to_numpy(float)
            fig, ax = plt.subplots(figsize=(8.2, max(5.0, 0.34 * len(dfh) + 1.5)))
            ax.axvline(0.0, lw=1.0, ls="--")
            ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", capsize=3)
            ax.set_yticks(y)
            ax.set_yticklabels(dfh["label"].tolist())
            ax.set_xlabel(
                f"Risk difference at H={int(H)} mo: P(failure | TX=1) - P(failure | TX=0)"
            )
            ax.set_title(
                f"Model-standardized treatment risk contrasts with posterior 90% intervals"
            )
            ax.grid(True, axis="x", alpha=0.18)
            savefig(outdir / f"TreatmentStandardizedRiskDifferenceForest_H{int(H)}.png")
    dfrmst = pd.DataFrame(rmst_rows)
    dfrmst.to_csv(outdir / "treatment_standardized_rmst_differences.csv", index=False)
    if len(dfrmst):
        for tau, dft in dfrmst.groupby("tau"):
            dft = dft.copy()
            dft["label"] = (
                dft["group"].astype(str) + " | " + dft["treatment_label"].astype(str)
            )
            dft = dft.sort_values("effect_mean_months").reset_index(drop=True)
            y = np.arange(len(dft))
            x = dft["effect_mean_months"].to_numpy(float)
            lo = dft["effect_q05_months"].to_numpy(float)
            hi = dft["effect_q95_months"].to_numpy(float)
            fig, ax = plt.subplots(figsize=(8.2, max(5.0, 0.34 * len(dft) + 1.5)))
            ax.axvline(0.0, lw=1.0, ls="--")
            ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", capsize=3)
            ax.set_yticks(y)
            ax.set_yticklabels(dft["label"].tolist())
            ax.set_xlabel(
                f"RMST difference through {int(tau)} mo: TX=1 minus TX=0 (months)"
            )
            ax.set_title(
                "Model-standardized treatment RMST contrasts with posterior 90% intervals"
            )
            ax.grid(True, axis="x", alpha=0.18)
            savefig(
                outdir / f"TreatmentStandardizedRMSTDifferenceForest_tau{int(tau)}.png"
            )


def _early_failure_counts(times, events, groups_idx, groups, split_name):
    """Return transparent fixed-horizon event counts and prevalence.

    The known-by-H subset is exactly the fixed-horizon binary evaluation set:
    early PFI failures by H plus patients observed event-free past H. Patients
    censored before H are counted separately and excluded from naive known-by-H
    prevalence. IPCW prevalence is also saved for the same horizon.
    """
    rows = []
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    groups_idx = np.asarray(groups_idx, int)

    def _one_row(H, mask, group_name):
        mask = np.asarray(mask, dtype=bool) & np.isfinite(times) & np.isfinite(events)
        n_total = int(np.sum(mask))
        early = mask & (events == 1) & (times <= float(H))
        event_free = mask & (times > float(H))
        censored_before = mask & (events == 0) & (times <= float(H))
        known = early | event_free
        valid_ipcw, y_ipcw, w_ipcw = _ipcw_binary_event_targets(
            times[mask], events[mask], float(H)
        )
        eval_ipcw = (
            np.asarray(valid_ipcw, bool)
            & np.isfinite(w_ipcw)
            & (np.asarray(w_ipcw, float) > 0.0)
        )
        yy = np.asarray(y_ipcw, int)[eval_ipcw]
        ww = np.asarray(w_ipcw, float)[eval_ipcw]
        weighted_n_early = float(np.sum(ww * yy)) if ww.size else 0.0
        ipcw_weight_sum = float(np.sum(ww)) if ww.size else 0.0
        return {
            "split": str(split_name),
            "H": float(H),
            "group": str(group_name),
            "n_total": n_total,
            "n_known_by_H": int(np.sum(known)),
            "n_early_events": int(np.sum(early)),
            "n_event_free_at_H": int(np.sum(event_free)),
            "n_censored_before_H": int(np.sum(censored_before)),
            "known_fraction": float(np.sum(known) / max(n_total, 1)),
            "censored_before_H_fraction": float(
                np.sum(censored_before) / max(n_total, 1)
            ),
            "early_event_rate_known": float(np.sum(early) / max(np.sum(known), 1)),
            "early_event_rate_total_naive": float(np.sum(early) / max(n_total, 1)),
            "event_free_rate_known": float(np.sum(event_free) / max(np.sum(known), 1)),
            "ipcw_weight_sum": ipcw_weight_sum,
            "weighted_n_early_events": weighted_n_early,
            "baseline_ipcw_event_rate": (
                float(weighted_n_early / max(ipcw_weight_sum, 1e-8))
                if ipcw_weight_sum > 0.0
                else np.nan
            ),
            "estimation_note": "known_by_H excludes censoring before H; IPCW prevalence uses censoring weights",
        }

    for H in [float(h) for h in EARLY_FAILURE_H_LIST]:
        finite_all = np.isfinite(times) & np.isfinite(events)
        rows.append(_one_row(H, finite_all, "pooled"))
        for gi, gname in enumerate(groups):
            rows.append(_one_row(H, groups_idx == gi, str(gname)))
    return rows


def _plot_top_early_failure_patient_distributions(
    P,
    p_mean,
    p_q05,
    p_q95,
    valid,
    y,
    status,
    g_te,
    groups,
    H,
    outdir,
    top_n=EARLY_FAILURE_TOP_PATIENT_N,
    file_prefix=None,
    title_prefix=None,
    row_extra=None,
):
    """Plot posterior predictive distributions for the highest-risk test patients.

    Each row is one test patient selected by posterior mean P(T <= H).  The
    horizontal violin shows the full MC posterior distribution for that patient,
    the thin interval shows the central 90% posterior interval, and the marker
    shape records the observed fixed-horizon status when known.
    """
    P = np.asarray(P, dtype=float)
    p_mean = np.asarray(p_mean, dtype=float)
    p_q05 = np.asarray(p_q05, dtype=float)
    p_q95 = np.asarray(p_q95, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    y = np.asarray(y, dtype=int)
    status = np.asarray(status, dtype=object)
    g_te = np.asarray(g_te, dtype=int)
    groups = list(groups)
    row_extra = dict(row_extra or {})

    finite_patient = np.isfinite(p_mean) & np.isfinite(p_q05) & np.isfinite(p_q95)
    if P.ndim != 2 or P.shape[1] != p_mean.shape[0] or int(np.sum(finite_patient)) == 0:
        return []

    ranked = np.argsort(-p_mean[finite_patient])
    patient_pool = np.where(finite_patient)[0][ranked]
    top = patient_pool[: max(1, min(int(top_n), patient_pool.size))]
    if top.size == 0:
        return []

    # Highest predicted-risk patient at the top of the figure.
    top = np.asarray(top, dtype=int)
    n_top = int(top.size)
    y_positions = np.arange(n_top, 0, -1)

    dist_data = []
    kept_positions = []
    kept_indices = []
    for pos, idx in zip(y_positions, top):
        vals = np.asarray(P[:, idx], dtype=float)
        vals = vals[np.isfinite(vals)]
        vals = np.clip(vals, 0.0, 1.0)
        if vals.size == 0:
            continue
        # Matplotlib's violinplot is more stable with at least two values.
        if vals.size == 1:
            vals = np.repeat(vals, 2)
        dist_data.append(vals)
        kept_positions.append(pos)
        kept_indices.append(int(idx))
    if len(dist_data) == 0:
        return []

    fig_h = max(6.2, 0.34 * len(dist_data) + 2.0)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    parts = ax.violinplot(
        dist_data,
        positions=kept_positions,
        vert=False,
        widths=0.74,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts.get("bodies", []):
        body.set_alpha(0.28)
        body.set_linewidth(0.8)

    status_labels = {
        "early_failure": "early failure by H",
        "event_free_at_H": "event-free through H",
        "censored_before_H": "censored before H",
    }
    marker_by_status = {
        "early_failure": "X",
        "event_free_at_H": "o",
        "censored_before_H": "^",
    }

    rows = []
    tick_labels = []
    for rank, (pos, idx) in enumerate(zip(kept_positions, kept_indices), start=1):
        vals = np.asarray(P[:, idx], dtype=float)
        vals = vals[np.isfinite(vals)]
        vals = np.clip(vals, 0.0, 1.0)
        stat = str(status[idx])
        marker = marker_by_status.get(stat, "o")
        group_name = (
            str(groups[int(g_te[idx])])
            if 0 <= int(g_te[idx]) < len(groups)
            else str(g_te[idx])
        )
        ax.hlines(pos, p_q05[idx], p_q95[idx], lw=1.6, alpha=0.95)
        ax.scatter(
            p_mean[idx],
            pos,
            s=42,
            marker=marker,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        tick_labels.append(f"{rank}. {group_name} | test idx {idx}")
        row = {
            "H": float(H),
            "risk_rank": int(rank),
            "patient_index_test": int(idx),
            "group": group_name,
            "label_status": stat,
            "early_failure_label": int(y[idx]) if valid[idx] else np.nan,
            "p_failure_mean": float(p_mean[idx]),
            "p_failure_q05": float(p_q05[idx]),
            "p_failure_q25": (
                float(np.quantile(vals, 0.25)) if vals.size else np.nan
            ),
            "p_failure_q50": (
                float(np.quantile(vals, 0.50)) if vals.size else np.nan
            ),
            "p_failure_q75": (
                float(np.quantile(vals, 0.75)) if vals.size else np.nan
            ),
            "p_failure_q95": float(p_q95[idx]),
            "posterior_mc_samples": int(vals.size),
        }
        row.update(row_extra)
        rows.append(row)

    ax.set_yticks(kept_positions)
    ax.set_yticklabels(tick_labels, fontsize=8.5)
    ax.set_xlabel(f"Posterior predicted P(PFI event by {int(float(H))} months)")
    ax.set_ylabel("Highest-risk test patients")
    if title_prefix is None:
        ax.set_title(f"Highest predicted early-failure risk, H={int(float(H))} months")
    else:
        ax.set_title(f"{title_prefix}, H={int(float(H))} months")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, axis="x", alpha=0.18)

    legend_handles = []
    legend_labels = []
    for stat, lab in status_labels.items():
        if any(str(status[idx]) == stat for idx in kept_indices):
            h = ax.scatter(
                [],
                [],
                marker=marker_by_status.get(stat, "o"),
                s=46,
                edgecolors="black",
                linewidths=0.5,
            )
            legend_handles.append(h)
            legend_labels.append(lab)
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            ncol=1,
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0, 0, 0.82, 1])

    if file_prefix is None:
        fname = f"EarlyFailure_Top{int(len(kept_indices))}_PatientPosteriorDistributions_H{int(float(H))}.png"
    else:
        fname = f"{_safe_fname(file_prefix)}_Top{int(len(kept_indices))}_PatientPosteriorDistributions_H{int(float(H))}.png"
    savefig(outdir / fname)
    return rows


def _plot_within_cancer_top_early_failure_patient_distributions(
    P,
    p_mean,
    p_q05,
    p_q95,
    valid,
    y,
    status,
    g_te,
    groups,
    H,
    outdir,
    top_n_per_group=EARLY_FAILURE_TOP_PATIENT_WITHIN_CANCER_N,
    file_prefix=None,
    title_prefix=None,
    row_extra=None,
):
    """Plot top predicted-risk patients separately within each cancer group.

    This is intentionally a posthoc audit/visualization only; it does not change
    training, prediction, or model selection.  It answers whether the highest
    predicted early-failure-risk patients inside Breast, Colon, and Lung have
    similar or different posterior probability distributions.
    """
    P = np.asarray(P, dtype=float)
    p_mean = np.asarray(p_mean, dtype=float)
    p_q05 = np.asarray(p_q05, dtype=float)
    p_q95 = np.asarray(p_q95, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    y = np.asarray(y, dtype=int)
    status = np.asarray(status, dtype=object)
    g_te = np.asarray(g_te, dtype=int)
    groups = list(groups)
    row_extra = dict(row_extra or {})

    finite_patient = np.isfinite(p_mean) & np.isfinite(p_q05) & np.isfinite(p_q95)
    if P.ndim != 2 or P.shape[1] != p_mean.shape[0] or int(np.sum(finite_patient)) == 0:
        return []

    status_labels = {
        "early_failure": "early failure by H",
        "event_free_at_H": "event-free through H",
        "censored_before_H": "censored before H",
    }
    marker_by_status = {
        "early_failure": "X",
        "event_free_at_H": "o",
        "censored_before_H": "^",
    }

    all_rows = []
    for gi, gname in enumerate(groups):
        in_group = finite_patient & (g_te == int(gi))
        group_pool = np.where(in_group)[0]
        if group_pool.size == 0:
            continue
        group_pool = group_pool[np.argsort(-p_mean[group_pool])]
        top = group_pool[: max(1, min(int(top_n_per_group), group_pool.size))]
        if top.size == 0:
            continue

        n_top = int(top.size)
        y_positions = np.arange(n_top, 0, -1)
        dist_data = []
        kept_positions = []
        kept_indices = []
        for pos, idx in zip(y_positions, top):
            vals = np.asarray(P[:, idx], dtype=float)
            vals = vals[np.isfinite(vals)]
            vals = np.clip(vals, 0.0, 1.0)
            if vals.size == 0:
                continue
            if vals.size == 1:
                vals = np.repeat(vals, 2)
            dist_data.append(vals)
            kept_positions.append(pos)
            kept_indices.append(int(idx))
        if len(dist_data) == 0:
            continue

        fig_h = max(4.8, 0.48 * len(dist_data) + 1.8)
        fig, ax = plt.subplots(figsize=(8.8, fig_h))
        parts = ax.violinplot(
            dist_data,
            positions=kept_positions,
            vert=False,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts.get("bodies", []):
            body.set_alpha(0.28)
            body.set_linewidth(0.8)

        tick_labels = []
        for rank, (pos, idx) in enumerate(zip(kept_positions, kept_indices), start=1):
            vals = np.asarray(P[:, idx], dtype=float)
            vals = vals[np.isfinite(vals)]
            vals = np.clip(vals, 0.0, 1.0)
            stat = str(status[idx])
            marker = marker_by_status.get(stat, "o")
            ax.hlines(pos, p_q05[idx], p_q95[idx], lw=1.6, alpha=0.95)
            ax.scatter(
                p_mean[idx],
                pos,
                s=42,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            tick_labels.append(f"{rank}. test idx {idx}")
            row = {
                "H": float(H),
                "selection_scope": "within_cancer",
                "group": str(gname),
                "within_cancer_risk_rank": int(rank),
                "patient_index_test": int(idx),
                "label_status": stat,
                "early_failure_label": int(y[idx]) if valid[idx] else np.nan,
                "p_failure_mean": float(p_mean[idx]),
                "p_failure_q05": float(p_q05[idx]),
                "p_failure_q25": (
                    float(np.quantile(vals, 0.25)) if vals.size else np.nan
                ),
                "p_failure_q50": (
                    float(np.quantile(vals, 0.50)) if vals.size else np.nan
                ),
                "p_failure_q75": (
                    float(np.quantile(vals, 0.75)) if vals.size else np.nan
                ),
                "p_failure_q95": float(p_q95[idx]),
                "posterior_mc_samples": int(vals.size),
            }
            row.update(row_extra)
            all_rows.append(row)

        ax.set_yticks(kept_positions)
        ax.set_yticklabels(tick_labels, fontsize=8.5)
        ax.set_xlabel(f"Posterior predicted P(PFI event by {int(float(H))} months)")
        ax.set_ylabel(f"Top {len(kept_indices)} {gname} test patients")
        if title_prefix is None:
            ax.set_title(
                f"Highest predicted early-failure risk within {gname}, H={int(float(H))} months"
            )
        else:
            ax.set_title(f"{title_prefix} {gname}, H={int(float(H))} months")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, axis="x", alpha=0.18)

        legend_handles = []
        legend_labels = []
        for stat, lab in status_labels.items():
            if any(str(status[idx]) == stat for idx in kept_indices):
                h = ax.scatter(
                    [],
                    [],
                    marker=marker_by_status.get(stat, "o"),
                    s=46,
                    edgecolors="black",
                    linewidths=0.5,
                )
                legend_handles.append(h)
                legend_labels.append(lab)
        if legend_handles:
            ax.legend(
                legend_handles,
                legend_labels,
                frameon=True,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                ncol=1,
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=[0, 0, 0.82, 1])

        safe_g = _safe_fname(str(gname))
        if file_prefix is None:
            fname = f"EarlyFailure_WithinCancerTop{int(len(kept_indices))}_{safe_g}_PatientPosteriorDistributions_H{int(float(H))}.png"
        else:
            fname = f"{_safe_fname(file_prefix)}_WithinCancerTop{int(len(kept_indices))}_{safe_g}_PatientPosteriorDistributions_H{int(float(H))}.png"
        savefig(outdir / fname)

    return all_rows


def _fixed_horizon_ipcw_metrics_from_predictions(p_mean, valid, y, w):
    """Censoring-aware fixed-horizon binary metrics for P(T <= H).

    This is used for within-cancer early-failure auditing.  It evaluates the
    fixed-horizon event label with IPCW weights, so it answers whether predicted
    early-failure probabilities discriminate observed early PFI failures within
    a given cancer group, not just pooled across cancers.
    """
    p_mean = np.asarray(p_mean, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    keep = valid & np.isfinite(p_mean) & np.isfinite(w) & (w > 0.0)
    yy = y[keep].astype(int)
    pp = np.clip(p_mean[keep].astype(float), 0.0, 1.0)
    ww = w[keep].astype(float)
    row = {
        "n_valid": int(yy.size),
        "n_early_events": int(np.sum(yy == 1)),
        "n_event_free_at_H": int(np.sum(yy == 0)),
        "ipcw_weight_sum": float(np.sum(ww)) if ww.size else 0.0,
        "weighted_n_early_events": float(np.sum(ww * yy)) if ww.size else 0.0,
        "baseline_ipcw_event_rate": np.nan,
        "baseline_naive_event_rate": np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
        "ipcw_brier_binary": np.nan,
        "skipped": False,
    }
    if yy.size == 0 or float(np.sum(ww)) <= 0.0:
        row["skipped"] = True
        return row
    row["baseline_ipcw_event_rate"] = float(np.sum(ww * yy) / max(np.sum(ww), 1e-8))
    row["baseline_naive_event_rate"] = float(np.mean(yy))
    row["ipcw_brier_binary"] = float(
        np.sum(ww * (yy - pp) ** 2) / max(np.sum(ww), 1e-8)
    )
    if yy.size < 3 or int(np.sum(yy == 1)) < 2 or int(np.sum(yy == 0)) < 2:
        row["skipped"] = True
        return row
    try:
        row["auroc"] = float(roc_auc_score(yy, pp, sample_weight=ww))
        row["auprc"] = float(average_precision_score(yy, pp, sample_weight=ww))
    except Exception:
        row["skipped"] = True
    return row


def _append_topk_rows_for_scope(rows, P, valid_mask, y, w, H, scope, group_name):
    """Append IPCW top-k precision/recall/lift rows for one scope."""
    P = np.asarray(P, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    n_valid = int(np.sum(valid_mask))
    if P.ndim != 2 or n_valid <= 0:
        return
    yv_all = y[valid_mask].astype(float)
    wv_all = w[valid_mask].astype(float)
    weighted_n_pos = float(np.sum(wv_all * yv_all))
    baseline = (
        float(weighted_n_pos / max(np.sum(wv_all), 1e-8))
        if np.sum(wv_all) > 0
        else np.nan
    )
    n_pos = int(np.sum(yv_all == 1.0))
    for frac in [float(f) for f in EARLY_FAILURE_TOPK_FRACS]:
        k = max(1, int(np.ceil(frac * n_valid)))
        recalls = []
        precisions = []
        for sidx in range(P.shape[0]):
            pv = np.asarray(P[sidx, valid_mask], dtype=float)
            ok = (
                np.isfinite(pv)
                & np.isfinite(yv_all)
                & np.isfinite(wv_all)
                & (wv_all > 0.0)
            )
            if int(np.sum(ok)) == 0:
                recalls.append(np.nan)
                precisions.append(np.nan)
                continue
            pv_ok = pv[ok]
            yy_ok = yv_all[ok]
            ww_ok = wv_all[ok]
            kk = max(1, min(k, pv_ok.size))
            top = np.argsort(-pv_ok)[:kk]
            weighted_tp = float(np.sum(ww_ok[top] * yy_ok[top]))
            selected_weight = float(np.sum(ww_ok[top]))
            precisions.append(
                weighted_tp / max(selected_weight, 1e-8)
                if selected_weight > 0.0
                else np.nan
            )
            recalls.append(
                weighted_tp / max(weighted_n_pos, 1e-8)
                if weighted_n_pos > 0.0
                else np.nan
            )
        recalls = np.asarray(recalls, dtype=float)
        precisions = np.asarray(precisions, dtype=float)
        rf = recalls[np.isfinite(recalls)]
        pf = precisions[np.isfinite(precisions)]
        pmean = float(np.mean(pf)) if pf.size else np.nan
        rows.append(
            {
                "scope": str(scope),
                "group": str(group_name),
                "H": float(H),
                "top_fraction": float(frac),
                "k": int(k),
                "n_valid": int(n_valid),
                "n_early_events": int(n_pos),
                "weighted_n_early_events": float(weighted_n_pos),
                "baseline_ipcw_event_rate": baseline,
                "precision_mean": pmean,
                "precision_q05": float(np.quantile(pf, 0.05)) if pf.size else np.nan,
                "precision_q95": float(np.quantile(pf, 0.95)) if pf.size else np.nan,
                "precision_lift_mean": (
                    float(pmean / baseline)
                    if np.isfinite(pmean) and np.isfinite(baseline) and baseline > 0.0
                    else np.nan
                ),
                "recall_mean": float(np.mean(rf)) if rf.size else np.nan,
                "recall_q05": float(np.quantile(rf, 0.05)) if rf.size else np.nan,
                "recall_q95": float(np.quantile(rf, 0.95)) if rf.size else np.nan,
                "posterior_mc_samples": int(P.shape[0]),
                "estimation_method": "IPCW top-k within-scope early-failure enrichment using posterior MC ranking uncertainty",
            }
        )


def _append_calibration_rows_for_scope(
    rows, P, p_mean, valid_mask, y, w, H, scope, group_name
):
    """Append IPCW calibration rows for one pooled/group scope."""
    P = np.asarray(P, dtype=float)
    p_mean = np.asarray(p_mean, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    keep = valid_mask & np.isfinite(p_mean) & np.isfinite(w) & (w > 0.0)
    if int(np.sum(keep)) < max(10, EARLY_FAILURE_CALIBRATION_BINS * 2):
        return
    pv = p_mean[keep]
    yv = y[keep]
    wv = w[keep]
    unique_n = len(np.unique(pv[np.isfinite(pv)]))
    if unique_n < 2:
        return
    try:
        bins = pd.qcut(
            pv,
            q=min(EARLY_FAILURE_CALIBRATION_BINS, unique_n),
            duplicates="drop",
        )
        Pk = P[:, keep]
        for b in pd.Series(bins).cat.categories:
            mb = np.asarray(bins == b)
            if int(np.sum(mb)) == 0:
                continue
            obs = float(np.sum(wv[mb] * yv[mb]) / max(np.sum(wv[mb]), 1e-8))
            pred_s = np.sum(Pk[:, mb] * wv[mb][None, :], axis=1) / max(
                np.sum(wv[mb]), 1e-8
            )
            rows.append(
                {
                    "scope": str(scope),
                    "group": str(group_name),
                    "H": float(H),
                    "bin": str(b),
                    "n": int(np.sum(mb)),
                    "ipcw_weight_sum": float(np.sum(wv[mb])),
                    "weighted_n_early_events": float(np.sum(wv[mb] * yv[mb])),
                    "unweighted_early_event_rate": float(np.mean(yv[mb])),
                    "pred_mean": float(np.mean(pred_s)),
                    "pred_q05": float(np.quantile(pred_s, 0.05)),
                    "pred_q95": float(np.quantile(pred_s, 0.95)),
                    "observed_ipcw_event_rate": obs,
                    "abs_calibration_error": float(abs(obs - float(np.mean(pred_s)))),
                    "signed_calibration_error_observed_minus_predicted": float(
                        obs - float(np.mean(pred_s))
                    ),
                    "estimation_method": "IPCW fixed-horizon calibration within scope",
                }
            )
    except Exception:
        return


def _calibration_summary_from_bins(df_bins):
    """Summarize IPCW calibration-bin output by scope, cancer group, and horizon."""
    if not isinstance(df_bins, pd.DataFrame) or len(df_bins) == 0:
        return pd.DataFrame()
    req = {"scope", "group", "H", "n", "pred_mean", "observed_ipcw_event_rate"}
    if not req.issubset(set(df_bins.columns)):
        return pd.DataFrame()
    df = df_bins.copy()
    for col in ["n", "pred_mean", "observed_ipcw_event_rate", "ipcw_weight_sum"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ipcw_weight_sum" not in df.columns:
        df["ipcw_weight_sum"] = df["n"]
    rows = []
    for (scope, group, H), dfg in df.groupby(["scope", "group", "H"], dropna=False):
        dfg = dfg[
            np.isfinite(dfg["pred_mean"]) & np.isfinite(dfg["observed_ipcw_event_rate"])
        ].copy()
        if len(dfg) == 0:
            continue
        err = dfg["observed_ipcw_event_rate"].to_numpy(float) - dfg[
            "pred_mean"
        ].to_numpy(float)
        n_w = dfg["n"].to_numpy(float)
        ipcw_w = dfg["ipcw_weight_sum"].to_numpy(float)
        n_w = np.where(np.isfinite(n_w) & (n_w > 0.0), n_w, 0.0)
        ipcw_w = np.where(np.isfinite(ipcw_w) & (ipcw_w > 0.0), ipcw_w, 0.0)
        n_den = float(np.sum(n_w))
        ipcw_den = float(np.sum(ipcw_w))
        rows.append(
            {
                "scope": str(scope),
                "group": str(group),
                "H": float(H),
                "n_bins": int(len(dfg)),
                "n_patients_in_bins": int(np.nansum(dfg["n"].to_numpy(float))),
                "ipcw_weight_sum_in_bins": ipcw_den,
                "mean_predicted_probability_bin_weighted": (
                    float(np.sum(n_w * dfg["pred_mean"].to_numpy(float)) / n_den)
                    if n_den > 0.0
                    else np.nan
                ),
                "mean_observed_ipcw_event_rate_bin_weighted": (
                    float(
                        np.sum(n_w * dfg["observed_ipcw_event_rate"].to_numpy(float))
                        / n_den
                    )
                    if n_den > 0.0
                    else np.nan
                ),
                "mean_signed_calibration_error_observed_minus_predicted": (
                    float(np.sum(n_w * err) / n_den) if n_den > 0.0 else np.nan
                ),
                "ece_n_weighted": (
                    float(np.sum(n_w * np.abs(err)) / n_den) if n_den > 0.0 else np.nan
                ),
                "ece_ipcw_weighted": (
                    float(np.sum(ipcw_w * np.abs(err)) / ipcw_den)
                    if ipcw_den > 0.0
                    else np.nan
                ),
                "max_abs_calibration_error": (
                    float(np.nanmax(np.abs(err))) if err.size else np.nan
                ),
                "estimation_method": "summary of IPCW fixed-horizon calibration bins",
            }
        )
    return pd.DataFrame(rows)


def _fixed_horizon_metric_core_from_arrays(p, y, w):
    """IPCW fixed-horizon binary metrics on already-selected arrays."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    keep = np.isfinite(p) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    pp = p[keep]
    yy = y[keep].astype(int)
    ww = w[keep].astype(float)
    out = {
        "n_valid": int(yy.size),
        "n_early_events": int(np.sum(yy == 1)),
        "n_event_free_at_H": int(np.sum(yy == 0)),
        "ipcw_weight_sum": float(np.sum(ww)) if ww.size else 0.0,
        "weighted_n_early_events": float(np.sum(ww * yy)) if ww.size else 0.0,
        "baseline_ipcw_event_rate": np.nan,
        "baseline_naive_event_rate": np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
        "ipcw_brier_binary": np.nan,
        "skipped": False,
    }
    if yy.size == 0 or float(np.sum(ww)) <= 0.0:
        out["skipped"] = True
        return out
    out["baseline_ipcw_event_rate"] = float(np.sum(ww * yy) / max(np.sum(ww), 1e-8))
    out["baseline_naive_event_rate"] = float(np.mean(yy))
    out["ipcw_brier_binary"] = float(
        np.sum(ww * (yy - pp) ** 2) / max(np.sum(ww), 1e-8)
    )
    if yy.size < 3 or int(np.sum(yy == 1)) < 2 or int(np.sum(yy == 0)) < 2:
        out["skipped"] = True
        return out
    try:
        out["auroc"] = float(roc_auc_score(yy, pp, sample_weight=ww))
        out["auprc"] = float(average_precision_score(yy, pp, sample_weight=ww))
    except Exception:
        out["skipped"] = True
    return out


def _stratified_bootstrap_indices(mask, y, g, rng, stratify_by_group=True):
    """Sample original indices with replacement within status or group-status strata."""
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(y, dtype=int)
    g = np.asarray(g, dtype=int)
    idx_all = np.where(mask)[0]
    if idx_all.size == 0:
        return idx_all
    pieces = []
    if stratify_by_group:
        keys = sorted(set((int(g[i]), int(y[i])) for i in idx_all))
        for gg, yy in keys:
            idx = idx_all[(g[idx_all] == gg) & (y[idx_all] == yy)]
            if idx.size:
                pieces.append(rng.choice(idx, size=idx.size, replace=True))
    else:
        for yy in sorted(set(int(y[i]) for i in idx_all)):
            idx = idx_all[y[idx_all] == yy]
            if idx.size:
                pieces.append(rng.choice(idx, size=idx.size, replace=True))
    if not pieces:
        return idx_all
    return np.concatenate(pieces)


def _topk_metrics_from_arrays(p, y, w, frac):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    keep = np.isfinite(p) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    pp = p[keep]
    yy = y[keep]
    ww = w[keep]
    n_valid = int(pp.size)
    if n_valid == 0:
        return {
            "k": 0,
            "precision": np.nan,
            "recall": np.nan,
            "precision_lift": np.nan,
            "baseline_ipcw_event_rate": np.nan,
        }
    k = max(1, int(np.ceil(float(frac) * n_valid)))
    order = np.argsort(-pp)[:k]
    weighted_pos = float(np.sum(ww * yy))
    baseline = float(weighted_pos / max(np.sum(ww), 1e-8)) if np.sum(ww) > 0 else np.nan
    selected_weight = float(np.sum(ww[order]))
    weighted_tp = float(np.sum(ww[order] * yy[order]))
    precision = (
        weighted_tp / max(selected_weight, 1e-8) if selected_weight > 0 else np.nan
    )
    recall = weighted_tp / max(weighted_pos, 1e-8) if weighted_pos > 0 else np.nan
    lift = (
        precision / baseline
        if np.isfinite(precision) and np.isfinite(baseline) and baseline > 0
        else np.nan
    )
    return {
        "k": int(k),
        "precision": float(precision) if np.isfinite(precision) else np.nan,
        "recall": float(recall) if np.isfinite(recall) else np.nan,
        "precision_lift": float(lift) if np.isfinite(lift) else np.nan,
        "baseline_ipcw_event_rate": (
            float(baseline) if np.isfinite(baseline) else np.nan
        ),
    }


def _q_summary(vals, qlo=0.05, qhi=0.95):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan, 0
    return (
        float(np.mean(vals)),
        float(np.quantile(vals, qlo)),
        float(np.quantile(vals, qhi)),
        int(vals.size),
    )


def _append_bootstrap_performance_rows(
    disc_rows,
    topk_rows,
    cal_rows,
    p_mean,
    valid_mask,
    y,
    w,
    g_te,
    H,
    scope,
    group_name,
    n_boot=500,
    seed=20260512,
):
    """Bootstrap fixed-horizon early-failure performance for one scope.

    This resamples test patients, not posterior draws.  IPCW weights are fixed
    from the full test set at the target horizon, which avoids unstable censoring
    KM estimates inside tiny bootstrap strata.
    """
    p_mean = np.asarray(p_mean, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    g_te = np.asarray(g_te, dtype=int)
    keep = valid_mask & np.isfinite(p_mean) & np.isfinite(w) & (w > 0.0)
    if int(np.sum(keep)) == 0 or int(n_boot) <= 0:
        return
    qlo, qhi = [float(v) for v in EARLY_FAILURE_BOOTSTRAP_Q]
    group_key = str(group_name)
    group_offset = sum((ii + 1) * ord(ch) for ii, ch in enumerate(group_key)) % 100000
    rng_seed = (
        int(seed)
        + int(round(float(H) * 100.0))
        + (0 if str(scope) == "pooled" else 10000 + group_offset)
    )
    rng = np.random.default_rng(rng_seed)
    stratify_by_group = str(scope) == "pooled"

    point = _fixed_horizon_metric_core_from_arrays(p_mean[keep], y[keep], w[keep])
    boot_metrics = {
        "auroc": [],
        "auprc": [],
        "ipcw_brier_binary": [],
        "baseline_ipcw_event_rate": [],
    }
    boot_topk = {
        float(fr): {"precision": [], "recall": [], "precision_lift": []}
        for fr in EARLY_FAILURE_TOPK_FRACS
    }

    # Define calibration bins from the original scope once, then resample patients.
    cal_bin_info = []
    try:
        pv0 = p_mean[keep]
        unique_n = len(np.unique(pv0[np.isfinite(pv0)]))
        if (
            int(np.sum(keep)) >= max(10, EARLY_FAILURE_CALIBRATION_BINS * 2)
            and unique_n >= 2
        ):
            bins0 = pd.qcut(
                pv0,
                q=min(EARLY_FAILURE_CALIBRATION_BINS, unique_n),
                duplicates="drop",
            )
            scope_indices = np.where(keep)[0]
            for b in pd.Series(bins0).cat.categories:
                mb_local = np.asarray(bins0 == b)
                if int(np.sum(mb_local)):
                    cal_bin_info.append((str(b), set(scope_indices[mb_local].tolist())))
    except Exception:
        cal_bin_info = []
    cal_boot = {
        b: {"observed_ipcw_event_rate": [], "pred_mean": [], "n": []}
        for b, _ in cal_bin_info
    }

    for _ in range(int(n_boot)):
        bs_idx = _stratified_bootstrap_indices(
            keep, y, g_te, rng, stratify_by_group=stratify_by_group
        )
        if bs_idx.size == 0:
            continue
        mrow = _fixed_horizon_metric_core_from_arrays(
            p_mean[bs_idx], y[bs_idx], w[bs_idx]
        )
        for metric in boot_metrics:
            boot_metrics[metric].append(mrow.get(metric, np.nan))
        for fr in EARLY_FAILURE_TOPK_FRACS:
            trow = _topk_metrics_from_arrays(p_mean[bs_idx], y[bs_idx], w[bs_idx], fr)
            for metric in boot_topk[float(fr)]:
                boot_topk[float(fr)][metric].append(trow.get(metric, np.nan))
        if cal_bin_info:
            bs_idx_arr = np.asarray(bs_idx, dtype=int)
            for b, bin_set in cal_bin_info:
                in_bin = np.array([int(i) in bin_set for i in bs_idx_arr], dtype=bool)
                if not np.any(in_bin):
                    continue
                ii = bs_idx_arr[in_bin]
                ww = w[ii]
                yy = y[ii]
                pp = p_mean[ii]
                denom = float(np.sum(ww))
                if denom <= 0.0:
                    continue
                cal_boot[b]["observed_ipcw_event_rate"].append(
                    float(np.sum(ww * yy) / denom)
                )
                cal_boot[b]["pred_mean"].append(float(np.sum(ww * pp) / denom))
                cal_boot[b]["n"].append(int(ii.size))

    drow = {
        "scope": str(scope),
        "group": str(group_name),
        "H": float(H),
        "bootstrap_n_requested": int(n_boot),
        "bootstrap_method": "stratified patient bootstrap with fixed full-test IPCW weights",
        "n_valid": int(point.get("n_valid", 0)),
        "n_early_events": int(point.get("n_early_events", 0)),
        "n_event_free_at_H": int(point.get("n_event_free_at_H", 0)),
    }
    for metric in ["auroc", "auprc", "ipcw_brier_binary", "baseline_ipcw_event_rate"]:
        mean, lo, hi, n_ok = _q_summary(boot_metrics[metric], qlo, qhi)
        drow[f"{metric}_point"] = (
            float(point.get(metric, np.nan))
            if np.isfinite(point.get(metric, np.nan))
            else np.nan
        )
        drow[f"{metric}_boot_mean"] = mean
        drow[f"{metric}_boot_q{int(qlo*100):02d}"] = lo
        drow[f"{metric}_boot_q{int(qhi*100):02d}"] = hi
        drow[f"{metric}_boot_n_success"] = int(n_ok)
    disc_rows.append(drow)

    for fr in EARLY_FAILURE_TOPK_FRACS:
        point_top = _topk_metrics_from_arrays(p_mean[keep], y[keep], w[keep], fr)
        row = {
            "scope": str(scope),
            "group": str(group_name),
            "H": float(H),
            "top_fraction": float(fr),
            "k_point": int(point_top.get("k", 0)),
            "bootstrap_n_requested": int(n_boot),
            "bootstrap_method": "stratified patient bootstrap with fixed full-test IPCW weights",
        }
        for metric in ["precision", "recall", "precision_lift"]:
            mean, lo, hi, n_ok = _q_summary(boot_topk[float(fr)][metric], qlo, qhi)
            row[f"{metric}_point"] = (
                float(point_top.get(metric, np.nan))
                if np.isfinite(point_top.get(metric, np.nan))
                else np.nan
            )
            row[f"{metric}_boot_mean"] = mean
            row[f"{metric}_boot_q{int(qlo*100):02d}"] = lo
            row[f"{metric}_boot_q{int(qhi*100):02d}"] = hi
            row[f"{metric}_boot_n_success"] = int(n_ok)
        topk_rows.append(row)

    for b, _ in cal_bin_info:
        obs_mean, obs_lo, obs_hi, obs_n = _q_summary(
            cal_boot[b]["observed_ipcw_event_rate"], qlo, qhi
        )
        pred_mean, pred_lo, pred_hi, pred_n = _q_summary(
            cal_boot[b]["pred_mean"], qlo, qhi
        )
        n_mean, n_lo, n_hi, n_n = _q_summary(cal_boot[b]["n"], qlo, qhi)
        cal_rows.append(
            {
                "scope": str(scope),
                "group": str(group_name),
                "H": float(H),
                "bin": str(b),
                "bootstrap_n_requested": int(n_boot),
                "bootstrap_method": "stratified patient bootstrap with fixed full-test IPCW weights and original risk-bin definitions",
                "observed_ipcw_event_rate_boot_mean": obs_mean,
                f"observed_ipcw_event_rate_boot_q{int(qlo*100):02d}": obs_lo,
                f"observed_ipcw_event_rate_boot_q{int(qhi*100):02d}": obs_hi,
                "observed_ipcw_event_rate_boot_n_success": int(obs_n),
                "pred_mean_boot_mean": pred_mean,
                f"pred_mean_boot_q{int(qlo*100):02d}": pred_lo,
                f"pred_mean_boot_q{int(qhi*100):02d}": pred_hi,
                "pred_mean_boot_n_success": int(pred_n),
                "n_boot_mean": n_mean,
                f"n_boot_q{int(qlo*100):02d}": n_lo,
                f"n_boot_q{int(qhi*100):02d}": n_hi,
                "n_boot_n_success": int(n_n),
            }
        )


def _plot_bootstrap_topk_summaries(df_boot_topk, outdir):
    """Save bootstrap performance-uncertainty top-k plots."""
    if not isinstance(df_boot_topk, pd.DataFrame) or not len(df_boot_topk):
        return
    outdir = Path(outdir)
    qlo, qhi = [float(v) for v in EARLY_FAILURE_BOOTSTRAP_Q]
    lo_col = f"precision_boot_q{int(qlo*100):02d}"
    hi_col = f"precision_boot_q{int(qhi*100):02d}"
    for scope_name, dfscope in df_boot_topk.groupby("scope"):
        for H, dfh in dfscope.groupby("H"):
            fig, ax = plt.subplots(figsize=(7.4, 5.1))
            for gname, dfg in dfh.groupby("group"):
                dfg = dfg.sort_values("top_fraction")
                x = 100.0 * dfg["top_fraction"].to_numpy(dtype=float)
                yv = dfg["precision_point"].to_numpy(dtype=float)
                lo = (
                    dfg[lo_col].to_numpy(dtype=float)
                    if lo_col in dfg.columns
                    else np.full_like(yv, np.nan)
                )
                hi = (
                    dfg[hi_col].to_numpy(dtype=float)
                    if hi_col in dfg.columns
                    else np.full_like(yv, np.nan)
                )
                ax.plot(x, yv, marker="o", lw=2.0, label=str(gname))
                ax.fill_between(x, lo, hi, alpha=0.14)
            ax.set_xlabel("Top predicted-risk fraction within scope (%)")
            ax.set_ylabel("Early-failure precision")
            title_scope = (
                "within-cancer" if str(scope_name) == "within_cancer" else "pooled"
            )
            ax.set_title(
                f"Bootstrap top-k early-failure precision, {title_scope}, H={int(float(H))} months"
            )
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, alpha=0.18)
            # Keep the legend outside the plotting area to avoid covering data.
            if len(dfh["group"].unique()) > 1:
                ax.legend(
                    frameon=True,
                    bbox_to_anchor=(1.02, 1.0),
                    loc="upper left",
                    borderaxespad=0.0,
                )
                fig.tight_layout(rect=[0, 0, 0.82, 1])
            savefig(
                outdir
                / f"EarlyFailure_BootstrapTopKPrecision_{_safe_fname(scope_name)}_H{int(float(H))}.png"
            )


def _plot_within_cancer_early_failure_summaries(df_disc, df_topk, df_cal, outdir):
    """Save compact within-cancer early-failure audit plots."""
    outdir = Path(outdir)
    # Heatmaps for within-cancer AUROC/AUPRC/Brier/baseline.
    if isinstance(df_disc, pd.DataFrame) and len(df_disc):
        dfg = df_disc[df_disc["scope"].astype(str) == "within_cancer"].copy()
        if len(dfg):
            for metric, label in [
                ("auroc", "IPCW AUROC"),
                ("auprc", "IPCW AUPRC"),
                ("baseline_ipcw_event_rate", "Baseline IPCW event rate"),
                ("ipcw_brier_binary", "IPCW binary Brier"),
            ]:
                if metric not in dfg.columns:
                    continue
                piv = dfg.pivot(index="group", columns="H", values=metric)
                if piv.empty:
                    continue
                fig, ax = plt.subplots(
                    figsize=(
                        1.25 * max(4, piv.shape[1]) + 2.2,
                        0.65 * max(3, piv.shape[0]) + 1.7,
                    )
                )
                im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto")
                ax.set_xticks(np.arange(piv.shape[1]))
                ax.set_xticklabels([str(int(float(c))) for c in piv.columns])
                ax.set_yticks(np.arange(piv.shape[0]))
                ax.set_yticklabels([str(i) for i in piv.index])
                ax.set_xlabel("Horizon (months)")
                ax.set_title(f"Within-cancer early-failure {label}")
                for ii in range(piv.shape[0]):
                    for jj in range(piv.shape[1]):
                        val = piv.to_numpy(dtype=float)[ii, jj]
                        txt = "NA" if not np.isfinite(val) else f"{val:.3f}"
                        ax.text(jj, ii, txt, ha="center", va="center", fontsize=9)
                cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
                cb.set_label(label)
                savefig(outdir / f"EarlyFailure_WithinCancer_{metric}_Heatmap.png")

    # Within-cancer top-k lines by horizon.
    if isinstance(df_topk, pd.DataFrame) and len(df_topk):
        dfg = df_topk[df_topk["scope"].astype(str) == "within_cancer"].copy()
        if len(dfg):
            for H, dfh in dfg.groupby("H"):
                for metric, label in [
                    ("precision_mean", "IPCW precision"),
                    ("precision_lift_mean", "IPCW precision lift over group baseline"),
                    ("recall_mean", "IPCW recall"),
                ]:
                    if metric not in dfh.columns:
                        continue
                    fig, ax = plt.subplots(figsize=(7.4, 5.1))
                    for gname, dfg2 in dfh.groupby("group"):
                        dfg2 = dfg2.sort_values("top_fraction")
                        x = 100.0 * dfg2["top_fraction"].to_numpy(dtype=float)
                        yv = dfg2[metric].to_numpy(dtype=float)
                        ax.plot(x, yv, marker="o", lw=2.0, label=str(gname))
                    ax.set_xlabel("Top predicted-risk fraction within cancer (%)")
                    ax.set_ylabel(label)
                    ax.set_title(
                        f"Within-cancer top-k early-failure enrichment, H={int(float(H))} months"
                    )
                    ax.grid(True, alpha=0.18)
                    if metric != "precision_lift_mean":
                        ax.set_ylim(0.0, 1.02)
                    ax.legend(
                        title="Cancer group",
                        frameon=True,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0,
                    )
                    fig.tight_layout(rect=[0, 0, 0.82, 1])
                    savefig(
                        outdir
                        / f"EarlyFailure_WithinCancer_TopK_{metric}_H{int(float(H))}.png"
                    )

    # Calibration curves by cancer for each horizon.
    if isinstance(df_cal, pd.DataFrame) and len(df_cal):
        dfg = df_cal[df_cal["scope"].astype(str) == "within_cancer"].copy()
        if len(dfg):
            for H, dfh in dfg.groupby("H"):
                fig, ax = plt.subplots(figsize=(6.2, 5.6))
                max_l = 0.05
                for gname, dfg2 in dfh.groupby("group"):
                    dfg2 = dfg2.sort_values("pred_mean")
                    x = dfg2["pred_mean"].to_numpy(dtype=float)
                    yv = dfg2["observed_ipcw_event_rate"].to_numpy(dtype=float)
                    lo = dfg2["pred_q05"].to_numpy(dtype=float)
                    hi = dfg2["pred_q95"].to_numpy(dtype=float)
                    ax.errorbar(
                        x,
                        yv,
                        xerr=np.vstack([x - lo, hi - x]),
                        fmt="o-",
                        capsize=3,
                        label=str(gname),
                    )
                    if x.size and yv.size:
                        max_l = max(
                            max_l, float(np.nanmax([np.nanmax(x), np.nanmax(yv)]))
                        )
                ax.plot([0, max_l], [0, max_l], ls="--", lw=1.0, color="gray")
                ax.set_xlim(0, min(1.0, max_l * 1.15))
                ax.set_ylim(0, min(1.0, max_l * 1.15))
                ax.set_xlabel("Mean predicted early-failure probability")
                ax.set_ylabel("Observed IPCW early-failure rate")
                ax.set_title(
                    f"Within-cancer early-failure calibration, H={int(float(H))} months"
                )
                ax.grid(True, alpha=0.18)
                ax.legend(
                    title="Cancer group",
                    frameon=True,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                )
                fig.tight_layout(rect=[0, 0, 0.80, 1])
                savefig(
                    outdir
                    / f"EarlyFailure_WithinCancer_Calibration_H{int(float(H))}.png"
                )


def run_early_failure_analysis(art, outdir):
    outdir = Path(outdir)
    model = art["model"]
    alpha_dt = art["alpha_dt"]
    dt_edges = art["dt_edges"]
    Xte_t = art["Xte_t"]
    g_te_t = art["g_te_t"]
    t_te = np.asarray(art["t_te"], float)
    e_te = np.asarray(art["e_te"], int)
    g_te = np.asarray(art["g_te"], int)
    groups = list(art["groups"])
    count_rows = []
    count_rows.extend(
        _early_failure_counts(art["t_tr"], art["e_tr"], art["g_tr"], groups, "train")
    )
    count_rows.extend(_early_failure_counts(t_te, e_te, g_te, groups, "test"))
    pd.DataFrame(count_rows).to_csv(
        outdir / "early_failure_event_counts.csv", index=False
    )

    topk_rows = []
    calib_rows = []
    pred_rows = []
    top_patient_distribution_rows = []
    within_cancer_top_patient_distribution_rows = []
    within_cancer_metric_rows = []
    within_cancer_topk_rows = []
    within_cancer_calibration_rows = []
    bootstrap_metric_rows = []
    bootstrap_topk_rows = []
    bootstrap_calibration_rows = []
    for H in [float(h) for h in EARLY_FAILURE_H_LIST]:
        SH = posterior_survival_at_H(
            model, Xte_t, g_te_t, H, alpha_dt, dt_edges, S=EARLY_FAILURE_MC
        )
        P = 1.0 - np.clip(SH, 0.0, 1.0)  # posterior samples of failure probability by H
        p_mean = np.mean(P, axis=0)
        p_q05 = np.quantile(P, 0.05, axis=0)
        p_q95 = np.quantile(P, 0.95, axis=0)
        valid, y, w = _ipcw_binary_event_targets(t_te, e_te, H)
        valid = np.asarray(valid, bool) & np.isfinite(p_mean) & np.isfinite(w) & (w > 0)
        y = np.asarray(y, int)
        w = np.asarray(w, float)
        status = np.full(t_te.shape[0], "censored_before_H", dtype=object)
        status[valid & (y == 1)] = "early_failure"
        status[valid & (y == 0)] = "event_free_at_H"

        # Within-cancer early-failure audit.  These summaries separate true
        # patient-level discrimination inside Breast/Colon/Lung from pooled
        # enrichment that can be driven by cancer-type baseline risk.
        metric_row = _fixed_horizon_ipcw_metrics_from_predictions(p_mean, valid, y, w)
        metric_row.update({"scope": "pooled", "group": "pooled", "H": float(H)})
        within_cancer_metric_rows.append(metric_row)
        _append_topk_rows_for_scope(
            within_cancer_topk_rows,
            P,
            valid,
            y,
            w,
            H,
            scope="pooled",
            group_name="pooled",
        )
        _append_calibration_rows_for_scope(
            within_cancer_calibration_rows,
            P,
            p_mean,
            valid,
            y,
            w,
            H,
            scope="pooled",
            group_name="pooled",
        )
        for gi, gname in enumerate(groups):
            mg = valid & (g_te == int(gi))
            metric_row = _fixed_horizon_ipcw_metrics_from_predictions(p_mean, mg, y, w)
            metric_row.update(
                {"scope": "within_cancer", "group": str(gname), "H": float(H)}
            )
            within_cancer_metric_rows.append(metric_row)
            _append_topk_rows_for_scope(
                within_cancer_topk_rows,
                P,
                mg,
                y,
                w,
                H,
                scope="within_cancer",
                group_name=str(gname),
            )
            _append_calibration_rows_for_scope(
                within_cancer_calibration_rows,
                P,
                p_mean,
                mg,
                y,
                w,
                H,
                scope="within_cancer",
                group_name=str(gname),
            )

        if int(EARLY_FAILURE_BOOTSTRAP_N) > 0:
            _append_bootstrap_performance_rows(
                bootstrap_metric_rows,
                bootstrap_topk_rows,
                bootstrap_calibration_rows,
                p_mean,
                valid,
                y,
                w,
                g_te,
                H,
                scope="pooled",
                group_name="pooled",
                n_boot=EARLY_FAILURE_BOOTSTRAP_N,
                seed=EARLY_FAILURE_BOOTSTRAP_SEED,
            )
            for gi, gname in enumerate(groups):
                mg = valid & (g_te == int(gi))
                _append_bootstrap_performance_rows(
                    bootstrap_metric_rows,
                    bootstrap_topk_rows,
                    bootstrap_calibration_rows,
                    p_mean,
                    mg,
                    y,
                    w,
                    g_te,
                    H,
                    scope="within_cancer",
                    group_name=str(gname),
                    n_boot=EARLY_FAILURE_BOOTSTRAP_N,
                    seed=EARLY_FAILURE_BOOTSTRAP_SEED,
                )

        for i in range(len(t_te)):
            pred_rows.append(
                {
                    "H": float(H),
                    "patient_index_test": int(i),
                    "group": (
                        str(groups[int(g_te[i])])
                        if 0 <= int(g_te[i]) < len(groups)
                        else str(g_te[i])
                    ),
                    "time_months": float(t_te[i]) if np.isfinite(t_te[i]) else np.nan,
                    "event": int(e_te[i]) if np.isfinite(e_te[i]) else np.nan,
                    "label_status": str(status[i]),
                    "early_failure_label": int(y[i]) if valid[i] else np.nan,
                    "ipcw_weight": float(w[i]) if valid[i] else np.nan,
                    "p_failure_mean": float(p_mean[i]),
                    "p_failure_q05": float(p_q05[i]),
                    "p_failure_q95": float(p_q95[i]),
                }
            )
        # Waterfall plot with posterior 90% band, sorted by posterior mean risk.
        order = np.argsort(p_mean)
        x = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        ax.fill_between(
            x, p_q05[order], p_q95[order], alpha=0.18, label="posterior 90% interval"
        )
        ax.plot(x, p_mean[order], lw=1.8, label="posterior mean")
        ev_ord = valid[order] & (y[order] == 1)
        non_ord = valid[order] & (y[order] == 0)
        ax.scatter(
            x[non_ord],
            p_mean[order][non_ord],
            s=12,
            alpha=0.65,
            label="event-free at H",
        )
        ax.scatter(
            x[ev_ord], p_mean[order][ev_ord], s=28, marker="x", label="early failure"
        )
        ax.set_xlabel("Test patients sorted by predicted early-failure probability")
        ax.set_ylabel(f"Posterior P(PFI event by {int(H)} mo)")
        ax.set_title(f"Ranked early-failure risk, H={int(H)} months")
        ax.grid(True, alpha=0.18)
        ax.legend(
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        savefig(outdir / f"EarlyFailure_Waterfall_H{int(H)}.png")

        if (
            EARLY_FAILURE_SAVE_POOLED_TOP_PATIENT_PLOTS
            and int(EARLY_FAILURE_TOP_PATIENT_N) > 0
        ):
            top_patient_distribution_rows.extend(
                _plot_top_early_failure_patient_distributions(
                    P=P,
                    p_mean=p_mean,
                    p_q05=p_q05,
                    p_q95=p_q95,
                    valid=valid,
                    y=y,
                    status=status,
                    g_te=g_te,
                    groups=groups,
                    H=H,
                    outdir=outdir,
                    top_n=EARLY_FAILURE_TOP_PATIENT_N,
                )
            )
        if (
            EARLY_FAILURE_SAVE_WITHIN_CANCER_TOP_PATIENT_PLOTS
            and int(EARLY_FAILURE_TOP_PATIENT_WITHIN_CANCER_N) > 0
        ):
            within_cancer_top_patient_distribution_rows.extend(
                _plot_within_cancer_top_early_failure_patient_distributions(
                    P=P,
                    p_mean=p_mean,
                    p_q05=p_q05,
                    p_q95=p_q95,
                    valid=valid,
                    y=y,
                    status=status,
                    g_te=g_te,
                    groups=groups,
                    H=H,
                    outdir=outdir,
                    top_n_per_group=EARLY_FAILURE_TOP_PATIENT_WITHIN_CANCER_N,
                )
            )

        # Risk distribution by observed status.
        df_status = pd.DataFrame(
            {
                "risk": p_mean[valid],
                "label": np.where(y[valid] == 1, "Early failure", "Event-free at H"),
            }
        )
        if df_status["label"].nunique() == 2:
            fig, ax = plt.subplots(figsize=(6.6, 5.0))
            data = [
                df_status.loc[df_status["label"] == lab, "risk"].to_numpy(float)
                for lab in ["Event-free at H", "Early failure"]
            ]
            ax.boxplot(
                data, labels=["Event-free at H", "Early failure"], showfliers=False
            )
            ax.scatter(
                np.ones(len(data[0]))
                + np.random.default_rng(123).normal(0, 0.03, len(data[0])),
                data[0],
                s=10,
                alpha=0.35,
            )
            ax.scatter(
                np.full(len(data[1]), 2.0)
                + np.random.default_rng(456).normal(0, 0.03, len(data[1])),
                data[1],
                s=14,
                alpha=0.45,
            )
            ax.set_ylabel(f"Posterior mean P(PFI event by {int(H)} mo)")
            ax.set_title(f"Predicted risk by fixed-horizon status, H={int(H)} months")
            ax.grid(True, axis="y", alpha=0.18)
            savefig(outdir / f"EarlyFailure_RiskDistribution_H{int(H)}.png")

        # Top-k enrichment with posterior uncertainty propagated through MC samples.
        # Because fixed-horizon early-failure labels are right-censored, precision
        # and recall are IPCW-weighted, matching the AUROC/AUPRC and calibration
        # summaries used elsewhere in the posthoc analysis.
        n_valid = int(np.sum(valid))
        yv_all = np.asarray(y[valid], dtype=float)
        wv_all = np.asarray(w[valid], dtype=float)
        n_pos = int(np.sum(yv_all == 1.0))
        weighted_n_pos = float(np.sum(wv_all * yv_all)) if n_valid > 0 else 0.0
        for frac in [float(f) for f in EARLY_FAILURE_TOPK_FRACS]:
            k = max(1, int(np.ceil(frac * n_valid))) if n_valid > 0 else 0
            recalls = []
            precisions = []
            selected_weight_sums = []
            for sidx in range(P.shape[0]):
                pv = P[sidx, valid]
                ord_s = np.argsort(-pv)
                top = ord_s[:k]
                if k <= 0 or len(top) == 0:
                    recalls.append(np.nan)
                    precisions.append(np.nan)
                    selected_weight_sums.append(0.0)
                    continue
                w_top = wv_all[top]
                y_top = yv_all[top]
                weighted_tp = float(np.sum(w_top * y_top))
                selected_weight = float(np.sum(w_top))
                selected_weight_sums.append(selected_weight)
                recalls.append(
                    weighted_tp / max(weighted_n_pos, 1e-8)
                    if weighted_n_pos > 0.0
                    else np.nan
                )
                precisions.append(
                    weighted_tp / max(selected_weight, 1e-8)
                    if selected_weight > 0.0
                    else np.nan
                )
            recalls = np.asarray(recalls, dtype=float)
            precisions = np.asarray(precisions, dtype=float)
            selected_weight_sums = np.asarray(selected_weight_sums, dtype=float)
            recall_finite = recalls[np.isfinite(recalls)]
            precision_finite = precisions[np.isfinite(precisions)]
            topk_rows.append(
                {
                    "H": float(H),
                    "top_fraction": float(frac),
                    "k": int(k),
                    "n_valid": int(n_valid),
                    "n_early_events": int(n_pos),
                    "weighted_n_early_events": float(weighted_n_pos),
                    "mean_selected_ipcw_weight_sum": (
                        float(np.nanmean(selected_weight_sums))
                        if selected_weight_sums.size
                        else np.nan
                    ),
                    "recall_mean": (
                        float(np.mean(recall_finite)) if recall_finite.size else np.nan
                    ),
                    "recall_q05": (
                        float(np.quantile(recall_finite, 0.05))
                        if recall_finite.size
                        else np.nan
                    ),
                    "recall_q95": (
                        float(np.quantile(recall_finite, 0.95))
                        if recall_finite.size
                        else np.nan
                    ),
                    "precision_mean": (
                        float(np.mean(precision_finite))
                        if precision_finite.size
                        else np.nan
                    ),
                    "precision_q05": (
                        float(np.quantile(precision_finite, 0.05))
                        if precision_finite.size
                        else np.nan
                    ),
                    "precision_q95": (
                        float(np.quantile(precision_finite, 0.95))
                        if precision_finite.size
                        else np.nan
                    ),
                    "posterior_mc_samples": int(P.shape[0]),
                    "estimation_method": "IPCW-weighted fixed-horizon top-k enrichment with posterior risk MC ranking uncertainty",
                }
            )
        # IPCW calibration bins.
        if int(np.sum(valid)) >= max(10, EARLY_FAILURE_CALIBRATION_BINS * 2):
            pv = p_mean[valid]
            yv = y[valid]
            wv = w[valid]
            try:
                bins = pd.qcut(
                    pv,
                    q=min(EARLY_FAILURE_CALIBRATION_BINS, len(np.unique(pv))),
                    duplicates="drop",
                )
                for b in pd.Series(bins).cat.categories:
                    mb = np.asarray(bins == b)
                    if int(np.sum(mb)) == 0:
                        continue
                    obs = float(np.sum(wv[mb] * yv[mb]) / max(np.sum(wv[mb]), 1e-8))
                    pred_s = np.sum(P[:, valid][:, mb] * wv[mb][None, :], axis=1) / max(
                        np.sum(wv[mb]), 1e-8
                    )
                    calib_rows.append(
                        {
                            "scope": "pooled",
                            "group": "pooled",
                            "H": float(H),
                            "bin": str(b),
                            "n": int(np.sum(mb)),
                            "ipcw_weight_sum": float(np.sum(wv[mb])),
                            "weighted_n_early_events": float(np.sum(wv[mb] * yv[mb])),
                            "unweighted_early_event_rate": float(np.mean(yv[mb])),
                            "pred_mean": float(np.mean(pred_s)),
                            "pred_q05": float(np.quantile(pred_s, 0.05)),
                            "pred_q95": float(np.quantile(pred_s, 0.95)),
                            "observed_ipcw_event_rate": obs,
                            "abs_calibration_error": float(
                                abs(obs - float(np.mean(pred_s)))
                            ),
                            "signed_calibration_error_observed_minus_predicted": float(
                                obs - float(np.mean(pred_s))
                            ),
                            "estimation_method": "IPCW fixed-horizon calibration, pooled",
                        }
                    )
            except Exception:
                pass
    pd.DataFrame(pred_rows).to_csv(
        outdir / "early_failure_patient_predictions_with_posterior_intervals.csv",
        index=False,
    )
    if len(top_patient_distribution_rows):
        pd.DataFrame(top_patient_distribution_rows).to_csv(
            outdir / "early_failure_top_patient_posterior_distributions.csv",
            index=False,
        )
    if len(within_cancer_top_patient_distribution_rows):
        pd.DataFrame(within_cancer_top_patient_distribution_rows).to_csv(
            outdir
            / "early_failure_within_cancer_top_patient_posterior_distributions.csv",
            index=False,
        )
    df_topk = pd.DataFrame(topk_rows)
    df_topk.to_csv(outdir / "early_failure_topk_enrichment.csv", index=False)
    if len(df_topk):
        for metric in ["recall", "precision"]:
            fig, ax = plt.subplots(figsize=(7.4, 5.2))
            for H, dfh in df_topk.groupby("H"):
                dfh = dfh.sort_values("top_fraction")
                x = 100.0 * dfh["top_fraction"].to_numpy(float)
                y = dfh[f"{metric}_mean"].to_numpy(float)
                lo = dfh[f"{metric}_q05"].to_numpy(float)
                hi = dfh[f"{metric}_q95"].to_numpy(float)
                ax.plot(x, y, marker="o", lw=2.0, label=f"H={int(H)} mo")
                ax.fill_between(x, lo, hi, alpha=0.16)
            ax.set_xlabel("Top predicted-risk fraction of test patients (%)")
            ax.set_ylabel(f"Early-failure {metric}")
            ax.set_title("Top-k early-failure enrichment")
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, alpha=0.18)
            ax.legend(
                frameon=True,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=[0, 0, 0.82, 1])
            savefig(outdir / f"EarlyFailure_TopK_{metric.capitalize()}_Enrichment.png")
    df_cal = pd.DataFrame(calib_rows)
    df_cal.to_csv(outdir / "early_failure_ipcw_calibration_bins.csv", index=False)
    df_cal_summary = _calibration_summary_from_bins(df_cal)
    if len(df_cal_summary):
        df_cal_summary.to_csv(
            outdir / "early_failure_ipcw_calibration_summary.csv", index=False
        )
    if len(df_cal):
        for H, dfh in df_cal.groupby("H"):
            fig, ax = plt.subplots(figsize=(5.8, 5.4))
            x = dfh["pred_mean"].to_numpy(float)
            y = dfh["observed_ipcw_event_rate"].to_numpy(float)
            lo = dfh["pred_q05"].to_numpy(float)
            hi = dfh["pred_q95"].to_numpy(float)
            ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", capsize=3)
            lim = max(
                0.05,
                float(np.nanmax([x.max() if x.size else 0, y.max() if y.size else 0])),
            )
            ax.plot([0, lim], [0, lim], ls="--", lw=1.0)
            ax.set_xlim(0, min(1.0, lim * 1.15))
            ax.set_ylim(0, min(1.0, lim * 1.15))
            ax.set_xlabel("Mean predicted early-failure probability")
            ax.set_ylabel("Observed IPCW early-failure rate")
            ax.set_title(f"Early-failure calibration (H={int(H)} mo)")
            ax.grid(True, alpha=0.18)
            savefig(outdir / f"EarlyFailure_Calibration_H{int(H)}.png")

    df_within_metrics = pd.DataFrame(within_cancer_metric_rows)
    df_within_topk = pd.DataFrame(within_cancer_topk_rows)
    df_within_cal = pd.DataFrame(within_cancer_calibration_rows)
    if len(df_within_metrics):
        df_prev = df_within_metrics.copy()
        df_prev.to_csv(
            outdir / "early_failure_horizon_prevalence_and_metric_counts.csv",
            index=False,
        )
    df_within_metrics.to_csv(
        outdir / "early_failure_within_cancer_discrimination.csv", index=False
    )
    df_within_topk.to_csv(
        outdir / "early_failure_within_cancer_topk_enrichment.csv", index=False
    )
    df_within_cal.to_csv(
        outdir / "early_failure_within_cancer_ipcw_calibration_bins.csv", index=False
    )
    df_within_cal_summary = _calibration_summary_from_bins(df_within_cal)
    if len(df_within_cal_summary):
        df_within_cal_summary.to_csv(
            outdir / "early_failure_within_cancer_ipcw_calibration_summary.csv",
            index=False,
        )
    _plot_within_cancer_early_failure_summaries(
        df_within_metrics, df_within_topk, df_within_cal, outdir
    )
    df_boot_metrics = pd.DataFrame(bootstrap_metric_rows)
    df_boot_topk = pd.DataFrame(bootstrap_topk_rows)
    df_boot_cal = pd.DataFrame(bootstrap_calibration_rows)
    if len(df_boot_metrics):
        df_boot_metrics.to_csv(
            outdir / "early_failure_bootstrap_discrimination.csv", index=False
        )
    if len(df_boot_topk):
        df_boot_topk.to_csv(
            outdir / "early_failure_bootstrap_topk_enrichment.csv", index=False
        )
    if len(df_boot_cal):
        df_boot_cal.to_csv(
            outdir / "early_failure_bootstrap_ipcw_calibration_bins.csv", index=False
        )
    if EARLY_FAILURE_SAVE_BOOTSTRAP_PLOTS and len(df_boot_topk):
        _plot_bootstrap_topk_summaries(df_boot_topk, outdir)


def _make_calibration_folds(g, events, n_splits, seed):
    g = np.asarray(g, dtype=int)
    events = np.asarray(events, dtype=int)
    n = int(g.size)
    n_splits = int(max(2, min(int(n_splits), n)))
    idx = np.arange(n)
    labels_ge = np.asarray([f"{int(gg)}_{int(ee)}" for gg, ee in zip(g, events)])
    try:
        counts = pd.Series(labels_ge).value_counts()
        if int(counts.min()) >= n_splits:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=int(seed)
            )
            return [
                (tr.astype(int), va.astype(int))
                for tr, va in splitter.split(idx, labels_ge)
            ]
    except Exception:
        pass
    try:
        counts = pd.Series(g).value_counts()
        if int(counts.min()) >= n_splits:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=int(seed)
            )
            return [
                (tr.astype(int), va.astype(int)) for tr, va in splitter.split(idx, g)
            ]
    except Exception:
        pass
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return [(tr.astype(int), va.astype(int)) for tr, va in splitter.split(idx)]


def _fit_calibration_fold_model_and_predict(
    X_train_raw,
    X_valid_raw,
    t_train,
    e_train,
    g_train,
    g_valid,
    groups,
    dt_edges,
    feat_names,
    shared_dim,
    private_dim,
    mt_latent_rank,
    output_rank,
    seed,
    fold_label,
):
    """Train one DGP on a calibration fold and predict held-out fixed-horizon risks."""
    set_all_seeds(int(seed))
    Xtr_t, Xva_t = standardize_from_train(X_train_raw, X_valid_raw)
    gtr_t = torch.tensor(np.asarray(g_train, int), dtype=torch.long, device=device)
    gva_t = torch.tensor(np.asarray(g_valid, int), dtype=torch.long, device=device)
    blocks_sub = infer_blocks_from_feat_names(list(feat_names))
    model = DGP(
        in_dim=Xtr_t.shape[1],
        Xref=Xtr_t,
        num_groups=len(groups),
        shared_dim=int(shared_dim),
        private_dim=int(private_dim),
        mt_shared_rank=int(mt_latent_rank),
        output_rank=int(output_rank),
        blocks=blocks_sub,
        seed=int(seed),
    ).to(device)
    verify_architecture(model)
    # Calibration fold refits must use the same global discrete-time baseline
    # shape as the main DGP training code: alpha_dt has one value per time bin.
    # A group-by-bin alpha creates [N, G, K] broadcasting inside expected_dt_ll_mc
    # and fails when N != G.
    alpha_dt = torch.nn.Parameter(
        torch.zeros(len(dt_edges) - 1, device=device, dtype=torch.float32)
    )
    opt = build_optimizer(model, alpha_dt)
    AR_tr, EV_tr = make_dt_masks(t_train, e_train, dt_edges)
    bin_weights_t = torch.tensor(
        make_dt_bin_weights(dt_edges), dtype=torch.float32, device=device
    )
    best_train_obj = np.inf
    best_state = None
    n_ev = max(1, int(np.sum(np.asarray(e_train, int))))
    for it in range(1, int(POSTHOC_CALIBRATION_EPOCHS) + 1):
        model.train()
        opt.zero_grad()
        ell = expected_dt_ll_mc(
            model,
            Xtr_t,
            gtr_t,
            AR_tr,
            EV_tr,
            alpha_dt,
            S=int(POSTHOC_CALIBRATION_MC_TRAIN),
            bin_weights=bin_weights_t,
        )
        kl_s, kl_p, kl_o = total_kl_details(model)
        kl_total = kl_s + kl_p + kl_o
        kl_beta = min(
            1.0, float(it) / max(1, int(0.30 * int(POSTHOC_CALIBRATION_EPOCHS)))
        )
        elbo = ell - kl_beta * kl_total
        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [alpha_dt], GRAD_CLIP)
        opt.step()
        train_obj = float(loss.detach().cpu())
        if it % PRINT_EVERY == 0 or it == 1:
            print(
                f"[CAL {fold_label} {it:04d}] ELL/evt={float(ell.detach().cpu())/n_ev:.4f} | "
                f"KL/evt={float(kl_total.detach().cpu())/n_ev:.4f} | "
                f"beta={kl_beta:.3f} | ELBO={float(elbo.detach().cpu()):.4f}",
                flush=True,
            )
        if train_obj < best_train_obj:
            best_train_obj = train_obj
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "alpha_dt": alpha_dt.detach().cpu().clone(),
            }
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        alpha_dt.data.copy_(
            best_state["alpha_dt"].to(device=device, dtype=torch.float32)
        )
    model.eval()
    pred = {}
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
        _active_cholesky_jitter()
    ):
        for H in POSTHOC_CALIBRATION_H_LIST:
            pred[float(H)] = _event_probability_mean_at_H(
                model,
                Xva_t,
                gva_t,
                float(H),
                alpha_dt,
                dt_edges,
                S=int(POSTHOC_CALIBRATION_PRED_MC),
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred


def _build_or_load_oof_calibration_predictions(art, outdir):
    """Build/load OOF native risks for posthoc calibration.

    Robust calibration behavior for final NPJ runs:
    - first tries the requested K-fold cross-fit partition;
    - if a fold-level refit fails, retries that fold with alternate random seeds;
    - if any fold remains failed, retries a fresh fold partition;
    - if full coverage is still unavailable, writes the best partial OOF table and
      continues using available OOF predictions rather than stopping the whole seed.

    This keeps calibration failures from killing the full performance/FC run while
    making every fallback explicit in posthoc_crossfit_fold_assignments.csv.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / "posthoc_crossfit_oof_native_risks.csv"
    fold_path = outdir / "posthoc_crossfit_fold_assignments.csv"
    if pred_path.exists() and (not POSTHOC_CALIBRATION_FORCE_REFIT):
        return pd.read_csv(pred_path)

    X = np.asarray(art["Xtr_np_sub"], dtype=float)
    t = np.asarray(art["t_tr"], dtype=float)
    e = np.asarray(art["e_tr"], dtype=int)
    g = np.asarray(art["g_tr"], dtype=int)
    n = int(X.shape[0])

    requested_k = int(max(2, min(int(POSTHOC_CALIBRATION_K_FOLDS), n)))
    candidate_ks = []
    for kk in [requested_k, requested_k - 1, 2]:
        kk = int(max(2, min(int(kk), n)))
        if kk not in candidate_ks:
            candidate_ks.append(kk)

    n_partition_attempts = int(max(1, int(POSTHOC_CALIBRATION_FOLD_PARTITION_RETRIES)))
    n_fold_retries = int(max(1, int(POSTHOC_CALIBRATION_FOLD_RETRY_SEEDS)))
    min_oof_fraction = float(max(0.0, min(1.0, float(POSTHOC_CALIBRATION_MIN_OOF_FRACTION))))

    best_pred_by_H = None
    best_fold_rows = None
    best_coverage = -1
    best_tag = "none"

    for k_idx, k in enumerate(candidate_ks):
        for partition_attempt in range(1, n_partition_attempts + 1):
            partition_seed = int(SEED + 913 + 10007 * partition_attempt + 1000003 * k_idx)
            folds = _make_calibration_folds(g, e, int(k), seed=partition_seed)
            pred_by_H = {
                float(H): np.full(n, np.nan, dtype=float)
                for H in POSTHOC_CALIBRATION_H_LIST
            }
            fold_rows = []
            print(
                f"[CAL] Trying OOF calibration partition: k={k}, attempt={partition_attempt}/{n_partition_attempts}, seed={partition_seed}",
                flush=True,
            )

            for fold_id, (fit_idx, hold_idx) in enumerate(folds, start=1):
                row = {
                    "partition_k": int(k),
                    "partition_attempt": int(partition_attempt),
                    "partition_seed": int(partition_seed),
                    "fold": int(fold_id),
                    "n_fit": int(len(fit_idx)),
                    "n_holdout": int(len(hold_idx)),
                    "events_fit": int(np.sum(e[fit_idx])),
                    "events_holdout": int(np.sum(e[hold_idx])),
                    "status": "not_run",
                    "retry_id": np.nan,
                    "fit_seed": np.nan,
                    "error": "",
                }
                print(
                    f"[CAL] Fold {fold_id}/{len(folds)}: fit n={len(fit_idx)}, holdout n={len(hold_idx)}",
                    flush=True,
                )

                fold_pred = None
                last_error = ""
                for retry_id in range(1, n_fold_retries + 1):
                    fit_seed = int(SEED + 1009 * fold_id + 104729 * retry_id + 10000019 * partition_attempt + 100000007 * k_idx)
                    try:
                        fold_pred = _fit_calibration_fold_model_and_predict(
                            X_train_raw=X[fit_idx],
                            X_valid_raw=X[hold_idx],
                            t_train=t[fit_idx],
                            e_train=e[fit_idx],
                            g_train=g[fit_idx],
                            g_valid=g[hold_idx],
                            groups=art["groups"],
                            dt_edges=art["dt_edges"],
                            feat_names=art["feat_names"],
                            shared_dim=art["shared_dim"],
                            private_dim=art["private_dim"],
                            mt_latent_rank=art["mt_latent_rank"],
                            output_rank=art["output_rank"],
                            seed=fit_seed,
                            fold_label=f"k{k}_attempt{partition_attempt}_fold{fold_id}_retry{retry_id}",
                        )
                        row["status"] = "ok" if retry_id == 1 else "ok_after_retry"
                        row["retry_id"] = int(retry_id)
                        row["fit_seed"] = int(fit_seed)
                        row["error"] = ""
                        break
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        print(
                            f"[CAL][WARN] Fold {fold_id} failed on retry {retry_id}/{n_fold_retries}: {last_error}",
                            flush=True,
                        )
                        fold_pred = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if fold_pred is None:
                    row["status"] = "failed_after_retries_left_as_nan"
                    row["error"] = last_error
                    print(
                        f"[CAL][WARN] Fold {fold_id} failed after {n_fold_retries} retries; this partition will be treated as incomplete.",
                        flush=True,
                    )
                else:
                    for H, pv in fold_pred.items():
                        pred_by_H[float(H)][hold_idx] = np.clip(np.asarray(pv, float), 0.0, 1.0)
                fold_rows.append(row)

            # Coverage = rows with finite OOF predictions for all requested horizons.
            if len(POSTHOC_CALIBRATION_H_LIST):
                finite_mat = np.vstack([np.isfinite(pred_by_H[float(H)]) for H in POSTHOC_CALIBRATION_H_LIST]).T
                coverage = int(np.sum(np.all(finite_mat, axis=1)))
            else:
                coverage = 0
            tag = f"k{k}_attempt{partition_attempt}"
            print(
                f"[CAL] OOF calibration partition {tag}: complete rows={coverage}/{n}",
                flush=True,
            )
            if coverage > best_coverage:
                best_coverage = coverage
                best_pred_by_H = pred_by_H
                best_fold_rows = fold_rows
                best_tag = tag
            if coverage == n:
                print(f"[CAL] Using complete OOF calibration partition {tag}.", flush=True)
                break
        if best_coverage == n:
            break

    if best_pred_by_H is None:
        # Absolute fallback: preserve pipeline completion with all NaNs. Downstream
        # calibration routines will mark models as skipped when insufficient OOF
        # data are available.
        best_pred_by_H = {float(H): np.full(n, np.nan, dtype=float) for H in POSTHOC_CALIBRATION_H_LIST}
        best_fold_rows = [{"status": "all_calibration_partitions_failed", "error": "no OOF predictions produced"}]
        best_coverage = 0

    if best_coverage < n:
        msg = (
            f"[CAL][WARN] No complete OOF calibration partition was available. "
            f"Using best partition {best_tag} with complete rows={best_coverage}/{n}. "
            f"Downstream calibrators will use available OOF predictions only."
        )
        print(msg, flush=True)
        if (best_coverage / max(1, n)) < min_oof_fraction and not bool(POSTHOC_CALIBRATION_ALLOW_PARTIAL_OOF):
            raise RuntimeError(
                f"OOF calibration coverage {best_coverage}/{n} is below required fraction {min_oof_fraction:.3f}."
            )

    pd.DataFrame(best_fold_rows).to_csv(fold_path, index=False)
    df = pd.DataFrame(
        {
            "train_index": np.arange(n, dtype=int),
            "time": t,
            "event": e,
            "group_idx": g,
            "group": [str(art["groups"][int(gg)]) for gg in g],
            "oof_calibration_partition_used": str(best_tag),
            "oof_calibration_complete_rows": int(best_coverage),
            "oof_calibration_total_rows": int(n),
        }
    )
    for H, arr in best_pred_by_H.items():
        df[f"p_native_oof_H{int(H)}"] = arr
    df.to_csv(pred_path, index=False)
    return df



def _inverse_softplus_scalar(y):
    """Stable scalar inverse of softplus, used to initialize positive slope parameters.

    softplus(x) = log(1 + exp(x)); this returns x such that softplus(x) ~= y.
    """
    y = float(y)
    if not np.isfinite(y) or y <= 0.0:
        raise ValueError(f"softplus inverse requires a positive finite scalar, got {y!r}")
    if y > 20.0:
        return y
    return float(np.log(np.expm1(y)))

def _safe_logit(p, eps=POSTHOC_CALIBRATION_EPS):
    p = np.clip(np.asarray(p, dtype=float), float(eps), 1.0 - float(eps))
    return np.log(p / (1.0 - p))


def _inv_logit_np(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _fit_ipcw_logistic_recalibrator(p, times, events, groups_idx, H, kind="pooled"):
    """Fit an IPCW-weighted logit recalibrator on fixed-horizon risks.

    requested kind="intercept_only":
        logit(p_cal) = alpha + logit(p)

    requested kind="pooled":
        logit(p_cal) = alpha + beta * logit(p)

    requested kind="group_intercept":
        logit(p_cal) = alpha + beta * logit(p) + centered group intercept

    Safety behavior:
      - If there are too few events/non-events for any calibration model, the
        identity transform is returned and the row is marked skipped.
      - If a slope-based model is requested but event counts are too small for a
        slope, the effective fitted model falls back to intercept_only.
      - If group_intercept is requested but group-level event/non-event counts
        are too small, the effective fitted model falls back to pooled if the
        slope gate passes, otherwise to intercept_only.
      - For slope-based models, beta is constrained positive via softplus so the
        calibrated risk transform cannot silently reverse the risk ranking.
    """
    p = np.asarray(p, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    groups_idx = np.asarray(groups_idx, dtype=int)
    H = float(H)

    valid, y_all, w_all = _ipcw_binary_event_targets(times, events, H)
    keep = (
        np.asarray(valid, bool)
        & np.isfinite(p)
        & np.isfinite(w_all)
        & (np.asarray(w_all, float) > 0.0)
    )
    p_fit = np.clip(p[keep], POSTHOC_CALIBRATION_EPS, 1.0 - POSTHOC_CALIBRATION_EPS)
    y = np.asarray(y_all, int)[keep]
    w = np.asarray(w_all, float)[keep]
    g = groups_idx[keep] if groups_idx.size else np.zeros(y.shape[0], dtype=int)
    n_groups = int(np.max(groups_idx)) + 1 if groups_idx.size else 1

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_early_censored = int(
        np.sum(np.isfinite(times) & np.isfinite(events) & (events == 0) & (times <= H))
    )

    group_event_counts = [0] * int(max(n_groups, 1))
    group_nonevent_counts = [0] * int(max(n_groups, 1))
    group_fit_counts = [0] * int(max(n_groups, 1))
    for gg in range(int(max(n_groups, 1))):
        mg = g == int(gg)
        group_fit_counts[gg] = int(np.sum(mg))
        group_event_counts[gg] = int(np.sum(y[mg] == 1))
        group_nonevent_counts[gg] = int(np.sum(y[mg] == 0))

    base = {
        "requested_kind": str(kind),
        "kind": str(kind),
        "effective_kind": str(kind),
        "H": H,
        "n_original": int(p.size),
        "n_fit": int(y.size),
        "n_pos_fit": n_pos,
        "n_neg_fit": n_neg,
        "n_early_censored_before_H": n_early_censored,
        "ipcw_weight_sum_fit": float(np.sum(w)) if w.size else 0.0,
        "weighted_event_rate_fit": (
            float(np.sum(w * y) / max(np.sum(w), 1e-8)) if w.size else np.nan
        ),
        "group_fit_counts": group_fit_counts,
        "group_event_counts": group_event_counts,
        "group_nonevent_counts": group_nonevent_counts,
        "group_fit_counts_json": json.dumps(group_fit_counts),
        "group_event_counts_json": json.dumps(group_event_counts),
        "group_nonevent_counts_json": json.dumps(group_nonevent_counts),
        "min_events_for_slope": int(POSTHOC_CALIBRATION_MIN_EVENTS_FOR_SLOPE),
        "min_nonevents_for_slope": int(POSTHOC_CALIBRATION_MIN_NONEVENTS_FOR_SLOPE),
        "min_events_per_group": int(POSTHOC_CALIBRATION_MIN_EVENTS_PER_GROUP),
        "min_nonevents_per_group": int(POSTHOC_CALIBRATION_MIN_NONEVENTS_PER_GROUP),
        "slope_constrained_nonnegative": bool(
            POSTHOC_CALIBRATION_CONSTRAIN_SLOPE_NONNEGATIVE
        ),
        "used_group_intercepts": False,
        "skipped": False,
        "fallback": "none",
        "alpha": np.nan,
        "beta": np.nan,
        "beta_raw": np.nan,
        "group_offsets": [0.0] * int(max(n_groups, 1)),
    }

    if (
        y.size < int(POSTHOC_CALIBRATION_MIN_N_FOR_ANY_MODEL)
        or n_pos < int(POSTHOC_CALIBRATION_MIN_EVENTS_FOR_ANY_MODEL)
        or n_neg < int(POSTHOC_CALIBRATION_MIN_NONEVENTS_FOR_ANY_MODEL)
        or float(np.sum(w)) <= 0.0
    ):
        base["skipped"] = True
        base["fallback"] = "insufficient_oof_events_for_any_calibration_identity_used"
        base["effective_kind"] = "identity"
        base["alpha"] = 0.0
        base["beta"] = 1.0
        return base

    kind_lower = str(kind).lower()
    if kind_lower in {"intercept_only", "calibration_in_the_large", "citl"}:
        effective_kind = "intercept_only"
    elif kind_lower in {"pooled", "group_intercept"}:
        slope_ok = n_pos >= int(
            POSTHOC_CALIBRATION_MIN_EVENTS_FOR_SLOPE
        ) and n_neg >= int(POSTHOC_CALIBRATION_MIN_NONEVENTS_FOR_SLOPE)
        if not slope_ok:
            effective_kind = "intercept_only"
            base["fallback"] = (
                "slope_model_insufficient_events_fell_back_to_intercept_only"
            )
        elif kind_lower == "group_intercept":
            represented = [
                gg for gg, cnt in enumerate(group_fit_counts) if int(cnt) > 0
            ]
            group_ok = (
                len(represented) >= 2
                and all(
                    group_event_counts[gg]
                    >= int(POSTHOC_CALIBRATION_MIN_EVENTS_PER_GROUP)
                    for gg in represented
                )
                and all(
                    group_nonevent_counts[gg]
                    >= int(POSTHOC_CALIBRATION_MIN_NONEVENTS_PER_GROUP)
                    for gg in represented
                )
            )
            if group_ok:
                effective_kind = "group_intercept"
            else:
                effective_kind = "pooled"
                base["fallback"] = (
                    "group_intercept_insufficient_group_events_fell_back_to_pooled"
                )
        else:
            effective_kind = "pooled"
    else:
        effective_kind = "intercept_only"
        base["fallback"] = (
            f"unknown_requested_kind_{kind_lower}_fell_back_to_intercept_only"
        )

    base["effective_kind"] = str(effective_kind)

    z = _safe_logit(p_fit)
    try:
        z_t = torch.tensor(z, dtype=torch.float64)
        y_t = torch.tensor(y.astype(float), dtype=torch.float64)
        w_t = torch.tensor(w / max(float(np.mean(w)), 1e-8), dtype=torch.float64)
        g_t = torch.tensor(g, dtype=torch.long)

        intercept_only = effective_kind == "intercept_only"
        use_group = effective_kind == "group_intercept"

        alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        params = [alpha]

        if intercept_only:
            beta_raw = None
        else:
            beta_raw = torch.nn.Parameter(
                torch.tensor(_inverse_softplus_scalar(1.0), dtype=torch.float64)
            )
            params.append(beta_raw)

        if use_group:
            group_offsets_raw = torch.nn.Parameter(
                torch.zeros(n_groups, dtype=torch.float64)
            )
            params.append(group_offsets_raw)
        else:
            group_offsets_raw = None

        opt = torch.optim.LBFGS(
            params, lr=0.25, max_iter=250, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad()
            if intercept_only:
                beta_val = torch.tensor(1.0, dtype=torch.float64)
                logits = alpha + z_t
                penalty = alpha.pow(2) * 0.0
            else:
                if bool(POSTHOC_CALIBRATION_CONSTRAIN_SLOPE_NONNEGATIVE):
                    beta_val = F.softplus(beta_raw) + 1e-8
                else:
                    beta_val = beta_raw
                logits = alpha + beta_val * z_t
                penalty = alpha.pow(2) * 0.0 + (beta_val - 1.0).pow(2) * float(
                    POSTHOC_CALIBRATION_L2
                )
            if group_offsets_raw is not None:
                offsets = group_offsets_raw - torch.mean(group_offsets_raw)
                logits = logits + offsets.index_select(0, g_t)
                penalty = penalty + float(POSTHOC_CALIBRATION_L2) * torch.mean(
                    offsets.pow(2)
                )
            bce = F.binary_cross_entropy_with_logits(
                logits, y_t, weight=w_t, reduction="sum"
            ) / torch.sum(w_t)
            loss = bce + penalty
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            offsets_np = np.zeros(n_groups, dtype=float)
            if group_offsets_raw is not None:
                off = group_offsets_raw - torch.mean(group_offsets_raw)
                offsets_np = off.detach().cpu().numpy().astype(float)

            if intercept_only:
                beta_val = torch.tensor(1.0, dtype=torch.float64)
                beta_raw_val = np.nan
            else:
                beta_raw_val = float(beta_raw.detach().cpu())
                if bool(POSTHOC_CALIBRATION_CONSTRAIN_SLOPE_NONNEGATIVE):
                    beta_val = F.softplus(beta_raw.detach().cpu()) + 1e-8
                else:
                    beta_val = beta_raw.detach().cpu()

            base["alpha"] = float(alpha.detach().cpu())
            base["beta"] = float(beta_val)
            base["beta_raw"] = beta_raw_val
            base["group_offsets"] = offsets_np.tolist()
            base["used_group_intercepts"] = bool(use_group)

            if not (np.isfinite(base["alpha"]) and np.isfinite(base["beta"])):
                raise FloatingPointError("non-finite calibration parameters")
            if (
                (not intercept_only)
                and bool(POSTHOC_CALIBRATION_CONSTRAIN_SLOPE_NONNEGATIVE)
                and base["beta"] < 0.0
            ):
                raise FloatingPointError(
                    "negative calibration slope despite positive constraint"
                )
    except Exception as exc:
        base["skipped"] = True
        base["fallback"] = (
            f"optimizer_failed_identity_used: {type(exc).__name__}: {exc}"
        )
        base["effective_kind"] = "identity"
        base["alpha"] = 0.0
        base["beta"] = 1.0
        base["beta_raw"] = np.nan
        base["group_offsets"] = [0.0] * int(max(n_groups, 1))
        base["used_group_intercepts"] = False
    return base


def _apply_recalibrator(p, groups_idx, fit):
    p = np.clip(
        np.asarray(p, dtype=float),
        POSTHOC_CALIBRATION_EPS,
        1.0 - POSTHOC_CALIBRATION_EPS,
    )
    alpha_val = float(fit.get("alpha", 0.0))
    beta_val = float(fit.get("beta", 1.0))
    if not (np.isfinite(alpha_val) and np.isfinite(beta_val)):
        alpha_val = 0.0
        beta_val = 1.0
    z = _safe_logit(p)
    eta = alpha_val + beta_val * z
    offsets = np.asarray(fit.get("group_offsets", []), dtype=float)
    if offsets.size:
        g = np.asarray(groups_idx, dtype=int).reshape(-1)
        add = np.zeros(g.shape[0], dtype=float)
        ok = (g >= 0) & (g < offsets.size)
        add[ok] = offsets[g[ok]]
        eta = np.asarray(eta, dtype=float)
        if eta.ndim == 1:
            eta = eta + add
        elif eta.ndim == 2 and eta.shape[1] == add.shape[0]:
            eta = eta + add[None, :]
        else:
            eta = eta + np.reshape(add, add.shape + (1,) * max(0, eta.ndim - 1))
    return np.clip(
        _inv_logit_np(eta), POSTHOC_CALIBRATION_EPS, 1.0 - POSTHOC_CALIBRATION_EPS
    )


def _apply_recalibrator_to_samples(P, groups_idx, fit):
    """Apply a fitted horizon-specific recalibrator to posterior risk draws [S,N]."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(
            f"Expected posterior risk samples with shape [S,N], got {P.shape}"
        )
    return _apply_recalibrator(P, groups_idx, fit)


def _append_recalibration_bins(
    rows,
    p,
    times,
    events,
    groups_idx,
    group_names,
    H,
    split,
    model_kind,
    risk_version,
    P_samples=None,
):
    p = np.asarray(p, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    groups_idx = np.asarray(groups_idx, dtype=int)
    valid, y, w = _ipcw_binary_event_targets(times, events, float(H))
    P_for_bins = p[None, :] if P_samples is None else np.asarray(P_samples, dtype=float)
    _append_calibration_rows_for_scope(
        rows,
        P_for_bins,
        p,
        valid,
        y,
        w,
        float(H),
        f"{split}_{risk_version}_{model_kind}_pooled",
        "ALL",
    )
    for gi, gname in enumerate(group_names):
        m = groups_idx == gi
        if int(np.sum(m)) == 0:
            continue
        _append_calibration_rows_for_scope(
            rows,
            P_for_bins[:, m],
            p[m],
            np.asarray(valid, bool)[m],
            np.asarray(y, int)[m],
            np.asarray(w, float)[m],
            float(H),
            f"{split}_{risk_version}_{model_kind}_within_cancer",
            str(gname),
        )


def _summarize_recalibrated_metrics(
    p, times, events, groups_idx, group_names, H, split, model_kind, risk_version
):
    rows = []
    p = np.asarray(p, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    groups_idx = np.asarray(groups_idx, dtype=int)
    valid, y, w = _ipcw_binary_event_targets(times, events, float(H))
    for scope, gi, gname, mask in [
        ("pooled", None, "ALL", np.ones_like(p, dtype=bool))
    ] + [
        ("within_cancer", int(gi), str(gname), groups_idx == int(gi))
        for gi, gname in enumerate(group_names)
    ]:
        keep = (
            mask
            & np.asarray(valid, bool)
            & np.isfinite(p)
            & np.isfinite(w)
            & (np.asarray(w) > 0.0)
        )
        core = _fixed_horizon_metric_core_from_arrays(
            p[keep], np.asarray(y)[keep], np.asarray(w)[keep]
        )
        pred_mean = _weighted_mean_safe(p[keep], np.asarray(w)[keep])
        rows.append(
            {
                "split": str(split),
                "model_kind": str(model_kind),
                "risk_version": str(risk_version),
                "scope": str(scope),
                "group_idx": gi if gi is not None else np.nan,
                "group": str(gname),
                "H": float(H),
                "ipcw_mean_predicted_event_probability": pred_mean,
                "ipcw_observed_event_rate": core.get(
                    "baseline_ipcw_event_rate", np.nan
                ),
                "calibration_in_the_large_observed_minus_predicted": (
                    float(core.get("baseline_ipcw_event_rate", np.nan) - pred_mean)
                    if np.isfinite(core.get("baseline_ipcw_event_rate", np.nan))
                    and np.isfinite(pred_mean)
                    else np.nan
                ),
                **core,
            }
        )
    return rows


def _plot_native_vs_calibrated_bins(df_bins, outdir):
    if not isinstance(df_bins, pd.DataFrame) or len(df_bins) == 0:
        return
    outdir = Path(outdir)
    df = df_bins.copy()
    needed = {"scope", "H", "pred_mean", "observed_ipcw_event_rate"}
    if not needed.issubset(df.columns):
        return
    for H, dfH in df.groupby("H"):
        # pooled native vs calibrated for each model kind
        for kind in sorted(
            set(
                str(x)
                for x in dfH.get("model_kind", pd.Series([], dtype=str))
                .dropna()
                .unique()
            )
        ):
            fig, ax = plt.subplots(figsize=(6.2, 5.6))
            for rv, marker in [("native", "o"), ("calibrated", "s")]:
                sub = dfH[
                    (dfH.get("scope_type", "") == "pooled")
                    & (dfH.get("model_kind", "") == str(kind))
                    & (dfH.get("risk_version", "") == str(rv))
                    & np.isfinite(pd.to_numeric(dfH["pred_mean"], errors="coerce"))
                    & np.isfinite(
                        pd.to_numeric(dfH["observed_ipcw_event_rate"], errors="coerce")
                    )
                ]
                if len(sub) == 0:
                    continue
                sub = sub.sort_values("pred_mean")
                ax.plot(
                    sub["pred_mean"].to_numpy(float),
                    sub["observed_ipcw_event_rate"].to_numpy(float),
                    marker=marker,
                    lw=1.8,
                    label=rv.capitalize(),
                )
            ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="gray")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted event probability")
            ax.set_ylabel("Observed IPCW event rate")
            ax.set_title(f"Post-hoc recalibration, {kind}, H={int(float(H))} months")
            ax.grid(True, alpha=0.18)
            ax.legend(loc="best", frameon=True)
            savefig(
                outdir
                / f"PosthocRecalibration_native_vs_calibrated_{kind}_H{int(float(H))}.png"
            )


def _posthoc_risk_label(risk_version, model_kind):
    if str(risk_version) == "native":
        return "Native DGP"
    return f"Calibrated DGP ({model_kind})"


def _risk_sample_summary(P):
    P = np.clip(np.asarray(P, dtype=float), 0.0, 1.0)
    return {
        "mean": np.mean(P, axis=0),
        "q05": np.quantile(P, 0.05, axis=0),
        "q25": np.quantile(P, 0.25, axis=0),
        "q50": np.quantile(P, 0.50, axis=0),
        "q75": np.quantile(P, 0.75, axis=0),
        "q95": np.quantile(P, 0.95, axis=0),
    }


def _fixed_horizon_status_from_ipcw(t, e, valid, y):
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=int)
    valid = np.asarray(valid, dtype=bool)
    y = np.asarray(y, dtype=int)
    status = np.full(t.shape[0], "censored_before_H", dtype=object)
    status[valid & (y == 1)] = "early_failure"
    status[valid & (y == 0)] = "event_free_at_H"
    return status


def _append_posthoc_patient_prediction_rows(
    rows, P, H, risk_version, model_kind, valid, y, w, status, t_te, e_te, g_te, groups
):
    P = np.asarray(P, dtype=float)
    summ = _risk_sample_summary(P)
    valid = np.asarray(valid, dtype=bool)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)
    status = np.asarray(status, dtype=object)
    t_te = np.asarray(t_te, dtype=float)
    e_te = np.asarray(e_te, dtype=int)
    g_te = np.asarray(g_te, dtype=int)
    groups = list(groups)
    for i in range(P.shape[1]):
        rows.append(
            {
                "H": float(H),
                "risk_version": str(risk_version),
                "model_kind": str(model_kind),
                "patient_index_test": int(i),
                "group_idx": int(g_te[i]),
                "group": (
                    str(groups[int(g_te[i])])
                    if 0 <= int(g_te[i]) < len(groups)
                    else str(g_te[i])
                ),
                "time_months": float(t_te[i]) if np.isfinite(t_te[i]) else np.nan,
                "event": int(e_te[i]) if np.isfinite(e_te[i]) else np.nan,
                "label_status": str(status[i]),
                "early_failure_label": int(y[i]) if valid[i] else np.nan,
                "ipcw_weight": float(w[i]) if valid[i] else np.nan,
                "p_failure_mean": float(summ["mean"][i]),
                "p_failure_q05": float(summ["q05"][i]),
                "p_failure_q25": float(summ["q25"][i]),
                "p_failure_q50": float(summ["q50"][i]),
                "p_failure_q75": float(summ["q75"][i]),
                "p_failure_q95": float(summ["q95"][i]),
                "posterior_mc_samples": int(P.shape[0]),
                "interval_note": "native intervals are posterior DGP risk draws; calibrated intervals transform each posterior draw through the fixed horizon-specific recalibrator",
            }
        )


def _plot_posthoc_waterfall(P, H, risk_version, model_kind, valid, y, outdir):
    outdir = Path(outdir)
    summ = _risk_sample_summary(P)
    p_mean = summ["mean"]
    p_q05 = summ["q05"]
    p_q95 = summ["q95"]
    valid = np.asarray(valid, dtype=bool)
    y = np.asarray(y, dtype=int)
    order = np.argsort(p_mean)
    x = np.arange(len(order))
    label = _posthoc_risk_label(risk_version, model_kind)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.fill_between(
        x,
        p_q05[order],
        p_q95[order],
        alpha=0.18,
        label="posterior 90% interval",
    )
    ax.plot(x, p_mean[order], lw=1.8, label="posterior mean")
    ev_ord = valid[order] & (y[order] == 1)
    non_ord = valid[order] & (y[order] == 0)
    ax.scatter(
        x[non_ord], p_mean[order][non_ord], s=12, alpha=0.65, label="event-free at H"
    )
    ax.scatter(
        x[ev_ord], p_mean[order][ev_ord], s=28, marker="x", label="early failure"
    )
    ax.set_xlabel("Test patients sorted by predicted early-failure probability")
    ax.set_ylabel(f"P(PFI event by {int(float(H))} mo)")
    ax.set_title(f"Ranked early-failure risk — {label}, H={int(float(H))} months")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.18)
    ax.legend(
        frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0
    )
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    savefig(
        outdir
        / f"PosthocRecalibration_{risk_version}_{model_kind}_Waterfall_H{int(float(H))}.png"
    )


def _plot_posthoc_risk_distribution(P, H, risk_version, model_kind, valid, y, outdir):
    outdir = Path(outdir)
    p_mean = _risk_sample_summary(P)["mean"]
    valid = np.asarray(valid, dtype=bool)
    y = np.asarray(y, dtype=int)
    df_status = pd.DataFrame(
        {
            "risk": p_mean[valid],
            "label": np.where(y[valid] == 1, "Early failure", "Event-free at H"),
        }
    )
    if len(df_status) == 0 or df_status["label"].nunique() != 2:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    data = [
        df_status.loc[df_status["label"] == lab, "risk"].to_numpy(float)
        for lab in ["Event-free at H", "Early failure"]
    ]
    ax.boxplot(data, labels=["Event-free at H", "Early failure"], showfliers=False)
    ax.scatter(
        np.ones(len(data[0]))
        + np.random.default_rng(123).normal(0, 0.03, len(data[0])),
        data[0],
        s=10,
        alpha=0.35,
    )
    ax.scatter(
        np.full(len(data[1]), 2.0)
        + np.random.default_rng(456).normal(0, 0.03, len(data[1])),
        data[1],
        s=10,
        alpha=0.35,
    )
    ax.set_ylabel(f"P(PFI event by {int(float(H))} mo)")
    ax.set_title(
        f"Early-failure risk by observed status — {_posthoc_risk_label(risk_version, model_kind)}, H={int(float(H))} months"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", alpha=0.18)
    savefig(
        outdir
        / f"PosthocRecalibration_{risk_version}_{model_kind}_RiskDistribution_H{int(float(H))}.png"
    )


def _append_posthoc_topk_rows(
    rows, P, valid, y, w, g_te, groups, H, risk_version, model_kind
):
    start = len(rows)
    _append_topk_rows_for_scope(
        rows, P, valid, y, w, H, scope="pooled", group_name="pooled"
    )
    for gi, gname in enumerate(groups):
        mg = np.asarray(valid, dtype=bool) & (np.asarray(g_te, dtype=int) == int(gi))
        _append_topk_rows_for_scope(
            rows, P, mg, y, w, H, scope="within_cancer", group_name=str(gname)
        )
    for r in rows[start:]:
        r["risk_version"] = str(risk_version)
        r["model_kind"] = str(model_kind)


def _append_posthoc_calibration_rows(
    rows, P, valid, y, w, g_te, groups, H, risk_version, model_kind
):
    start = len(rows)
    summ = _risk_sample_summary(P)
    p_mean = summ["mean"]
    _append_calibration_rows_for_scope(
        rows, P, p_mean, valid, y, w, H, scope="pooled", group_name="pooled"
    )
    for gi, gname in enumerate(groups):
        mg = np.asarray(valid, dtype=bool) & (np.asarray(g_te, dtype=int) == int(gi))
        _append_calibration_rows_for_scope(
            rows, P, p_mean, mg, y, w, H, scope="within_cancer", group_name=str(gname)
        )
    for r in rows[start:]:
        r["risk_version"] = str(risk_version)
        r["model_kind"] = str(model_kind)


def _plot_posthoc_topk_enrichment(df_topk, outdir):
    if not isinstance(df_topk, pd.DataFrame) or len(df_topk) == 0:
        return
    outdir = Path(outdir)
    for (risk_version, model_kind, scope), dfs in df_topk.groupby(
        ["risk_version", "model_kind", "scope"], dropna=False
    ):
        for metric in ["recall", "precision"]:
            fig, ax = plt.subplots(figsize=(7.4, 5.2))
            for H, dfh in dfs.groupby("H"):
                dfh = dfh.sort_values("top_fraction")
                x = 100.0 * dfh["top_fraction"].to_numpy(float)
                yv = dfh[f"{metric}_mean"].to_numpy(float)
                lo = dfh[f"{metric}_q05"].to_numpy(float)
                hi = dfh[f"{metric}_q95"].to_numpy(float)
                ax.plot(x, yv, marker="o", lw=2.0, label=f"H={int(float(H))} mo")
                ax.fill_between(x, lo, hi, alpha=0.16)
            ax.set_xlabel("Top predicted-risk fraction of test patients (%)")
            ax.set_ylabel(f"Early-failure {metric}")
            ax.set_title(
                f"Top-k early-failure enrichment — {_posthoc_risk_label(risk_version, model_kind)} — {scope}"
            )
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, alpha=0.18)
            ax.legend(
                frameon=True,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=[0, 0, 0.82, 1])
            savefig(
                outdir
                / f"PosthocRecalibration_{risk_version}_{model_kind}_{_safe_fname(scope)}_TopK_{metric.capitalize()}_Enrichment.png"
            )


def _plot_posthoc_calibration_by_version(df_cal, outdir):
    if not isinstance(df_cal, pd.DataFrame) or len(df_cal) == 0:
        return
    outdir = Path(outdir)
    needed = {
        "risk_version",
        "model_kind",
        "scope",
        "H",
        "pred_mean",
        "pred_q05",
        "pred_q95",
        "observed_ipcw_event_rate",
    }
    if not needed.issubset(df_cal.columns):
        return
    for (risk_version, model_kind, scope, H), dfh in df_cal.groupby(
        ["risk_version", "model_kind", "scope", "H"], dropna=False
    ):
        # The within-cancer rows are multi-group; the compact calibration figure is most interpretable pooled.
        if str(scope) != "pooled":
            continue
        dfh = dfh.sort_values("pred_mean")
        x = dfh["pred_mean"].to_numpy(float)
        yv = dfh["observed_ipcw_event_rate"].to_numpy(float)
        lo = dfh["pred_q05"].to_numpy(float)
        hi = dfh["pred_q95"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(yv)
        if int(np.sum(ok)) == 0:
            continue
        x, yv, lo, hi = x[ok], yv[ok], lo[ok], hi[ok]
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        ax.errorbar(x, yv, xerr=np.vstack([x - lo, hi - x]), fmt="o", capsize=3)
        lim = max(0.05, float(np.nanmax([np.nanmax(x), np.nanmax(yv)])))
        ax.plot([0, lim], [0, lim], ls="--", lw=1.0)
        ax.set_xlim(0, min(1.0, lim * 1.15))
        ax.set_ylim(0, min(1.0, lim * 1.15))
        ax.set_xlabel("Mean predicted early-failure probability")
        ax.set_ylabel("Observed IPCW early-failure rate")
        ax.set_title(
            f"Early-failure calibration — {_posthoc_risk_label(risk_version, model_kind)}, H={int(float(H))} mo"
        )
        ax.grid(True, alpha=0.18)
        savefig(
            outdir
            / f"PosthocRecalibration_{risk_version}_{model_kind}_Calibration_H{int(float(H))}.png"
        )


def _save_posthoc_early_failure_outputs_for_version(
    P,
    H,
    risk_version,
    model_kind,
    valid,
    y,
    w,
    status,
    t_te,
    e_te,
    g_te,
    groups,
    outdir,
    patient_rows,
    topk_rows,
    calibration_rows,
    top_patient_rows,
    within_cancer_top_patient_rows,
):
    P = np.clip(
        np.asarray(P, dtype=float),
        POSTHOC_CALIBRATION_EPS,
        1.0 - POSTHOC_CALIBRATION_EPS,
    )
    H = float(H)
    outdir = Path(outdir)
    label_kind = str(model_kind)
    _append_posthoc_patient_prediction_rows(
        patient_rows,
        P,
        H,
        risk_version,
        label_kind,
        valid,
        y,
        w,
        status,
        t_te,
        e_te,
        g_te,
        groups,
    )
    _append_posthoc_topk_rows(
        topk_rows, P, valid, y, w, g_te, groups, H, risk_version, label_kind
    )
    _append_posthoc_calibration_rows(
        calibration_rows, P, valid, y, w, g_te, groups, H, risk_version, label_kind
    )
    _plot_posthoc_waterfall(P, H, risk_version, label_kind, valid, y, outdir)
    _plot_posthoc_risk_distribution(P, H, risk_version, label_kind, valid, y, outdir)
    summ = _risk_sample_summary(P)
    file_prefix = f"PosthocRecalibration_{risk_version}_{label_kind}"
    title_prefix = f"Highest predicted early-failure risk — {_posthoc_risk_label(risk_version, label_kind)}"
    row_extra = {"risk_version": str(risk_version), "model_kind": label_kind}
    if (
        EARLY_FAILURE_SAVE_POOLED_TOP_PATIENT_PLOTS
        and int(EARLY_FAILURE_TOP_PATIENT_N) > 0
    ):
        top_patient_rows.extend(
            _plot_top_early_failure_patient_distributions(
                P=P,
                p_mean=summ["mean"],
                p_q05=summ["q05"],
                p_q95=summ["q95"],
                valid=valid,
                y=y,
                status=status,
                g_te=g_te,
                groups=groups,
                H=H,
                outdir=outdir,
                top_n=EARLY_FAILURE_TOP_PATIENT_N,
                file_prefix=file_prefix,
                title_prefix=title_prefix,
                row_extra=row_extra,
            )
        )
    if (
        EARLY_FAILURE_SAVE_WITHIN_CANCER_TOP_PATIENT_PLOTS
        and int(EARLY_FAILURE_TOP_PATIENT_WITHIN_CANCER_N) > 0
    ):
        within_cancer_top_patient_rows.extend(
            _plot_within_cancer_top_early_failure_patient_distributions(
                P=P,
                p_mean=summ["mean"],
                p_q05=summ["q05"],
                p_q95=summ["q95"],
                valid=valid,
                y=y,
                status=status,
                g_te=g_te,
                groups=groups,
                H=H,
                outdir=outdir,
                top_n_per_group=EARLY_FAILURE_TOP_PATIENT_WITHIN_CANCER_N,
                file_prefix=file_prefix,
                title_prefix=f"Highest predicted early-failure risk — {_posthoc_risk_label(risk_version, label_kind)} within",
                row_extra=row_extra,
            )
        )


def run_posthoc_crossfitted_recalibration(art, outdir):
    """Fit and evaluate IPCW post-hoc survival-risk recalibration.

    Uses only outer-train out-of-fold predictions to learn calibration parameters.
    Then applies those fixed parameters to the final full-train model's outer-test
    predictions.  This leaves the final DGP training size unchanged.

    Native DGP outputs are preserved.  Additional calibrated outputs are saved
    alongside them, including calibrated posterior risk intervals obtained by
    transforming each posterior risk draw through the horizon-specific calibrator.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    oof_df = _build_or_load_oof_calibration_predictions(art, outdir)
    model = art["model"]
    alpha_dt = art["alpha_dt"]
    dt_edges = art["dt_edges"]
    Xte_t = art["Xte_t"]
    g_te_t = art["g_te_t"]
    t_tr = np.asarray(art["t_tr"], dtype=float)
    e_tr = np.asarray(art["e_tr"], dtype=int)
    g_tr = np.asarray(art["g_tr"], dtype=int)
    t_te = np.asarray(art["t_te"], dtype=float)
    e_te = np.asarray(art["e_te"], dtype=int)
    g_te = np.asarray(art["g_te"], dtype=int)
    groups = list(art["groups"])

    fit_rows = []
    metric_rows = []
    bin_rows = []
    posthoc_patient_rows = []
    posthoc_topk_rows = []
    posthoc_calibration_rows = []
    posthoc_top_patient_rows = []
    posthoc_within_cancer_top_patient_rows = []
    test_risk_df = pd.DataFrame(
        {
            "test_index": np.arange(len(t_te), dtype=int),
            "time": t_te,
            "event": e_te,
            "group_idx": g_te,
            "group": [str(groups[int(gg)]) for gg in g_te],
        }
    )

    for H in POSTHOC_CALIBRATION_H_LIST:
        H = float(H)
        H_int = int(H)
        col = f"p_native_oof_H{H_int}"
        if col not in oof_df.columns:
            continue
        p_oof = oof_df[col].to_numpy(dtype=float)

        # Draw-level posterior risks from the final full-train DGP.  The native
        # posterior intervals are computed from these draws; calibrated intervals
        # below are computed by transforming these same draws through the fixed
        # recalibration model.
        S_test_native = posterior_survival_at_H(
            model,
            Xte_t,
            g_te_t,
            H,
            alpha_dt,
            dt_edges,
            S=int(POSTHOC_CALIBRATION_PRED_MC),
        )
        P_test_native = np.clip(1.0 - np.clip(S_test_native, 0.0, 1.0), 0.0, 1.0)
        native_summ = _risk_sample_summary(P_test_native)
        p_test_native = native_summ["mean"]
        test_risk_df[f"p_native_H{H_int}"] = p_test_native
        test_risk_df[f"p_native_q05_H{H_int}"] = native_summ["q05"]
        test_risk_df[f"p_native_q25_H{H_int}"] = native_summ["q25"]
        test_risk_df[f"p_native_q50_H{H_int}"] = native_summ["q50"]
        test_risk_df[f"p_native_q75_H{H_int}"] = native_summ["q75"]
        test_risk_df[f"p_native_q95_H{H_int}"] = native_summ["q95"]

        valid_H, y_H, w_H = _ipcw_binary_event_targets(t_te, e_te, H)
        valid_H = (
            np.asarray(valid_H, bool)
            & np.isfinite(p_test_native)
            & np.isfinite(w_H)
            & (np.asarray(w_H) > 0.0)
        )
        y_H = np.asarray(y_H, dtype=int)
        w_H = np.asarray(w_H, dtype=float)
        status_H = _fixed_horizon_status_from_ipcw(t_te, e_te, valid_H, y_H)

        # Save native posthoc early-failure plots/tables once per horizon.  The
        # original uncalibrated EarlyFailure_* outputs from run_early_failure_analysis
        # are not touched; these native files provide a one-to-one counterpart to
        # the calibrated PosthocRecalibration_* outputs below.
        _save_posthoc_early_failure_outputs_for_version(
            P=P_test_native,
            H=H,
            risk_version="native",
            model_kind="native",
            valid=valid_H,
            y=y_H,
            w=w_H,
            status=status_H,
            t_te=t_te,
            e_te=e_te,
            g_te=g_te,
            groups=groups,
            outdir=outdir,
            patient_rows=posthoc_patient_rows,
            topk_rows=posthoc_topk_rows,
            calibration_rows=posthoc_calibration_rows,
            top_patient_rows=posthoc_top_patient_rows,
            within_cancer_top_patient_rows=posthoc_within_cancer_top_patient_rows,
        )

        for kind in POSTHOC_CALIBRATION_MODEL_KINDS:
            kind = str(kind)
            fit = _fit_ipcw_logistic_recalibrator(p_oof, t_tr, e_tr, g_tr, H, kind=kind)
            fit_row = dict(fit)
            fit_row["group_offsets_json"] = json.dumps(fit_row.pop("group_offsets"))
            fit_rows.append(fit_row)
            effective_kind = str(fit.get("effective_kind", kind))
            fallback_reason = str(fit.get("fallback", "none"))
            calibrator_skipped = bool(fit.get("skipped", False))

            # Native test metrics are repeated per requested kind so native/calibrated deltas are directly joinable.
            native_metric_rows = _summarize_recalibrated_metrics(
                p_test_native, t_te, e_te, g_te, groups, H, "test", kind, "native"
            )
            for r in native_metric_rows:
                r["effective_model_kind"] = effective_kind
                r["calibration_fallback"] = fallback_reason
                r["calibrator_skipped"] = calibrator_skipped
            metric_rows.extend(native_metric_rows)
            _append_recalibration_bins(
                bin_rows,
                p_test_native,
                t_te,
                e_te,
                g_te,
                groups,
                H,
                "test",
                kind,
                "native",
                P_samples=P_test_native,
            )

            # Calibrated posterior risk distribution: transform every posterior
            # risk draw, then summarize.  This gives calibrated uncertainty
            # intervals instead of only calibrating the posterior mean.
            P_test_cal = _apply_recalibrator_to_samples(P_test_native, g_te, fit)
            cal_summ = _risk_sample_summary(P_test_cal)
            p_test_cal = cal_summ["mean"]
            test_risk_df[f"p_calibrated_{kind}_H{H_int}"] = p_test_cal
            test_risk_df[f"p_calibrated_{kind}_q05_H{H_int}"] = cal_summ["q05"]
            test_risk_df[f"p_calibrated_{kind}_q25_H{H_int}"] = cal_summ["q25"]
            test_risk_df[f"p_calibrated_{kind}_q50_H{H_int}"] = cal_summ["q50"]
            test_risk_df[f"p_calibrated_{kind}_q75_H{H_int}"] = cal_summ["q75"]
            test_risk_df[f"p_calibrated_{kind}_q95_H{H_int}"] = cal_summ["q95"]

            calibrated_metric_rows = _summarize_recalibrated_metrics(
                p_test_cal, t_te, e_te, g_te, groups, H, "test", kind, "calibrated"
            )
            for r in calibrated_metric_rows:
                r["effective_model_kind"] = effective_kind
                r["calibration_fallback"] = fallback_reason
                r["calibrator_skipped"] = calibrator_skipped
            metric_rows.extend(calibrated_metric_rows)
            _append_recalibration_bins(
                bin_rows,
                p_test_cal,
                t_te,
                e_te,
                g_te,
                groups,
                H,
                "test",
                kind,
                "calibrated",
                P_samples=P_test_cal,
            )
            _save_posthoc_early_failure_outputs_for_version(
                P=P_test_cal,
                H=H,
                risk_version="calibrated",
                model_kind=kind,
                valid=valid_H,
                y=y_H,
                w=w_H,
                status=status_H,
                t_te=t_te,
                e_te=e_te,
                g_te=g_te,
                groups=groups,
                outdir=outdir,
                patient_rows=posthoc_patient_rows,
                topk_rows=posthoc_topk_rows,
                calibration_rows=posthoc_calibration_rows,
                top_patient_rows=posthoc_top_patient_rows,
                within_cancer_top_patient_rows=posthoc_within_cancer_top_patient_rows,
            )

    df_fit = pd.DataFrame(fit_rows)
    df_fit.to_csv(outdir / "posthoc_crossfit_calibration_model_params.csv", index=False)
    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(
        outdir / "posthoc_crossfit_calibration_test_metrics.csv", index=False
    )
    df_bins = pd.DataFrame(bin_rows)
    # Add parseable columns for plotting/summarizing.  Scope strings are of the
    # form test_native_pooled_pooled or test_calibrated_group_intercept_within_cancer.
    if len(df_bins):
        parsed_rows = []
        for scope_str in df_bins["scope"].astype(str).tolist():
            split = ""
            risk_version = ""
            model_kind = ""
            scope_type = ""
            for rv in ("native", "calibrated"):
                prefix = f"test_{rv}_"
                if scope_str.startswith(prefix):
                    split = "test"
                    risk_version = rv
                    rest = scope_str[len(prefix) :]
                    if rest.endswith("_within_cancer"):
                        scope_type = "within_cancer"
                        model_kind = rest[: -len("_within_cancer")]
                    elif rest.endswith("_pooled"):
                        scope_type = "pooled"
                        model_kind = rest[: -len("_pooled")]
                    else:
                        model_kind = rest
                    break
            parsed_rows.append(
                {
                    "split": split,
                    "risk_version": risk_version,
                    "model_kind": model_kind,
                    "scope_type": scope_type,
                }
            )
        parsed_df = pd.DataFrame(parsed_rows)
        for col in parsed_df.columns:
            df_bins[col] = parsed_df[col].to_numpy()
        df_bins.to_csv(
            outdir / "posthoc_crossfit_calibration_test_bins.csv", index=False
        )
        df_bin_summary = _calibration_summary_from_bins(df_bins)
        df_bin_summary.to_csv(
            outdir / "posthoc_crossfit_calibration_test_bin_summary.csv", index=False
        )
        _plot_native_vs_calibrated_bins(df_bins, outdir)
    else:
        df_bins.to_csv(
            outdir / "posthoc_crossfit_calibration_test_bins.csv", index=False
        )

    # Dedicated native + calibrated early-failure outputs with draw-level intervals.
    df_posthoc_pat = pd.DataFrame(posthoc_patient_rows)
    df_posthoc_pat.to_csv(
        outdir
        / "posthoc_crossfit_early_failure_patient_predictions_native_and_calibrated_intervals.csv",
        index=False,
    )
    df_posthoc_topk = pd.DataFrame(posthoc_topk_rows)
    df_posthoc_topk.to_csv(
        outdir
        / "posthoc_crossfit_early_failure_topk_enrichment_native_and_calibrated.csv",
        index=False,
    )
    if len(df_posthoc_topk):
        _plot_posthoc_topk_enrichment(df_posthoc_topk, outdir)
    df_posthoc_cal = pd.DataFrame(posthoc_calibration_rows)
    df_posthoc_cal.to_csv(
        outdir
        / "posthoc_crossfit_early_failure_ipcw_calibration_bins_native_and_calibrated.csv",
        index=False,
    )
    df_posthoc_cal_summary = _calibration_summary_from_bins(df_posthoc_cal)
    if len(df_posthoc_cal_summary):
        df_posthoc_cal_summary.to_csv(
            outdir
            / "posthoc_crossfit_early_failure_ipcw_calibration_summary_native_and_calibrated.csv",
            index=False,
        )
    if len(df_posthoc_cal):
        _plot_posthoc_calibration_by_version(df_posthoc_cal, outdir)
    if len(posthoc_top_patient_rows):
        pd.DataFrame(posthoc_top_patient_rows).to_csv(
            outdir
            / "posthoc_crossfit_early_failure_top_patient_posterior_distributions_native_and_calibrated.csv",
            index=False,
        )
    if len(posthoc_within_cancer_top_patient_rows):
        pd.DataFrame(posthoc_within_cancer_top_patient_rows).to_csv(
            outdir
            / "posthoc_crossfit_early_failure_within_cancer_top_patient_posterior_distributions_native_and_calibrated.csv",
            index=False,
        )

    test_risk_df.to_csv(
        outdir / "posthoc_crossfit_test_native_and_calibrated_risks.csv", index=False
    )
    meta = {
        "method": "outer-train cross-fitted IPCW logit recalibration of fixed-horizon DGP risks",
        "primary_kind": str(POSTHOC_CALIBRATION_PRIMARY_KIND),
        "model_kinds_saved": list(map(str, POSTHOC_CALIBRATION_MODEL_KINDS)),
        "horizons_months": [float(h) for h in POSTHOC_CALIBRATION_H_LIST],
        "k_folds": int(POSTHOC_CALIBRATION_K_FOLDS),
        "final_dgp_training_size_reduced": False,
        "test_outcomes_used_to_fit_calibrator": False,
        "calibration_target": "fixed-horizon event probability p_i(H)=1-S_i(H)",
        "calibrated_interval_target": "each posterior event-probability draw is transformed through the fitted horizon-specific recalibrator before q05/q50/q95 are computed",
        "native_outputs_preserved": True,
        "uncalibrated_native_outputs_saved_alongside_calibrated_outputs": True,
        "safety_gates": {
            "min_n_for_any_model": int(POSTHOC_CALIBRATION_MIN_N_FOR_ANY_MODEL),
            "min_events_for_any_model": int(
                POSTHOC_CALIBRATION_MIN_EVENTS_FOR_ANY_MODEL
            ),
            "min_nonevents_for_any_model": int(
                POSTHOC_CALIBRATION_MIN_NONEVENTS_FOR_ANY_MODEL
            ),
            "min_events_for_slope": int(POSTHOC_CALIBRATION_MIN_EVENTS_FOR_SLOPE),
            "min_nonevents_for_slope": int(POSTHOC_CALIBRATION_MIN_NONEVENTS_FOR_SLOPE),
            "min_events_per_group": int(POSTHOC_CALIBRATION_MIN_EVENTS_PER_GROUP),
            "min_nonevents_per_group": int(POSTHOC_CALIBRATION_MIN_NONEVENTS_PER_GROUP),
            "slope_constrained_nonnegative": bool(
                POSTHOC_CALIBRATION_CONSTRAIN_SLOPE_NONNEGATIVE
            ),
        },
        "notes": "Horizon-specific calibration is saved for fixed risk horizons. Intercept-only is the primary conservative calibrated version. Slope-based calibration is constrained to beta >= 0 and falls back to intercept-only if event counts are too small. Group-intercept calibration falls back to pooled if per-cancer event/non-event counts are too small. Full calibrated survival curves are not constructed, so monotonicity across horizons is not assumed.",
    }
    with open(
        outdir / "posthoc_crossfit_calibration_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2)
    return {
        "params": df_fit,
        "metrics": df_metrics,
        "bins": df_bins,
        "patient_predictions": df_posthoc_pat,
    }




# ------------------------ DGP-FC PRIORITIZED RAW-EXPRESSION PSEUDOTIME ------------------------

_PT_GENE_TOKEN = re.compile(r"[A-Z0-9-]+")


def _pt_sample_key(x):
    """Normalize TCGA sample IDs exactly like preprocessing: TCGA-xx-xxxx-01.

    Raw GDC RNA columns commonly contain vial/analyte suffixes such as 01A.
    Preprocessing stores SAMPLE_KEY using only the first two characters of the
    fourth TCGA barcode token, so pseudotime RNA loading must use the same key.
    """
    s = str(x).strip().split(":")[0].strip().replace(".", "-")
    toks = s.split("-")
    if len(toks) >= 4 and toks[0].upper() == "TCGA":
        return "-".join([toks[0], toks[1], toks[2], toks[3][:2]]).upper()
    return s.upper()


def _pt_canon_gene_symbol(x):
    """Canonicalize RNA gene symbols to match preprocessing/loadings indices."""
    s = str(x).upper().strip()
    for sep in ("///", "//", "|", ";", "/", ","):
        if sep in s:
            s = s.split(sep)[0].strip()
    m = _PT_GENE_TOKEN.search(s)
    return m.group(0) if m else s


def _pt_load_table_guess(path: Path, index_col=None):
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
        )
        if df.shape[1] == 1 and "," in str(df.columns[0]):
            return pd.read_csv(path, index_col=index_col, dtype=str, encoding="utf-8-sig", low_memory=False)
        return df
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python", index_col=index_col, dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep=None, engine="python", index_col=index_col, dtype=str, encoding="latin1")


def _pt_clean_collapse_expression(expr):
    expr = expr.copy()
    expr.index = expr.index.astype(str).map(_pt_canon_gene_symbol)
    expr = expr.loc[expr.index.notna()]
    expr = expr[~expr.index.astype(str).str.strip().isin(["", "nan", "NaN", "None"])]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    # Collapse duplicated symbols conservatively by mean using the same canonical
    # symbol convention used to build selected PC loadings during preprocessing.
    expr = expr.groupby(expr.index.astype(str), sort=False).mean()
    expr = expr.replace([np.inf, -np.inf], np.nan)
    return expr


def _pt_load_rna_matrix_subset(path, sample_keys, genes):
    """Load raw/normalized RNA matrix as genes x samples, restricted to requested genes/samples."""
    path = Path(path)
    df = _pt_load_table_guess(path)
    sample_cols = [c for c in df.columns if str(c).upper().startswith("TCGA")]
    gene_col = next(
        (
            c
            for c in ["gene_name", "gene", "GeneSymbol", "GENE_NAME", "SYMBOL", "symbol"]
            if c in df.columns
        ),
        None,
    )
    if gene_col is None:
        non_tcga = [c for c in df.columns if not str(c).upper().startswith("TCGA")]
        gene_col = non_tcga[0] if len(non_tcga) else df.columns[0]
    if len(sample_cols) == 0:
        raise ValueError(f"No TCGA sample columns detected in RNA file: {path}")
    keep_full = []
    seen = set()
    for full in sample_cols:
        norm = _pt_sample_key(full)
        if norm not in seen:
            keep_full.append(full)
            seen.add(norm)
    expr = df[[gene_col] + keep_full].copy().rename(columns={gene_col: "gene_name"}).set_index("gene_name")
    expr.columns = [_pt_sample_key(c) for c in keep_full]
    sample_keys = [_pt_sample_key(x) for x in sample_keys]
    keep_samples = [c for c in sample_keys if c in set(expr.columns)]
    genes = [_pt_canon_gene_symbol(g) for g in genes]
    if len(keep_samples) == 0:
        raise ValueError(f"No sample overlap while loading RNA pseudotime matrix from {path}")
    expr = _pt_clean_collapse_expression(expr.loc[:, keep_samples])
    keep_genes = [g for g in genes if g in set(expr.index.astype(str))]
    if len(keep_genes) == 0:
        raise ValueError(f"No selected-gene overlap while loading RNA pseudotime matrix from {path}")
    expr = expr.loc[keep_genes, keep_samples].copy()
    return expr


def _pt_source_manifest_from_prep(prep_dir):
    prep_dir = Path(prep_dir) if prep_dir is not None else None
    if prep_dir is None:
        return {}
    report_path = prep_dir / "tcga_allcancer_preprocess_report.json"
    if not report_path.exists():
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
    except Exception:
        return {}
    out = {}
    for row in rep.get("source_manifest", []):
        cc = str(row.get("cancer_code", "")).upper()
        rp = row.get("rna_file", None)
        if cc and rp:
            out[cc] = str(rp)
    return out


def _pt_cancer_code_from_row(row):
    if "CANCER_CODE" in row and pd.notna(row["CANCER_CODE"]):
        cc = str(row["CANCER_CODE"]).upper()
        if cc in {"BRCA", "LUAD", "COAD"}:
            return cc
    g = str(row.get("group", row.get("GROUP_LABEL", ""))).lower()
    if "breast" in g or "brca" in g:
        return "BRCA"
    if "lung" in g or "luad" in g:
        return "LUAD"
    if "colon" in g or "coad" in g:
        return "COAD"
    return str(row.get("group", "UNKNOWN")).upper()


def _pt_select_dgp_fc_backprojected_genes(outdir, prep_dir=None, top_per_cancer=None):
    """Select genes from DGP-FC top PC backprojections.

    Sources, in priority order:
    1. overall FC PC backprojection table, if already written;
    2. treatment-adjusted FC table plus selected PC loadings, so treatment-conditioned
       FC contributes to the pseudotime gene set without averaging treatment profiles.
    """
    outdir = Path(outdir)
    if top_per_cancer is None:
        top_per_cancer = int(DGP_FC_PSEUDOTIME_TOP_BACKPROJECTED_GENES_PER_CANCER)
    rows = []

    # Source 1: already backprojected overall FC genes.
    path = outdir / "pc_to_gene_backprojection_FCtop.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            if "gene" in df.columns and "cancer_code" in df.columns:
                df = df.copy()
                df["gene"] = df["gene"].astype(str)
                df["cancer_code"] = df["cancer_code"].astype(str).str.upper()
                fc = pd.to_numeric(df.get("fc_score", np.nan), errors="coerce").fillna(0.0).clip(lower=0.0)
                loading = pd.to_numeric(df.get("abs_loading", np.nan), errors="coerce")
                if loading.isna().all() and "loading_oriented" in df.columns:
                    loading = pd.to_numeric(df["loading_oriented"], errors="coerce").abs()
                loading = loading.fillna(0.0).abs()
                df["dgp_fc_gene_score"] = fc * loading
                for _, r in df.iterrows():
                    rows.append({
                        "cancer_code": str(r["cancer_code"]).upper(),
                        "gene": str(r["gene"]),
                        "dgp_fc_gene_score": float(r.get("dgp_fc_gene_score", 0.0)),
                        "fc_score": float(r.get("fc_score", np.nan)) if np.isfinite(pd.to_numeric(pd.Series([r.get("fc_score", np.nan)]), errors="coerce").iloc[0]) else np.nan,
                        "abs_loading": float(loading.loc[_]) if _ in loading.index else np.nan,
                        "selection_source": "overall_fc_pc_backprojection",
                    })
        except Exception:
            pass

    # Source 2: treatment-adjusted FC PC features directly backprojected through loadings.
    if prep_dir is not None:
        prep_dir = Path(prep_dir)
        tx_fc_path = outdir / "treatment_adjusted_feature_collapse" / "treatment_adjusted_feature_collapsing_pooled_and_group.csv"
        if tx_fc_path.exists():
            loadings_by_cancer = {}
            for cancer_code in ALL_CANCER_CODES:
                lp = prep_dir / f"tcga_{str(cancer_code).lower()}_selected_pc_loadings.csv"
                if lp.exists():
                    try:
                        loadings_by_cancer[str(cancer_code)] = pd.read_csv(lp, index_col=0)
                    except Exception:
                        pass
            try:
                df_fc = pd.read_csv(tx_fc_path)
                if "feature" in df_fc.columns and "FC_score" in df_fc.columns:
                    df_fc = df_fc.copy()
                    if "FC_conditioning_feature_not_ranked" in df_fc.columns:
                        df_fc = df_fc[~df_fc["FC_conditioning_feature_not_ranked"].astype(bool)].copy()
                    df_fc["FC_score"] = pd.to_numeric(df_fc["FC_score"], errors="coerce")
                    df_fc = df_fc[np.isfinite(df_fc["FC_score"].to_numpy(float))].copy()
                    df_fc = df_fc.sort_values("FC_score", ascending=False).head(max(100, int(6 * FC_TOPN * len(FC_H_LIST))))
                    for _, rr in df_fc.iterrows():
                        feat = str(rr.get("feature", ""))
                        cc, pc = _parse_pc_feature_name(feat)
                        if pc is None:
                            continue
                        candidates = []
                        if cc is not None and str(cc) in loadings_by_cancer:
                            candidates = [str(cc)]
                        else:
                            candidates = [c for c, df_load in loadings_by_cancer.items() if pc in df_load.columns]
                        for ccode in candidates:
                            df_load = loadings_by_cancer.get(ccode)
                            if df_load is None or pc not in df_load.columns:
                                continue
                            load = pd.to_numeric(df_load[pc], errors="coerce").dropna()
                            if load.empty:
                                continue
                            # Top loading genes per treatment-adjusted FC PC keep the raw-expression matrix compact.
                            top = load.abs().sort_values(ascending=False).head(50)
                            for gene, absload in top.items():
                                rows.append({
                                    "cancer_code": str(ccode).upper(),
                                    "gene": str(gene),
                                    "dgp_fc_gene_score": float(rr["FC_score"]) * float(absload),
                                    "fc_score": float(rr["FC_score"]),
                                    "abs_loading": float(absload),
                                    "selection_source": "treatment_adjusted_fc_pc_loading",
                                })
            except Exception:
                pass

    if len(rows) == 0:
        return pd.DataFrame()
    df_all = pd.DataFrame(rows)
    df_all = df_all[df_all["gene"].astype(str).str.len() > 0].copy()
    agg = (
        df_all.groupby(["cancer_code", "gene"], dropna=False)
        .agg(
            dgp_fc_gene_score=("dgp_fc_gene_score", "max"),
            mean_dgp_fc_gene_score=("dgp_fc_gene_score", "mean"),
            max_fc_score=("fc_score", "max"),
            max_abs_loading=("abs_loading", "max"),
            n_backprojection_hits=("gene", "size"),
            selection_sources=("selection_source", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    agg = agg.sort_values(["cancer_code", "dgp_fc_gene_score"], ascending=[True, False])
    out = []
    for cc, sub in agg.groupby("cancer_code", dropna=False):
        out.append(sub.head(int(top_per_cancer)).copy())
    return pd.concat(out, axis=0, ignore_index=True) if len(out) else pd.DataFrame()

def _pt_covariate_design(patient_df, pooled=True):
    """Nuisance covariates for expression trajectory fitting.

    Stage, age, survival, PFI, and DGP risk are intentionally excluded. They are
    validation/annotation variables, not covariates used to fit pseudotime.
    """
    d = patient_df.reset_index(drop=True).copy()
    frames = []
    names = []
    if bool(pooled):
        g = d.get("group", pd.Series(["group"] * len(d))).astype(str).copy()
        vc = g.value_counts(dropna=False)
        g = g.where(g.map(vc).fillna(0) >= int(DGP_FC_PSEUDOTIME_COVARIATE_MIN_N), "OTHER_RARE_CANCER")
        A = pd.get_dummies(g, prefix="cancer", drop_first=True, dtype=float)
        if A.shape[1] > 0:
            frames.append(A)
            names.extend(A.columns.astype(str).tolist())
    tx = d.get("treatment_profile", pd.Series(["TX=UNKNOWN"] * len(d))).astype(str).copy()
    vc = tx.value_counts(dropna=False)
    tx = tx.where(tx.map(vc).fillna(0) >= int(DGP_FC_PSEUDOTIME_COVARIATE_MIN_N), "TX=OTHER_RARE")
    A = pd.get_dummies(tx, prefix="tx", drop_first=True, dtype=float)
    if A.shape[1] > 0:
        frames.append(A)
        names.extend(A.columns.astype(str).tolist())
    if len(frames) == 0:
        return np.zeros((len(d), 0), dtype=float), []
    C = pd.concat(frames, axis=1).to_numpy(dtype=float)
    return C, names


def _pt_standardize_expression(Y):
    Y = np.asarray(Y, float)
    med = np.nanmedian(Y, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(Y)
    if bad.any():
        Y = Y.copy()
        rows, cols = np.where(bad)
        Y[rows, cols] = med[cols]
    mu = np.mean(Y, axis=0, keepdims=True)
    sd = np.std(Y, axis=0, ddof=0, keepdims=True)
    sd = np.where((sd < 1e-8) | (~np.isfinite(sd)), 1.0, sd)
    return (Y - mu) / sd


def _pt_residualize(Y, C):
    Y = np.asarray(Y, float)
    C = np.asarray(C, float)
    if C.ndim != 2 or C.shape[1] == 0:
        return Y.copy()
    X = np.column_stack([np.ones(Y.shape[0]), C])
    try:
        B = np.linalg.pinv(X) @ Y
        return Y - X @ B + B[0:1, :]
    except Exception:
        return Y.copy()


def _pt_pca2_and_tau(Y, random_state=0):
    Y = np.asarray(Y, float)
    if Y.shape[1] == 0:
        return np.zeros((Y.shape[0], 2)), np.linspace(0, 1, Y.shape[0]), np.nan, np.zeros((0, 2))
    Yc = Y - np.nanmean(Y, axis=0, keepdims=True)
    try:
        u, svals, vt = np.linalg.svd(Yc, full_matrices=False)
        coords = u[:, :2] * svals[:2]
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
        pc1 = coords[:, 0]
        denom = np.nanmax(pc1) - np.nanmin(pc1)
        tau = (pc1 - np.nanmin(pc1)) / (denom if denom > 1e-12 else 1.0)
        evr = (svals[0] ** 2) / max(float(np.sum(svals ** 2)), 1e-12) if len(svals) else np.nan
        load = vt[:2, :].T if vt.size else np.zeros((Y.shape[1], 2))
        if load.shape[1] == 1:
            load = np.column_stack([load[:, 0], np.zeros(load.shape[0])])
        return coords[:, :2], tau, float(evr), load[:, :2]
    except Exception:
        tau = np.linspace(0, 1, Y.shape[0])
        return np.column_stack([tau, np.zeros_like(tau)]), tau, np.nan, np.zeros((Y.shape[1], 2))


def _logit_np(x):
    x = np.clip(np.asarray(x, float), 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


def _pt_fit_phenopath_style_map(Y, C, tau_init, seed=0):
    """Small Python MAP approximation of the PhenoPath mean model.

    Y_ig = alpha_g + C_i gamma_g + tau_i lambda_g + tau_i C_i beta_g + eps_ig.
    This is not the R package; it is a penalized MAP fit of the same mean structure.
    """
    rng = np.random.default_rng(int(seed))
    Y = np.asarray(Y, float)
    C = np.asarray(C, float)
    if C.ndim != 2:
        C = np.zeros((Y.shape[0], 0), dtype=float)
    n, p = Y.shape
    k = C.shape[1]
    tau0 = np.clip(np.asarray(tau_init, float), 1e-4, 1.0 - 1e-4)
    # Orientation is arbitrary at this stage; no risk/outcome enters the fit.
    dev = device
    Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    Ct = torch.tensor(C, dtype=torch.float32, device=dev)
    raw_tau = torch.nn.Parameter(torch.tensor(_logit_np(tau0), dtype=torch.float32, device=dev))
    alpha = torch.nn.Parameter(torch.tensor(np.nanmean(Y, axis=0), dtype=torch.float32, device=dev))
    gamma = torch.nn.Parameter(torch.zeros(k, p, dtype=torch.float32, device=dev))
    lam = torch.nn.Parameter(torch.zeros(p, dtype=torch.float32, device=dev))
    beta = torch.nn.Parameter(torch.zeros(k, p, dtype=torch.float32, device=dev))
    # Initialize lambda from residualized Y on tau0.
    try:
        Yr = _pt_residualize(Y, C)
        tt = tau0 - tau0.mean()
        denom = float(np.sum(tt ** 2))
        if denom > 1e-10:
            lam0 = (tt[:, None] * (Yr - Yr.mean(axis=0, keepdims=True))).sum(axis=0) / denom
            with torch.no_grad():
                lam.copy_(torch.tensor(lam0, dtype=torch.float32, device=dev))
    except Exception:
        pass
    opt = torch.optim.Adam([raw_tau, alpha, gamma, lam, beta], lr=float(DGP_FC_PSEUDOTIME_MAP_LR))
    tau0t = torch.tensor(tau0, dtype=torch.float32, device=dev)
    l2 = float(DGP_FC_PSEUDOTIME_L2)
    tau_pen = float(DGP_FC_PSEUDOTIME_TAU_INIT_PENALTY)
    last_loss = np.nan
    for _ in range(int(DGP_FC_PSEUDOTIME_MAP_EPOCHS)):
        opt.zero_grad()
        tau = torch.sigmoid(raw_tau)
        pred = alpha.view(1, -1) + tau.view(-1, 1) * lam.view(1, -1)
        if k > 0:
            pred = pred + Ct @ gamma
            pred = pred + torch.einsum("nk,n,kp->np", Ct, tau, beta)
        mse = torch.mean((Yt - pred) ** 2)
        reg = l2 * (torch.mean(gamma ** 2) + torch.mean(lam ** 2) + torch.mean(beta ** 2))
        # Weak identifiability stabilizer: tau remains close to the covariate-adjusted PC1 seed,
        # but can move if the full interaction model improves expression fit.
        stab = tau_pen * torch.mean((tau - tau0t) ** 2)
        loss = mse + reg + stab
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_tau, alpha, gamma, lam, beta], 10.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
    with torch.no_grad():
        tau = torch.sigmoid(raw_tau).detach().cpu().numpy()
        params = {
            "alpha": alpha.detach().cpu().numpy(),
            "gamma": gamma.detach().cpu().numpy(),
            "lambda": lam.detach().cpu().numpy(),
            "beta": beta.detach().cpu().numpy(),
            "final_loss": last_loss,
        }
    return tau, params


def _pt_orient_tau(tau, patient_df):
    tau = np.asarray(tau, float)
    d = patient_df.reset_index(drop=True).copy()
    # Orientation only: no fitting uses DGP risk. Prefer integrated multihorizon DGP risk as a sign convention.
    risk_cols = [f"risk_H{int(h)}" for h in DGP_FC_PSEUDOTIME_RISK_ORIENTATION_H_LIST if f"risk_H{int(h)}" in d.columns]
    target = None
    target_name = "none"
    if len(risk_cols):
        target = d[risk_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy(float)
        target_name = "mean_DGP_predicted_risk_across_horizons"
    elif "PFI_EVENT" in d.columns:
        target = pd.to_numeric(d["PFI_EVENT"], errors="coerce").to_numpy(float)
        target_name = "PFI_EVENT_fallback"
    flipped = False
    rho = np.nan
    if target is not None and np.isfinite(target).sum() >= 5:
        rho = _raw_spearman(tau, target)
        if np.isfinite(rho) and rho < 0:
            tau = 1.0 - tau
            flipped = True
    # Normalize after possible flip.
    mn, mx = np.nanmin(tau), np.nanmax(tau)
    tau = (tau - mn) / (mx - mn if mx > mn else 1.0)
    return tau, {"orientation_target": target_name, "orientation_spearman_before_flip": float(rho) if np.isfinite(rho) else np.nan, "orientation_flipped": bool(flipped)}


def _pt_gene_trends(Y, genes, tau, seed=0):
    Y = np.asarray(Y, float)
    tau = np.asarray(tau, float)
    rows = []
    rng = np.random.default_rng(int(seed))
    B = int(DGP_FC_PSEUDOTIME_TREND_PERMUTATIONS)
    tt = tau - np.nanmean(tau)
    denom = float(np.nansum(tt ** 2))
    for j, gene in enumerate(genes):
        y = Y[:, j]
        rho = _raw_spearman(tau, y)
        slope = np.nan
        if denom > 1e-12:
            slope = float(np.nansum(tt * (y - np.nanmean(y))) / denom)
        p = np.nan
        if B > 0 and np.isfinite(rho):
            cnt = 1
            for _ in range(B):
                rp = _raw_spearman(rng.permutation(tau), y)
                if np.isfinite(rp) and abs(rp) >= abs(rho):
                    cnt += 1
            p = float(cnt / (B + 1))
        rows.append({"gene": str(gene), "rho_with_tau": float(rho) if np.isfinite(rho) else np.nan, "slope_per_tau": slope, "permutation_p_two_sided": p})
    df = pd.DataFrame(rows)
    if len(df):
        df["abs_rho"] = pd.to_numeric(df["rho_with_tau"], errors="coerce").abs()
        df["q_BH"] = _bh_fdr(pd.to_numeric(df["permutation_p_two_sided"], errors="coerce").to_numpy(float))
        df = df.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    return df


def _pt_validation_rows(scope, patient_df, tau_col="tau_resistance"):
    d = patient_df.copy()
    tau = pd.to_numeric(d[tau_col], errors="coerce").to_numpy(float)
    rows = []
    for H in [float(h) for h in FC_H_LIST]:
        col = f"risk_H{int(H)}"
        if col in d.columns:
            rows.append({"scope": str(scope), "target": col, "association": "spearman", "value": _raw_spearman(tau, pd.to_numeric(d[col], errors="coerce").to_numpy(float))})
    for col in ["PFI_TIME_MONTHS", "PFI_EVENT", "risk_mean_across_horizons", "risk_max_across_horizons"]:
        if col in d.columns:
            rows.append({"scope": str(scope), "target": col, "association": "spearman", "value": _raw_spearman(tau, pd.to_numeric(d[col], errors="coerce").to_numpy(float))})
    return rows


def _pt_plot_km(patient_df, tau_col, title, outpath):
    d = patient_df.copy()
    if tau_col not in d.columns:
        return
    try:
        d["tau_tertile"] = pd.qcut(pd.to_numeric(d[tau_col], errors="coerce"), q=3, labels=["Low pseudotime", "Mid pseudotime", "High pseudotime"], duplicates="drop")
    except Exception:
        return
    labels = [str(x) for x in d["tau_tertile"].dropna().unique().tolist()]
    if len(labels) < 2:
        return
    masks = [(d["tau_tertile"].astype(str) == lab).to_numpy(bool) for lab in labels]
    km_plot_groups(d["PFI_TIME_MONTHS"].to_numpy(float), d["PFI_EVENT"].to_numpy(int), masks, labels, title, outpath, admin_cap=False)


def _pt_plot_3d(patient_df, title, outpath, risk_col=None):
    d = patient_df.copy()
    req = ["DGP_latent_landscape_1", "DGP_latent_landscape_2", "tau_resistance"]
    if any(c not in d.columns for c in req):
        return
    x = pd.to_numeric(d["DGP_latent_landscape_1"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(d["DGP_latent_landscape_2"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(d["tau_resistance"], errors="coerce").to_numpy(float)
    if risk_col is not None and risk_col in d.columns:
        c = pd.to_numeric(d[risk_col], errors="coerce").to_numpy(float)
        clabel = str(risk_col)
    else:
        c = z
        clabel = "Resistance pseudotime"
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(ok.sum()) < 5:
        return
    fig = plt.figure(figsize=(8.0, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x[ok], y[ok], z[ok], c=c[ok], s=38, alpha=0.82)
    # Connect pseudotime-bin centroids to show the inferred cross-sectional progression path.
    dd = d.loc[ok].copy()
    try:
        dd["tau_bin"] = pd.qcut(pd.to_numeric(dd["tau_resistance"], errors="coerce"), q=5, labels=False, duplicates="drop")
        cent = dd.groupby("tau_bin", dropna=True).agg(DGP_latent_landscape_1=("DGP_latent_landscape_1", "median"), DGP_latent_landscape_2=("DGP_latent_landscape_2", "median"), tau_resistance=("tau_resistance", "median")).reset_index()
        if len(cent) >= 2:
            ax.plot(cent["DGP_latent_landscape_1"], cent["DGP_latent_landscape_2"], cent["tau_resistance"], color="black", lw=2.2, marker="o", markersize=4)
    except Exception:
        pass
    ax.set_xlabel("DGP latent PC1")
    ax.set_ylabel("DGP latent PC2")
    ax.set_zlabel("Expression pseudotime")
    ax.set_title(title, loc="left")
    cb = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.10)
    cb.set_label(clabel)
    savefig(outpath)


def _pt_plot_heatmap(Y, genes, patient_df, title, outpath):
    if Y.size == 0 or len(genes) == 0 or "tau_resistance" not in patient_df.columns:
        return
    trends = _pt_gene_trends(Y, genes, patient_df["tau_resistance"].to_numpy(float), seed=int(DGP_FC_PSEUDOTIME_RANDOM_STATE))
    top = trends.sort_values("abs_rho", ascending=False).head(int(DGP_FC_PSEUDOTIME_HEATMAP_TOP_GENES))["gene"].astype(str).tolist()
    idx = [i for i, g in enumerate(genes) if str(g) in set(top)]
    if len(idx) == 0:
        return
    order = np.argsort(pd.to_numeric(patient_df["tau_resistance"], errors="coerce").to_numpy(float))
    M = np.asarray(Y[:, idx], float)[order, :].T
    fig_w = max(8.0, min(16.0, 0.06 * M.shape[1] + 5.0))
    fig_h = max(6.0, 0.24 * M.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(M, aspect="auto", interpolation="nearest", vmin=-2.5, vmax=2.5)
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels([genes[i] for i in idx], fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel("Patients ordered by expression pseudotime")
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("Standardized expression")
    savefig(outpath)


def _run_dgp_fc_expression_pseudotime_scope(scope_name, patient_df, expr_df, H_lat, pooled=True, scope_dir=None):
    scope_dir = Path(scope_dir)
    scope_dir.mkdir(parents=True, exist_ok=True)
    d = patient_df.reset_index(drop=True).copy()
    n = len(d)
    if n < int(DGP_FC_PSEUDOTIME_MIN_N):
        return None, [{"scope": str(scope_name), "status": "skipped", "reason": f"n<{int(DGP_FC_PSEUDOTIME_MIN_N)}", "n": n}]
    sample_keys = d["SAMPLE_KEY"].astype(str).tolist() if "SAMPLE_KEY" in d.columns else []
    genes = [g for g in expr_df.index.astype(str).tolist()]
    # Align expression columns to patient order.
    cols = [sk for sk in sample_keys if sk in expr_df.columns]
    if len(cols) != len(sample_keys):
        d = d[d["SAMPLE_KEY"].astype(str).isin(expr_df.columns)].copy().reset_index(drop=True)
        sample_keys = d["SAMPLE_KEY"].astype(str).tolist()
        if len(d) < int(DGP_FC_PSEUDOTIME_MIN_N):
            return None, [{"scope": str(scope_name), "status": "skipped", "reason": "insufficient RNA-aligned samples", "n": len(d)}]
    Yraw = expr_df.loc[:, sample_keys].T.to_numpy(dtype=float)
    good = np.isfinite(np.nanvar(Yraw, axis=0)) & (np.nanvar(Yraw, axis=0) > 1e-10)
    Yraw = Yraw[:, good]
    genes = [g for g, keep in zip(genes, good) if bool(keep)]
    if len(genes) < int(DGP_FC_PSEUDOTIME_MIN_GENES):
        return None, [{"scope": str(scope_name), "status": "skipped", "reason": f"genes<{int(DGP_FC_PSEUDOTIME_MIN_GENES)}", "n": len(d), "n_genes": len(genes)}]
    Y = _pt_standardize_expression(Yraw)
    C, cov_names = _pt_covariate_design(d, pooled=bool(pooled))
    Y_resid = _pt_residualize(Y, C)
    expr_coords, tau_seed, evr, load = _pt_pca2_and_tau(Y_resid, random_state=int(DGP_FC_PSEUDOTIME_RANDOM_STATE))
    tau_map, params = _pt_fit_phenopath_style_map(Y, C, tau_seed, seed=int(DGP_FC_PSEUDOTIME_RANDOM_STATE + len(d) + len(genes)))
    tau_oriented, orient_info = _pt_orient_tau(tau_map, d)
    d["pseudotime_scope"] = str(scope_name)
    d["tau_seed_covariate_adjusted_PC1"] = tau_seed
    d["tau_phenopath_style_raw"] = tau_map
    d["tau_resistance"] = tau_oriented
    d["n_pseudotime_genes"] = int(len(genes))
    d["n_covariates"] = int(len(cov_names))
    d["pseudotime_feature_source"] = "raw_expression_of_DGP_FC_backprojected_genes"
    d["pseudotime_covariates"] = ";".join(cov_names)
    # DGP latent PCA coordinates for landscape visualization.
    if H_lat is not None and np.asarray(H_lat).ndim == 2 and np.asarray(H_lat).shape[0] >= len(patient_df):
        # H_lat is full test order; map by original test_index if available.
        idx = d["test_index"].to_numpy(int) if "test_index" in d.columns else np.arange(len(d))
        Hs = np.asarray(H_lat, float)[idx]
        coords, _, _, latent_load = _pt_pca2_and_tau(_pt_standardize_expression(Hs), random_state=int(DGP_FC_PSEUDOTIME_RANDOM_STATE))
        d["DGP_latent_landscape_1"] = coords[:, 0]
        d["DGP_latent_landscape_2"] = coords[:, 1]
    d.to_csv(scope_dir / "pseudotime_patient_table.csv", index=False)
    pd.DataFrame({"gene": genes}).to_csv(scope_dir / "pseudotime_gene_list.csv", index=False)
    pd.DataFrame(load, index=genes, columns=["expr_PC1_loading", "expr_PC2_loading"]).to_csv(scope_dir / "expression_pca_loadings_for_pseudotime_seed.csv")
    trend = _pt_gene_trends(Y, genes, d["tau_resistance"].to_numpy(float), seed=int(DGP_FC_PSEUDOTIME_RANDOM_STATE + len(genes)))
    trend.to_csv(scope_dir / "pseudotime_gene_expression_trends.csv", index=False)
    pd.DataFrame(_pt_validation_rows(scope_name, d, tau_col="tau_resistance")).to_csv(scope_dir / "pseudotime_validation_correlations.csv", index=False)
    risk_col = "risk_H24" if "risk_H24" in d.columns else ("risk_mean_across_horizons" if "risk_mean_across_horizons" in d.columns else None)
    _pt_plot_3d(d, f"{scope_name}: DGP latent landscape with expression pseudotime", scope_dir / "pseudotime_DGP_latent_landscape_3d.png", risk_col=risk_col)
    _pt_plot_km(d, "tau_resistance", f"{scope_name}: PFI by expression pseudotime tertile", scope_dir / "pseudotime_KM_tertiles.png")
    _pt_plot_heatmap(Y, genes, d, f"{scope_name}: DGP-FC genes ordered by expression pseudotime", scope_dir / "pseudotime_gene_heatmap.png")
    readme = {
        "analysis": "DGP-FC prioritized raw-expression pseudotime",
        "Y_matrix": "raw/normalized gene expression restricted to genes backprojected from DGP-FC top gene-expression PCs",
        "covariates_used_in_fit": cov_names,
        "clinical_stage_age_outcome_DGP_risk_in_fit": False,
        "orientation_only": orient_info,
        "method": "Python penalized MAP approximation to PhenoPath mean structure: Y = alpha + C gamma + tau lambda + tau*C beta + error; tau initialized from covariate-adjusted expression PC1",
        "final_loss": float(params.get("final_loss", np.nan)) if isinstance(params, dict) else np.nan,
        "n": int(len(d)),
        "n_genes": int(len(genes)),
        "expression_PC1_explained_variance_ratio_seed": float(evr) if np.isfinite(evr) else np.nan,
    }
    with open(scope_dir / "README_dgp_fc_expression_pseudotime.json", "w", encoding="utf-8") as f:
        json.dump(readme, f, indent=2)
    return d, [{"scope": str(scope_name), "status": "completed", "n": int(len(d)), "n_genes": int(len(genes)), "n_covariates": int(len(cov_names))}]


def run_dgp_fc_gene_expression_pseudotime_analysis(art, outdir):
    """Run B: DGP-FC prioritized raw-expression pseudotime.

    Clinical features such as stage/age are not included in Y and are not used as
    nuisance covariates. Cancer type and exact treatment profile are included as
    covariates because they define design/context rather than progression labels.
    """
    if not ENABLE_DGP_FC_GENE_EXPRESSION_PSEUDOTIME:
        return None
    outdir = Path(outdir)
    pt_dir = outdir / "dgp_fc_gene_expression_pseudotime"
    pt_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = art.get("prep_dir", None)
    source_manifest = _pt_source_manifest_from_prep(prep_dir)
    selected = _pt_select_dgp_fc_backprojected_genes(outdir, prep_dir=prep_dir)
    if selected is None or len(selected) == 0:
        _posthoc_log(outdir, "[POSTHOC] Skipping DGP-FC gene-expression pseudotime: no pc_to_gene_backprojection_FCtop.csv selected genes found.")
        return None
    selected.to_csv(pt_dir / "selected_DGP_FC_backprojected_genes_for_pseudotime.csv", index=False)
    model = art["model"]
    Xte_t = art["Xte_t"]
    g_te_t = art["g_te_t"]
    g_te = np.asarray(art["g_te"], int)
    t_te = np.asarray(art["t_te"], float)
    e_te = np.asarray(art["e_te"], int)
    groups = [str(x) for x in art["groups"]]
    feat_names = [str(x) for x in art["feat_names"]]
    meta_te = art.get("meta_te")
    X_s = Xte_t.detach().cpu().numpy()
    H_lat = _latent_matrix_for_landscape(model, Xte_t, g_te_t, sample=False)
    A_tx, tx_cols = _treatment_profile_from_metadata_or_design(meta_te, X_s, feat_names)
    tx_labels = _make_treatment_profile_labels(A_tx, tx_cols)
    patient_df = pd.DataFrame({
        "test_index": np.arange(len(t_te), dtype=int),
        "PFI_TIME_MONTHS": t_te,
        "PFI_EVENT": e_te,
        "group_idx": g_te,
        "group": [groups[int(g)] if int(g) < len(groups) else f"group{int(g)}" for g in g_te],
        "treatment_profile": tx_labels,
    })
    if meta_te is not None and len(meta_te) == len(patient_df):
        meta_work = meta_te.reset_index(drop=True).copy()
        for col in ["SAMPLE_KEY", "PATIENT_ID", "CANCER_CODE", "STAGE", "AGE"]:
            if col in meta_work.columns:
                patient_df[col] = meta_work[col].astype(str).to_numpy()
    if "SAMPLE_KEY" not in patient_df.columns:
        _posthoc_log(outdir, "[POSTHOC] Skipping DGP-FC gene-expression pseudotime: SAMPLE_KEY missing from metadata.")
        return None
    for H in [float(h) for h in sorted(set(list(FC_H_LIST) + list(DGP_FC_PSEUDOTIME_RISK_ORIENTATION_H_LIST)) )]:
        try:
            risk, risk_version = _landscape_risk_vector(art, outdir, H)
            patient_df[f"risk_H{int(H)}"] = np.asarray(risk, float)
            patient_df[f"risk_version_H{int(H)}"] = str(risk_version)
        except Exception:
            pass
    risk_cols = [c for c in patient_df.columns if re.match(r"^risk_H\d+$", str(c))]
    if risk_cols:
        patient_df["risk_mean_across_horizons"] = patient_df[risk_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        patient_df["risk_max_across_horizons"] = patient_df[risk_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    _make_treatment_profile_count_table(patient_df).to_csv(pt_dir / "pseudotime_treatment_profile_counts_by_cancer.csv", index=False)
    # Load raw expression for selected genes by cancer.
    expr_by_cancer = {}
    status = []
    for cc in ALL_CANCER_CODES:
        genes_cc = selected.loc[selected["cancer_code"].astype(str).str.upper() == str(cc), "gene"].astype(str).drop_duplicates().tolist()
        if len(genes_cc) < int(DGP_FC_PSEUDOTIME_MIN_GENES):
            status.append({"scope": str(cc), "status": "skipped", "reason": "insufficient selected genes", "n_genes": len(genes_cc)})
            continue
        rna_path = source_manifest.get(str(cc), None)
        if rna_path is None or not Path(rna_path).exists():
            status.append({"scope": str(cc), "status": "skipped", "reason": "RNA source file missing from preprocess manifest", "n_genes": len(genes_cc)})
            continue
        psub = patient_df[patient_df.apply(_pt_cancer_code_from_row, axis=1).astype(str).str.upper() == str(cc)].copy()
        try:
            expr_by_cancer[cc] = _pt_load_rna_matrix_subset(rna_path, psub["SAMPLE_KEY"].astype(str).tolist(), genes_cc)
        except Exception as exc:
            status.append({"scope": str(cc), "status": "skipped", "reason": f"RNA load failed: {type(exc).__name__}: {exc}", "n_genes": len(genes_cc)})
    all_tables = []
    # Pooled scope: concatenate cancer-specific selected genes into a block-diagonal-like matrix by joining on samples;
    # missing genes for other cancers remain NaN and are median-filled during standardization. Cancer is included as a covariate.
    if len(expr_by_cancer):
        pooled_expr = pd.concat(expr_by_cancer.values(), axis=0)
        # duplicated genes across cancers are allowed but collapsed by symbol to avoid duplicate rows.
        pooled_expr = pooled_expr.groupby(pooled_expr.index.astype(str), sort=False).mean()
        tab, stat = _run_dgp_fc_expression_pseudotime_scope("pooled", patient_df, pooled_expr, H_lat, pooled=True, scope_dir=pt_dir / "pooled")
        status.extend(stat)
        if tab is not None:
            all_tables.append(tab)
    # Cancer-specific scopes.
    for cc in ALL_CANCER_CODES:
        if cc not in expr_by_cancer:
            continue
        mask = patient_df.apply(_pt_cancer_code_from_row, axis=1).astype(str).str.upper() == str(cc)
        psub = patient_df.loc[mask].copy().reset_index(drop=True)
        # For cancer-specific fit, treatment profile is still a covariate; cancer is not.
        tab, stat = _run_dgp_fc_expression_pseudotime_scope(str(cc), psub, expr_by_cancer[cc], H_lat, pooled=False, scope_dir=pt_dir / str(cc))
        status.extend(stat)
        if tab is not None:
            all_tables.append(tab)
    if len(all_tables):
        pd.concat(all_tables, axis=0, ignore_index=True).to_csv(pt_dir / "pseudotime_patient_table_all_scopes.csv", index=False)
    pd.DataFrame(status).to_csv(pt_dir / "pseudotime_scope_status.csv", index=False)
    readme = {
        "analysis": "DGP-FC prioritized raw-expression pseudotime",
        "feature_selection": "top genes are selected by DGP feature-collapse ranking of gene-expression PCs followed by PC-loading backprojection to raw genes",
        "clinical_features_included": False,
        "stage_age_included": False,
        "covariates": "pooled: cancer type + exact treatment profile; cancer-specific: exact treatment profile only",
        "DGP_risk_used_for_fit": False,
        "DGP_risk_used_for_orientation_only": True,
        "note": "Clinical variables, PFI outcome, and DGP risk are validation/annotation variables, not pseudotime features.",
    }
    with open(pt_dir / "README_dgp_fc_gene_expression_pseudotime.json", "w", encoding="utf-8") as f:
        json.dump(readme, f, indent=2)
    return pd.concat(all_tables, axis=0, ignore_index=True) if len(all_tables) else None

def run_posthoc_analysis(art, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        _posthoc_log(outdir, "[POSTHOC] Starting posthoc analysis")
        model = art["model"]
        alpha_dt = art["alpha_dt"]
        dt_edges = art["dt_edges"]
        Xte_t = art["Xte_t"]
        Xtr_t = art["Xtr_t"]
        g_te_t = art["g_te_t"]
        g_tr_t = art["g_tr_t"]
        t_te = art["t_te"]
        e_te = art["e_te"]
        g_te = art["g_te"]
        groups = art["groups"]
        feat_names = art["feat_names"]

        # group c-indices at H60 for titles
        _posthoc_log(outdir, "[POSTHOC] Computing pooled and group H60 discrimination")
        risk_te_60 = posterior_neglogS_mean_at_H(
            model, Xte_t, g_te_t, 60.0, alpha_dt, dt_edges, S=FINAL_EVAL_MC_SAMPLES
        )
        risk_tr_60 = posterior_neglogS_mean_at_H(
            model, Xtr_t, g_tr_t, 60.0, alpha_dt, dt_edges, S=FINAL_EVAL_MC_SAMPLES
        )
        sign = train_orientation_sign(art["t_tr"], art["e_tr"], risk_tr_60)
        risk_te_60 = apply_orientation(risk_te_60, sign)
        group_cindex_map = {}
        rows = []
        for gi, gname in enumerate(groups):
            m = g_te == gi
            c = np.nan
            if int(m.sum()) >= 10 and int(np.asarray(e_te)[m].sum()) >= 3:
                try:
                    c, _ = harrell_c(t_te[m], risk_te_60[m], e_te[m], auto_orient=False)
                except Exception:
                    c = np.nan
            group_cindex_map[str(gname)] = float(c) if np.isfinite(c) else np.nan
            rows.append(
                {"group": str(gname), "c_test_H60": group_cindex_map[str(gname)]}
            )
        pd.DataFrame(rows).to_csv(outdir / "group_cindex_test_H60.csv", index=False)

        # pooled/group KM and predicted survival bands
        _posthoc_log(outdir, "[POSTHOC] Building KM and posterior survival band plots")
        Xall_t = torch.cat([Xtr_t, Xte_t], dim=0)
        gall_t = torch.cat([g_tr_t, g_te_t], dim=0)
        t_all = np.concatenate([art["t_tr"], t_te])
        e_all = np.concatenate([art["e_tr"], e_te])
        g_all = np.concatenate([art["g_tr"], g_te])
        t_edges_te, S_edges_te = posterior_survival_curve_edges(
            model, Xte_t, g_te_t, alpha_dt, dt_edges, S=FINAL_PRED_BAND_MC
        )
        t_edges_all, S_edges_all = posterior_survival_curve_edges(
            model, Xall_t, gall_t, alpha_dt, dt_edges, S=FINAL_PRED_BAND_MC
        )
        for H_post in [float(h) for h in POSTERIOR_H_LIST]:
            _posthoc_log(
                outdir, f"[POSTHOC] Posterior-risk tertile plots @ H={int(H_post)} mo"
            )
            S_H_te = posterior_survival_at_H(
                model,
                Xte_t,
                g_te_t,
                H_post,
                alpha_dt,
                dt_edges,
                S=FINAL_EVAL_MC_SAMPLES,
            )
            S_H_tr = posterior_survival_at_H(
                model,
                Xtr_t,
                g_tr_t,
                H_post,
                alpha_dt,
                dt_edges,
                S=FINAL_EVAL_MC_SAMPLES,
            )
            S_H_all = posterior_survival_at_H(
                model,
                Xall_t,
                gall_t,
                H_post,
                alpha_dt,
                dt_edges,
                S=FINAL_EVAL_MC_SAMPLES,
            )
            gL_te, gM_te, gH_te, _, _, stats_te = train_defined_tertiles(
                S_H_tr, S_H_te, context=f"held-out risk tertiles H={int(H_post)} mo"
            )
            gL_all, gM_all, gH_all, _, _, stats_all = train_defined_tertiles(
                S_H_tr, S_H_all, context=f"pooled display using training cutpoints H={int(H_post)} mo"
            )
            if bool(stats_te.get("fallback_used", False)):
                _posthoc_log(
                    outdir,
                    f"[POSTHOC][WARN] Test posterior-risk tertiles H={int(H_post)} mo used {stats_te.get('method')}: {stats_te.get('fallback_reason')}",
                )
            if bool(stats_all.get("fallback_used", False)):
                _posthoc_log(
                    outdir,
                    f"[POSTHOC][WARN] Pooled posterior-risk tertiles H={int(H_post)} mo used {stats_all.get('method')}: {stats_all.get('fallback_reason')}",
                )
            labels = ["Low risk", "Medium risk", "High risk"]
            km_test_out = outdir / f"KM_test_postH_H{int(H_post)}.png"
            km_test_title = f"Test-set Kaplan–Meier survival by predicted-risk tertiles (H={int(H_post)} mo)"
            km_plot_groups(
                t_te,
                e_te,
                [gL_te, gM_te, gH_te],
                labels,
                km_test_title,
                km_test_out,
                admin_cap=False,
            )
            _save_km_display_window_variants(
                t_te,
                e_te,
                [gL_te, gM_te, gH_te],
                labels,
                km_test_title,
                km_test_out,
                admin_cap=False,
            )
            km_all_out = outdir / f"KM_all_postH_H{int(H_post)}.png"
            km_all_title = f"Pooled Kaplan–Meier survival by predicted-risk tertiles (H={int(H_post)} mo)"
            km_plot_groups(
                t_all,
                e_all,
                [gL_all, gM_all, gH_all],
                labels,
                km_all_title,
                km_all_out,
                admin_cap=False,
            )
            _save_km_display_window_variants(
                t_all,
                e_all,
                [gL_all, gM_all, gH_all],
                labels,
                km_all_title,
                km_all_out,
                admin_cap=False,
            )
            plot_model_pred_survival_bands(
                t_edges_te,
                S_edges_te,
                {"Low risk": gL_te, "Medium risk": gM_te, "High risk": gH_te},
                f"Test posterior predictive survival by posterior-risk tertiles (H={int(H_post)} mo)",
                outdir / f"PredSurvBands_test_postH_H{int(H_post)}.png",
            )
            plot_km_and_pred_survival_side_by_side(
                t_te,
                e_te,
                [gL_te, gM_te, gH_te],
                labels,
                f"Test KM by posterior-risk tertiles (H={int(H_post)} mo)",
                f"Test posterior predictive survival (H={int(H_post)} mo)",
                outdir / f"KM_PredSurv_test_postH_H{int(H_post)}.png",
                True,
                t_edges_te,
                S_edges_te,
                {"Low risk": gL_te, "Medium risk": gM_te, "High risk": gH_te},
            )
            rows_within = []
            for gi, gname in enumerate(groups):
                m = g_te == gi
                n = int(np.sum(m))
                d = int(np.sum(e_te[m])) if n else 0
                if n < 10 or d < 3:
                    rows_within.append(
                        {
                            "group_idx": gi,
                            "group": gname,
                            "n": n,
                            "events": d,
                            "grouping_method": np.nan,
                            "fallback_used": np.nan,
                            "fallback_reason": np.nan,
                            "skipped": True,
                        }
                    )
                    continue
                mtr = np.asarray(art["g_tr"]) == gi
                SHg = S_H_te[:, m]
                gL, gM, gH, _, _, stats_g = train_defined_tertiles(
                    S_H_tr[:, mtr], SHg,
                    context=f"{gname} held-out risk tertiles H={int(H_post)} mo"
                )
                if bool(stats_g.get("fallback_used", False)):
                    _posthoc_log(
                        outdir,
                        f"[POSTHOC][WARN] {gname} test posterior-risk tertiles H={int(H_post)} mo used {stats_g.get('method')}: {stats_g.get('fallback_reason')}",
                    )
                safe_g = _safe_fname(gname)
                km_group_out = (
                    outdir / f"KM_test_WITHIN_group_postH_H{int(H_post)}__{safe_g}.png"
                )
                km_group_title = f"{gname}: Kaplan–Meier survival by predicted-risk tertiles (H={int(H_post)} mo)"
                p = km_plot_groups(
                    t_te[m],
                    e_te[m],
                    [gL, gM, gH],
                    labels,
                    km_group_title,
                    km_group_out,
                    admin_cap=False,
                )
                _save_km_display_window_variants(
                    t_te[m],
                    e_te[m],
                    [gL, gM, gH],
                    labels,
                    km_group_title,
                    km_group_out,
                    admin_cap=False,
                )
                pred_masks = {"Low risk": gL, "Medium risk": gM, "High risk": gH}
                pred_S_edges = S_edges_te[:, m, :]
                plot_model_pred_survival_bands(
                    t_edges_te,
                    pred_S_edges,
                    pred_masks,
                    f"{gname} — Posterior predictive survival by posterior-risk tertiles (H={int(H_post)} mo)",
                    outdir
                    / f"PredSurvBands_test_WITHIN_group_postH_H{int(H_post)}__{safe_g}.png",
                )
                plot_km_and_pred_survival_side_by_side(
                    t_te[m],
                    e_te[m],
                    [gL, gM, gH],
                    labels,
                    f"{gname} — Empirical KM (H={int(H_post)} mo)",
                    f"{gname} — Posterior predictive survival (H={int(H_post)} mo)",
                    outdir
                    / f"KM_PredSurv_test_WITHIN_group_postH_H{int(H_post)}__{safe_g}.png",
                    True,
                    t_edges_te,
                    pred_S_edges,
                    pred_masks,
                )
                rows_within.append(
                    {
                        "group_idx": gi,
                        "group": gname,
                        "n": n,
                        "events": d,
                        "n_low": int(gL.sum()),
                        "n_med": int(gM.sum()),
                        "n_high": int(gH.sum()),
                        "grouping_method": stats_g.get("method"),
                        "fallback_used": bool(stats_g.get("fallback_used", False)),
                        "fallback_reason": stats_g.get("fallback_reason"),
                        "logrank_p_cap": float(p) if np.isfinite(p) else np.nan,
                        "skipped": False,
                        "H": float(H_post),
                    }
                )
            pd.DataFrame(rows_within).to_csv(
                outdir / f"KM_test_WITHIN_group_postH_H{int(H_post)}__summary.csv",
                index=False,
            )

        meta_te = art.get("meta_te")
        if _has_treatment_flags(meta_te) and ENABLE_TREATMENT_COMBO_KM:
            try:
                _posthoc_log(
                    outdir, "[POSTHOC] Treatment-conditioned posterior-H subgroup plots"
                )
                meta_te = meta_te.reset_index(drop=True).copy()
                if len(meta_te) != len(t_te):
                    _posthoc_log(
                        outdir,
                        f"[POSTHOC][WARN] meta_te length {len(meta_te)} does not match test length {len(t_te)}; skipping treatment-conditioned plots",
                    )
                else:
                    meta_te["TREATMENT_COMBO"] = build_treatment_combo_series(meta_te)
                    combo_counts = summarize_treatment_combos(meta_te)
                    combo_counts.to_csv(
                        outdir / "treatment_combo_counts_test.csv", index=False
                    )
                    combo_overall = (
                        meta_te.groupby("TREATMENT_COMBO", dropna=False)
                        .agg(
                            n=("TREATMENT_COMBO", "size"),
                            events=(
                                "PFI_EVENT",
                                lambda s: int(
                                    pd.to_numeric(s, errors="coerce").fillna(0).sum()
                                ),
                            ),
                        )
                        .reset_index()
                        .sort_values(
                            ["n", "events", "TREATMENT_COMBO"],
                            ascending=[False, False, True],
                        )
                        .reset_index(drop=True)
                    )
                    combo_overall.to_csv(
                        outdir / "treatment_combo_counts_test_overall.csv", index=False
                    )
                    combo_labels_all = meta_te["TREATMENT_COMBO"].astype(str).to_numpy()
                    t_tr = np.asarray(art["t_tr"], float)
                    e_tr = np.asarray(art["e_tr"], int)
                    treatment_summary_rows = []
                    for H_post in [float(h) for h in TREATMENT_POSTERIOR_H_LIST]:
                        _posthoc_log(
                            outdir,
                            f"[POSTHOC] Treatment-conditioned subgroup plots @ H={int(H_post)} mo",
                        )
                        S_H_te_combo = posterior_survival_at_H(
                            model,
                            Xte_t,
                            g_te_t,
                            H_post,
                            alpha_dt,
                            dt_edges,
                            S=FINAL_EVAL_MC_SAMPLES,
                        )
                        risk_te_combo_raw = np.mean(
                            -np.log(np.clip(S_H_te_combo, 1e-12, 1.0)), axis=0
                        )
                        risk_tr_combo_raw = posterior_neglogS_mean_at_H(
                            model,
                            Xtr_t,
                            g_tr_t,
                            H_post,
                            alpha_dt,
                            dt_edges,
                            S=FINAL_EVAL_MC_SAMPLES,
                        )
                        sign_combo = train_orientation_sign(
                            t_tr, e_tr, risk_tr_combo_raw
                        )
                        risk_te_combo = apply_orientation(risk_te_combo_raw, sign_combo)
                        rows_combo_H = []
                        for gi, gname in enumerate(groups):
                            mg = np.asarray(g_te == gi, bool)
                            if int(np.sum(mg)) == 0:
                                continue
                            meta_g = meta_te.loc[mg].copy()
                            vc = meta_g["TREATMENT_COMBO"].value_counts(dropna=False)
                            if int(TREATMENT_COMBO_MAX_PER_GROUP) > 0:
                                vc = vc.head(int(TREATMENT_COMBO_MAX_PER_GROUP))
                            for combo_name in vc.index.tolist():
                                mcombo = mg & (combo_labels_all == str(combo_name))
                                n_combo = int(np.sum(mcombo))
                                d_combo = int(np.sum(e_te[mcombo])) if n_combo else 0
                                row_base = {
                                    "H": float(H_post),
                                    "group": str(gname),
                                    "treatment_combo": str(combo_name),
                                    "n": n_combo,
                                    "events": d_combo,
                                }
                                if n_combo < int(
                                    TREATMENT_COMBO_MIN_N
                                ) or d_combo < int(TREATMENT_COMBO_MIN_EVENTS):
                                    row_base.update(
                                        {
                                            "grouping_method": np.nan,
                                            "fallback_used": np.nan,
                                            "fallback_reason": np.nan,
                                            "skipped": True,
                                            "reason": f"n<{int(TREATMENT_COMBO_MIN_N)} or events<{int(TREATMENT_COMBO_MIN_EVENTS)}",
                                        }
                                    )
                                    rows_combo_H.append(row_base)
                                    treatment_summary_rows.append(row_base.copy())
                                    continue
                                SH_sub = S_H_te_combo[:, mcombo]
                                gL_sub, gM_sub, gH_sub, _, _, stats_sub = post_groups(
                                    SH_sub,
                                    context=f"{gname} / {combo_name} test posterior-risk tertiles H={int(H_post)} mo",
                                )
                                if bool(stats_sub.get("fallback_used", False)):
                                    _posthoc_log(
                                        outdir,
                                        f"[POSTHOC][WARN] {gname} / {combo_name} test posterior-risk tertiles H={int(H_post)} mo used {stats_sub.get('method')}: {stats_sub.get('fallback_reason')}",
                                    )
                                labels_sub = ["Low risk", "Medium risk", "High risk"]
                                safe_group = _safe_fname(gname)
                                safe_combo = _safe_fname(combo_name)
                                km_combo_out = (
                                    outdir
                                    / f"KM_test_WITHIN_group_treatment_combo_postH_H{int(H_post)}__{safe_group}__{safe_combo}.png"
                                )
                                km_combo_title = f"{gname}: {combo_name} — Kaplan–Meier survival by predicted-risk tertiles (H={int(H_post)} mo)"
                                p_combo = km_plot_groups(
                                    t_te[mcombo],
                                    e_te[mcombo],
                                    [gL_sub, gM_sub, gH_sub],
                                    labels_sub,
                                    km_combo_title,
                                    km_combo_out,
                                    admin_cap=False,
                                )
                                _save_km_display_window_variants(
                                    t_te[mcombo],
                                    e_te[mcombo],
                                    [gL_sub, gM_sub, gH_sub],
                                    labels_sub,
                                    km_combo_title,
                                    km_combo_out,
                                    admin_cap=False,
                                )
                                pred_masks = {
                                    "Low risk": gL_sub,
                                    "Medium risk": gM_sub,
                                    "High risk": gH_sub,
                                }
                                pred_S_edges = S_edges_te[:, mcombo, :]
                                plot_model_pred_survival_bands(
                                    t_edges_te,
                                    pred_S_edges,
                                    pred_masks,
                                    f"{gname} — {combo_name} — Posterior predictive survival by posterior-risk tertiles (H={int(H_post)} mo)",
                                    outdir
                                    / f"PredSurvBands_test_WITHIN_group_treatment_combo_postH_H{int(H_post)}__{safe_group}__{safe_combo}.png",
                                )
                                plot_km_and_pred_survival_side_by_side(
                                    t_te[mcombo],
                                    e_te[mcombo],
                                    [gL_sub, gM_sub, gH_sub],
                                    labels_sub,
                                    f"{gname} — {combo_name} — Empirical KM (H={int(H_post)} mo)",
                                    f"{gname} — {combo_name} — Posterior predictive survival (H={int(H_post)} mo)",
                                    outdir
                                    / f"KM_PredSurv_test_WITHIN_group_treatment_combo_postH_H{int(H_post)}__{safe_group}__{safe_combo}.png",
                                    True,
                                    t_edges_te,
                                    pred_S_edges,
                                    pred_masks,
                                )
                                c_combo = np.nan
                                if n_combo >= 10 and d_combo >= 3:
                                    try:
                                        c_combo, _ = harrell_c(
                                            t_te[mcombo],
                                            risk_te_combo[mcombo],
                                            e_te[mcombo],
                                            auto_orient=False,
                                        )
                                    except Exception:
                                        c_combo = np.nan
                                hr_info = _fit_high_vs_low_hr(
                                    t_te[mcombo], e_te[mcombo], gL_sub, gH_sub
                                )
                                row_base.update(
                                    {
                                        "n_low": int(np.sum(gL_sub)),
                                        "n_med": int(np.sum(gM_sub)),
                                        "n_high": int(np.sum(gH_sub)),
                                        "grouping_method": stats_sub.get("method"),
                                        "fallback_used": bool(
                                            stats_sub.get("fallback_used", False)
                                        ),
                                        "fallback_reason": stats_sub.get(
                                            "fallback_reason"
                                        ),
                                        "c_test": (
                                            float(c_combo)
                                            if np.isfinite(c_combo)
                                            else np.nan
                                        ),
                                        "logrank_p": (
                                            float(p_combo)
                                            if np.isfinite(p_combo)
                                            else np.nan
                                        ),
                                        "skipped": False,
                                        "reason": "",
                                    }
                                )
                                row_base.update(hr_info)
                                rows_combo_H.append(row_base)
                                treatment_summary_rows.append(row_base.copy())
                        df_combo_H = pd.DataFrame(rows_combo_H)
                        df_combo_H.to_csv(
                            outdir
                            / f"treatment_combo_postH_H{int(H_post)}__summary.csv",
                            index=False,
                        )
                        plot_treatment_hr_forest(
                            (
                                df_combo_H[
                                    df_combo_H.get("skipped", False) == False
                                ].copy()
                                if len(df_combo_H)
                                else df_combo_H
                            ),
                            H_post,
                            outdir
                            / f"TreatmentCombo_HR_Forest_test_postH_H{int(H_post)}.png",
                            title=f"High-vs-low posterior risk HR within cancer × treatment subgroup (H={int(H_post)} mo)",
                        )
                    pd.DataFrame(treatment_summary_rows).to_csv(
                        outdir / "treatment_combo_postH_all_summary.csv", index=False
                    )
            except Exception as exc:
                _posthoc_log(
                    outdir,
                    f"[POSTHOC][WARN] Treatment-conditioned block failed: {type(exc).__name__}: {exc}",
                )

        # model-standardized treatment effects, avoiding tiny treatment-specific KM strata
        try:
            if _has_treatment_flags(art.get("meta_te")):
                _posthoc_log(
                    outdir,
                    "[POSTHOC] Model-standardized treatment risk, survival, and RMST contrasts",
                )
                run_standardized_treatment_effect_analysis(art, outdir)
        except Exception as exc:
            _posthoc_log(
                outdir,
                f"[POSTHOC][WARN] Standardized treatment-effect block failed: {type(exc).__name__}: {exc}",
            )

        # early failure prediction summaries with Bayesian uncertainty
        try:
            _posthoc_log(
                outdir,
                "[POSTHOC] Early failure prediction, calibration, top-k enrichment, and posterior intervals",
            )
            run_early_failure_analysis(art, outdir)
        except Exception as exc:
            _posthoc_log(
                outdir,
                f"[POSTHOC][WARN] Early-failure block failed: {type(exc).__name__}: {exc}",
            )

        # risk UQ, trajectories, heatmaps
        try:
            _posthoc_log(
                outdir,
                "[POSTHOC] Risk-UQ, representative posterior distributions, trajectories, and latent heatmaps",
            )
            plot_risk_uq_violin(
                model,
                Xte_t,
                g_te_t,
                groups,
                alpha_dt,
                dt_edges,
                outdir,
                H_list=RISK_UQ_H_LIST,
            )
            df_risk_summary = export_posterior_risk_summary(
                model,
                Xte_t,
                g_te_t,
                t_te,
                e_te,
                groups,
                alpha_dt,
                dt_edges,
                outdir,
                H_list=RISK_UQ_H_LIST,
            )
            plot_representative_patient_posterior_risk_distributions(
                model,
                Xte_t,
                g_te_t,
                t_te,
                e_te,
                groups,
                alpha_dt,
                dt_edges,
                outdir,
                n_per_group=UQ_REPRESENTATIVE_DENSITIES_PER_GROUP,
                H_list=RISK_UQ_H_LIST,
                S=RISK_UQ_MC,
            )
            export_group_uncertainty_summary(df_risk_summary, outdir)
            plot_representative_patient_risk_trajectories(
                model,
                Xte_t,
                g_te_t,
                t_te,
                e_te,
                groups,
                alpha_dt,
                dt_edges,
                outdir,
                group_cindex_map,
                n_per_group=REPRESENTATIVE_PATIENTS_N,
            )
            _plot_latent_structure_heatmap(model, Xte_t, g_te_t, groups, outdir)
            _export_mt_shared_latent_matrices(model, groups, outdir)
            _export_mt_output_matrices(model, groups, outdir)
            export_branch_uncertainty_decomposition(
                model,
                Xte_t,
                g_te_t,
                t_te,
                e_te,
                groups,
                alpha_dt,
                dt_edges,
                outdir,
                H_list=BRANCH_DECOMP_H_LIST,
                S=BRANCH_DECOMP_MC,
            )
        except Exception as exc:
            _posthoc_log(
                outdir,
                f"[POSTHOC][WARN] Risk/UQ/heatmap block failed: {type(exc).__name__}: {exc}",
            )

        # Curves, Brier, IBS, Dxy summaries
        _posthoc_log(
            outdir, "[POSTHOC] AUROC/AUPRC, Brier, IBS, and Somers' Dxy summaries"
        )
        curve_rows = []
        metric_rows = []
        group_metric_rows = []
        t_edges_bs, S_edges_bs = posterior_survival_curve_edges(
            model, Xte_t, g_te_t, alpha_dt, dt_edges, S=FINAL_EVAL_MC_SAMPLES
        )
        mean_S_edges_bs = np.mean(S_edges_bs, axis=0)
        brier_curve_rows = []
        group_brier_curve_rows = []
        for j, t_point in enumerate(np.asarray(t_edges_bs[1:], float), start=1):
            bs = brier_score_ipcw_at_H(
                mean_S_edges_bs[:, j], t_te, e_te, float(t_point)
            )
            brier_curve_rows.append(
                {
                    "time_months": float(t_point),
                    "brier": float(bs) if np.isfinite(bs) else np.nan,
                }
            )
            for gi, gname in enumerate(groups):
                mg = np.asarray(g_te) == gi
                n_g = int(np.sum(mg))
                e_g = int(np.sum(np.asarray(e_te)[mg])) if n_g else 0
                if n_g >= 10 and e_g >= 3:
                    bs_g = brier_score_ipcw_at_H(
                        mean_S_edges_bs[mg, j],
                        np.asarray(t_te)[mg],
                        np.asarray(e_te)[mg],
                        float(t_point),
                    )
                else:
                    bs_g = np.nan
                group_brier_curve_rows.append(
                    {
                        "time_months": float(t_point),
                        "group": str(gname),
                        "n": n_g,
                        "events": e_g,
                        "brier": float(bs_g) if np.isfinite(bs_g) else np.nan,
                    }
                )
        brier_curve_df = pd.DataFrame(brier_curve_rows)
        brier_curve_df.to_csv(outdir / "test_brier_curve.csv", index=False)
        pd.DataFrame(group_brier_curve_rows).to_csv(
            outdir / "test_group_brier_curve.csv", index=False
        )
        ok_brier = brier_curve_df[
            np.isfinite(pd.to_numeric(brier_curve_df["brier"], errors="coerce"))
        ].copy()
        if len(ok_brier) >= 2:
            fig, ax = plt.subplots(figsize=(7.4, 5.0))
            ax.plot(
                ok_brier["time_months"].to_numpy(dtype=float),
                ok_brier["brier"].to_numpy(dtype=float),
                lw=2.0,
            )
            ax.set_xlabel("Time (months)")
            ax.set_ylabel("IPCW Brier score")
            ax.set_title("Test-set IPCW Brier curve")
            ax.grid(True, alpha=0.18)
            savefig(outdir / "test_brier_curve.png")

        for H_curve in [float(h) for h in POSTERIOR_H_LIST]:
            _posthoc_log(
                outdir,
                f"[POSTHOC] IPCW discrimination/calibration curves @ H={int(H_curve)} mo",
            )
            risk_te_curve = posterior_neglogS_mean_at_H(
                model,
                Xte_t,
                g_te_t,
                float(H_curve),
                alpha_dt,
                dt_edges,
                S=FINAL_EVAL_MC_SAMPLES,
            )
            risk_tr_curve = posterior_neglogS_mean_at_H(
                model,
                Xtr_t,
                g_tr_t,
                float(H_curve),
                alpha_dt,
                dt_edges,
                S=FINAL_EVAL_MC_SAMPLES,
            )
            sign_curve = train_orientation_sign(art["t_tr"], art["e_tr"], risk_tr_curve)
            risk_te_curve = apply_orientation(risk_te_curve, sign_curve)
            c_test = np.nan
            try:
                c_test, _ = harrell_c(t_te, risk_te_curve, e_te, auto_orient=False)
            except Exception:
                c_test = np.nan
            dxy_test = float(2.0 * c_test - 1.0) if np.isfinite(c_test) else np.nan
            surv_te_curve = np.mean(
                posterior_survival_at_H(
                    model,
                    Xte_t,
                    g_te_t,
                    float(H_curve),
                    alpha_dt,
                    dt_edges,
                    S=FINAL_EVAL_MC_SAMPLES,
                ),
                axis=0,
            )
            bs_at_H = brier_score_ipcw_at_H(surv_te_curve, t_te, e_te, float(H_curve))
            ibs_val = _ibs_from_brier_curve(
                brier_curve_df["time_months"].to_numpy(dtype=float),
                brier_curve_df["brier"].to_numpy(dtype=float),
                float(H_curve),
            )
            event_prob_te_curve = 1.0 - np.clip(surv_te_curve, 0.0, 1.0)
            curve_row = plot_auroc_auprc_curves(
                event_prob_te_curve,
                t_te,
                e_te,
                float(H_curve),
                outdir,
                prefix="test",
                scope="pooled",
                group="pooled",
                title_prefix="Test set",
            )
            curve_rows.append(curve_row)
            if SAVE_WITHIN_CANCER_AUROC_AUPRC_CURVES:
                for gi_curve, gname_curve in enumerate(groups):
                    mg_curve = np.asarray(g_te) == int(gi_curve)
                    if int(np.sum(mg_curve)) <= 0:
                        continue
                    safe_group_curve = _safe_fname(str(gname_curve))
                    group_curve_row = plot_auroc_auprc_curves(
                        event_prob_te_curve[mg_curve],
                        np.asarray(t_te)[mg_curve],
                        np.asarray(e_te)[mg_curve],
                        float(H_curve),
                        outdir,
                        prefix=f"test_WITHIN_{safe_group_curve}",
                        scope="within_cancer",
                        group=str(gname_curve),
                        title_prefix=f"{str(gname_curve)} test set",
                    )
                    curve_rows.append(group_curve_row)
            metric_rows.append(
                {
                    "H": int(H_curve),
                    "discrimination_method": "IPCW fixed-horizon AUROC/AUPRC",
                    "c_test": float(c_test) if np.isfinite(c_test) else np.nan,
                    "somers_dxy_test": (
                        float(dxy_test) if np.isfinite(dxy_test) else np.nan
                    ),
                    "brier_at_H": float(bs_at_H) if np.isfinite(bs_at_H) else np.nan,
                    "ibs_up_to_H": float(ibs_val) if np.isfinite(ibs_val) else np.nan,
                }
            )
            for gi, gname in enumerate(groups):
                mg = np.asarray(g_te) == gi
                n_g = int(np.sum(mg))
                e_g = int(np.sum(np.asarray(e_te)[mg])) if n_g else 0
                c_g = np.nan
                dxy_g = np.nan
                bs_g = np.nan
                ibs_g = np.nan
                if n_g >= 10 and e_g >= 3:
                    try:
                        c_g, _ = harrell_c(
                            np.asarray(t_te)[mg],
                            np.asarray(risk_te_curve)[mg],
                            np.asarray(e_te)[mg],
                            auto_orient=False,
                        )
                    except Exception:
                        c_g = np.nan
                    dxy_g = float(2.0 * c_g - 1.0) if np.isfinite(c_g) else np.nan
                    bs_g = brier_score_ipcw_at_H(
                        surv_te_curve[mg],
                        np.asarray(t_te)[mg],
                        np.asarray(e_te)[mg],
                        float(H_curve),
                    )
                    df_bg = pd.DataFrame(group_brier_curve_rows)
                    ibs_g = _ibs_from_brier_curve(
                        df_bg.loc[
                            df_bg["group"].astype(str) == str(gname), "time_months"
                        ].to_numpy(dtype=float),
                        df_bg.loc[
                            df_bg["group"].astype(str) == str(gname), "brier"
                        ].to_numpy(dtype=float),
                        float(H_curve),
                    )
                group_curve_metric = (
                    _auroc_auprc_summary(
                        event_prob_te_curve[mg],
                        np.asarray(t_te)[mg],
                        np.asarray(e_te)[mg],
                        float(H_curve),
                    )
                    if n_g > 0
                    else {
                        "auroc": np.nan,
                        "auprc": np.nan,
                        "n_eval": 0,
                        "n_pos": 0,
                        "n_neg": 0,
                    }
                )
                group_metric_rows.append(
                    {
                        "H": int(H_curve),
                        "group": str(gname),
                        "discrimination_method": "IPCW fixed-horizon AUROC/AUPRC",
                        "n": n_g,
                        "events": e_g,
                        "fixed_horizon_n_eval": int(
                            group_curve_metric.get("n_eval", 0)
                        ),
                        "fixed_horizon_n_pos": int(group_curve_metric.get("n_pos", 0)),
                        "fixed_horizon_n_neg": int(group_curve_metric.get("n_neg", 0)),
                        "AUROC": (
                            float(group_curve_metric.get("auroc", np.nan))
                            if np.isfinite(group_curve_metric.get("auroc", np.nan))
                            else np.nan
                        ),
                        "AUPRC": (
                            float(group_curve_metric.get("auprc", np.nan))
                            if np.isfinite(group_curve_metric.get("auprc", np.nan))
                            else np.nan
                        ),
                        "c_test": float(c_g) if np.isfinite(c_g) else np.nan,
                        "somers_dxy_test": (
                            float(dxy_g) if np.isfinite(dxy_g) else np.nan
                        ),
                        "brier_at_H": float(bs_g) if np.isfinite(bs_g) else np.nan,
                        "ibs_up_to_H": float(ibs_g) if np.isfinite(ibs_g) else np.nan,
                    }
                )
        if len(curve_rows):
            pd.DataFrame(curve_rows).to_csv(
                outdir / "test_AUROC_AUPRC_summary.csv", index=False
            )
        if len(metric_rows):
            df_metrics = pd.DataFrame(metric_rows)
            if len(curve_rows):
                df_curve_rows = pd.DataFrame(curve_rows)
                df_curve_pooled = df_curve_rows[
                    df_curve_rows.get("scope", "pooled").astype(str) == "pooled"
                ].copy()
                df_metrics = df_metrics.merge(df_curve_pooled, on="H", how="left")
            df_metrics.to_csv(outdir / "test_survival_metric_summary.csv", index=False)
            df_metrics[["H", "somers_dxy_test"]].to_csv(
                outdir / "test_somers_dxy_summary.csv", index=False
            )
            df_metrics[["H", "brier_at_H", "ibs_up_to_H"]].to_csv(
                outdir / "test_brier_ibs_summary.csv", index=False
            )
            for _, r in df_metrics.sort_values("H").iterrows():
                msg = (
                    f"[POSTHOC][METRICS] H={int(r['H'])} | "
                    f"AUROC={float(r['AUROC']) if np.isfinite(r.get('AUROC', np.nan)) else np.nan:.4f} | "
                    f"AUPRC={float(r['AUPRC']) if np.isfinite(r.get('AUPRC', np.nan)) else np.nan:.4f} | "
                    f"SomersDxy={float(r['somers_dxy_test']) if np.isfinite(r.get('somers_dxy_test', np.nan)) else np.nan:.4f} | "
                    f"Brier={float(r['brier_at_H']) if np.isfinite(r.get('brier_at_H', np.nan)) else np.nan:.4f} | "
                    f"IBS={float(r['ibs_up_to_H']) if np.isfinite(r.get('ibs_up_to_H', np.nan)) else np.nan:.4f}"
                )
                print(msg)
                _posthoc_log(outdir, msg)
        if len(group_metric_rows):
            df_group_metrics = pd.DataFrame(group_metric_rows)
            df_group_metrics.to_csv(
                outdir / "test_group_survival_metric_summary.csv", index=False
            )
            df_group_metrics[["H", "group", "somers_dxy_test"]].to_csv(
                outdir / "test_group_somers_dxy_summary.csv", index=False
            )
            df_group_metrics[["H", "group", "brier_at_H", "ibs_up_to_H"]].to_csv(
                outdir / "test_group_brier_ibs_summary.csv", index=False
            )
            for _, r in df_group_metrics.sort_values(["H", "group"]).iterrows():
                msg = (
                    f"[POSTHOC][GROUP METRICS] H={int(r['H'])} | {r['group']} | "
                    f"AUROC={float(r['AUROC']) if np.isfinite(r.get('AUROC', np.nan)) else np.nan:.4f} | "
                    f"AUPRC={float(r['AUPRC']) if np.isfinite(r.get('AUPRC', np.nan)) else np.nan:.4f} | "
                    f"SomersDxy={float(r['somers_dxy_test']) if np.isfinite(r.get('somers_dxy_test', np.nan)) else np.nan:.4f} | "
                    f"Brier={float(r['brier_at_H']) if np.isfinite(r.get('brier_at_H', np.nan)) else np.nan:.4f} | "
                    f"IBS={float(r['ibs_up_to_H']) if np.isfinite(r.get('ibs_up_to_H', np.nan)) else np.nan:.4f}"
                )
                print(msg)
                _posthoc_log(outdir, msg)

        try:
            _posthoc_log(
                outdir, "[POSTHOC] Horizon calibration and groupwise coverage summaries"
            )
            export_horizon_calibration_summary(
                model,
                Xte_t,
                g_te_t,
                t_te,
                e_te,
                groups,
                alpha_dt,
                dt_edges,
                outdir,
                H_list=POSTERIOR_H_LIST,
            )
            export_groupwise_coverage_summary(
                model,
                Xte_t,
                g_te_t,
                t_te,
                e_te,
                groups,
                alpha_dt,
                dt_edges,
                outdir,
                H_list=POSTERIOR_H_LIST,
            )
        except Exception as exc:
            _posthoc_log(
                outdir,
                f"[POSTHOC][WARN] Calibration/coverage block failed: {type(exc).__name__}: {exc}",
            )


        # Run recalibration before any FC work so KM/posterior-risk/calibration outputs are produced first.

        if ENABLE_POSTHOC_RECALIBRATION:
            try:
                _posthoc_log(
                    outdir,
                    "[POSTHOC] Cross-fitted IPCW logit recalibration on outer-train predictions",
                )
                run_posthoc_crossfitted_recalibration(art, outdir)
            except Exception as exc:
                _posthoc_log(
                    outdir,
                    f"[POSTHOC][WARN] Post-hoc recalibration block failed: {type(exc).__name__}: {exc}",
                )
                _write_posthoc_traceback(outdir, "posthoc_recalibration_exception", exc)
        # Feature collapsing on the best test split only
        _posthoc_log(outdir, "[POSTHOC] Feature-collapsing analysis")
        Xte_s = art["Xte_t"].detach().cpu().numpy()

        _posthoc_log(
            outdir,
            "[POSTHOC] Legacy WFC blocks are disabled; the single corrected treatment-conditioned WFC runs from the saved-artifact array.",
        )

        pc_fc_summary_rows = []
        pc_fc_gene_rows = []
        pc_backprojection_available = _has_pc_features(feat_names)
        if not pc_backprojection_available:
            _posthoc_log(
                outdir,
                "[POSTHOC] Skipping FC-driven PC backprojection: no gene-expression PC features were detected in the design matrix.",
            )
        if ENABLE_BRANCH_FC:
            try:
                _posthoc_log(outdir, "[POSTHOC] Branch-specific FC baseline")
                df_branch = branch_specific_fc_df(
                    model, Xte_s, g_te, feat_names, collapse_mode=FC_COLLAPSE_MODE
                )
            except Exception as exc:
                df_branch = None
                _posthoc_log(
                    outdir,
                    f"[POSTHOC][WARN] branch_specific_fc_df failed: {type(exc).__name__}: {exc}",
                )
        else:
            df_branch = None
            _posthoc_log(
                outdir,
                "[POSTHOC] Skipping branch FC; treatment-adjusted and treatment-conditioned FC remain enabled.",
            )

        for H_fc in ([float(h) for h in FC_H_LIST] if (ENABLE_BRANCH_FC or ENABLE_OVERALL_FC) else []):
            try:
                _posthoc_log(outdir, f"[POSTHOC] FC support computation for branch/treatment outputs @ H={int(H_fc)} mo")
                df_fc = overall_fc_df(
                    model,
                    Xte_s,
                    g_te,
                    feat_names,
                    alpha_dt,
                    dt_edges,
                    H=H_fc,
                    S=FC_S,
                    R=FC_REPEATS,
                    use_crn=FC_USE_CRN,
                )
                if ENABLE_OVERALL_FC:
                    df_fc.to_csv(
                        outdir / f"feature_collapsing_test_H{int(H_fc)}.csv", index=False
                    )
                    _plot_topN_fc(
                        df_fc,
                        "FC_score",
                        f"Test feature collapsing importance @ H={int(H_fc)} mo",
                        outdir / f"feature_collapsing_test_H{int(H_fc)}_top{FC_TOPN}.png",
                        topN=FC_TOPN,
                        xlabel="Wasserstein FC score",
                    )
                nlS_mean_fc = np.mean(
                    neglogS_samples_at_H_mt(
                        model,
                        Xte_s,
                        g_te_t,
                        float(H_fc),
                        alpha_dt,
                        dt_edges,
                        S=int(FC_S),
                    ),
                    axis=0,
                )
                if ENABLE_OVERALL_FC and pc_backprojection_available:
                    try:
                        _posthoc_log(
                            outdir,
                            f"[POSTHOC] FC-driven PC backprojection @ H={int(H_fc)} mo",
                        )
                        summary_rows_h, gene_rows_h = (
                            run_fc_driven_pc_backprojection_multiH(
                                art.get("prep_dir"),
                                outdir,
                                df_fc,
                                feat_names,
                                Xte_s,
                                nlS_mean_fc,
                                H_fc,
                                top_pc_features=min(10, int(FC_TOPN)),
                                top_genes=24,
                                scope_tag="test",
                            )
                        )
                        pc_fc_summary_rows.extend(summary_rows_h)
                        pc_fc_gene_rows.extend(gene_rows_h)
                    except Exception as exc:
                        _posthoc_log(
                            outdir,
                            f"[POSTHOC][WARN] FC backprojection failed @ H={int(H_fc)}: {type(exc).__name__}: {exc}",
                        )

                if df_branch is not None:
                    risk_map = dict(
                        zip(
                            df_fc["feature"].astype(str),
                            pd.to_numeric(df_fc["FC_score"], errors="coerce"),
                        )
                    )
                    delta_map = dict(
                        zip(
                            df_fc["feature"].astype(str),
                            pd.to_numeric(df_fc["FC_delta_mean_nlS"], errors="coerce"),
                        )
                    )
                    df_branch_h = df_branch.copy()
                    df_branch_h["risk_score"] = pd.to_numeric(
                        df_branch_h["feature"].map(risk_map), errors="coerce"
                    )
                    df_branch_h["delta_risk_overall"] = pd.to_numeric(
                        df_branch_h["feature"].map(delta_map), errors="coerce"
                    )
                    df_branch_h.to_csv(
                        outdir / f"branch_feature_collapsing_test_H{int(H_fc)}.csv",
                        index=False,
                    )
                    risk_xlim_fc = _positive_xlim_from_values(df_branch_h["risk_score"])
                    delta_xlim_fc = _signed_xlim_from_values(
                        df_branch_h["delta_risk_overall"]
                    )
                    shared_xlim_fc = _positive_xlim_from_values(
                        df_branch_h["shared_latent_fc"]
                    )
                    private_xlim_cols = ["private_latent_fc"] + [
                        c
                        for c in df_branch_h.columns
                        if str(c).startswith("private_latent_fc_group")
                    ]
                    private_xlim_fc = _positive_xlim_from_values(
                        [
                            df_branch_h[c]
                            for c in private_xlim_cols
                            if c in df_branch_h.columns
                        ]
                    )
                    _plot_fc_triptych(
                        df_branch_h,
                        "shared_latent_fc",
                        "delta_risk_overall",
                        f"Test feature perturbation to shared latent component @ H={int(H_fc)} mo",
                        "Shared latent component perturbation",
                        outdir / f"branch_fc_shared_test_H{int(H_fc)}.png",
                        branch_color="#355C7D",
                        sort_col="shared_latent_fc",
                        risk_xlim=risk_xlim_fc,
                        branch_xlim=shared_xlim_fc,
                        delta_xlim=delta_xlim_fc,
                    )
                    _plot_fc_triptych(
                        df_branch_h,
                        "private_latent_fc",
                        "delta_risk_overall",
                        f"Test feature perturbation to group-specific latent component @ H={int(H_fc)} mo",
                        "Group-specific latent component perturbation",
                        outdir / f"branch_fc_private_test_H{int(H_fc)}.png",
                        branch_color="#7A4E7E",
                        sort_col="private_latent_fc",
                        risk_xlim=risk_xlim_fc,
                        branch_xlim=private_xlim_fc,
                        delta_xlim=delta_xlim_fc,
                    )
                    for gi, gname in enumerate(groups):
                        private_col = f"private_latent_fc_group{int(gi):02d}"
                        if private_col not in df_branch_h.columns:
                            continue
                        safe_g = _safe_fname(gname)
                        dfg_private = df_branch_h[
                            ["feature", "risk_score", private_col, "delta_risk_overall"]
                        ].copy()
                        dfg_private = dfg_private.rename(
                            columns={private_col: "private_latent_fc_group"}
                        )
                        dfg_private = dfg_private.sort_values(
                            "private_latent_fc_group", ascending=False
                        ).reset_index(drop=True)
                        dfg_private.to_csv(
                            outdir
                            / f"branch_feature_collapsing_private_{safe_g}_test_H{int(H_fc)}.csv",
                            index=False,
                        )
                        _plot_fc_triptych(
                            dfg_private,
                            "private_latent_fc_group",
                            "delta_risk_overall",
                            f"{gname}: feature perturbation to group-specific latent component @ H={int(H_fc)} mo",
                            f"{gname} group-specific latent component perturbation",
                            outdir
                            / f"branch_fc_private_{safe_g}_test_H{int(H_fc)}.png",
                            branch_color="#7A4E7E",
                            sort_col="private_latent_fc_group",
                            risk_xlim=risk_xlim_fc,
                            branch_xlim=private_xlim_fc,
                            delta_xlim=delta_xlim_fc,
                        )
            except Exception as exc:
                _posthoc_log(
                    outdir,
                    f"[POSTHOC][WARN] Branch/overall FC support block failed @ H={int(H_fc)}: {type(exc).__name__}: {exc}",
                )

        if len(pc_fc_summary_rows):
            (
                pd.DataFrame(pc_fc_summary_rows)
                .sort_values(["scope", "H", "fc_score"], ascending=[True, True, False])
                .to_csv(outdir / "topPC_FC_multiH_scopes_summary.csv", index=False)
            )
        if len(pc_fc_gene_rows):
            pd.concat(pc_fc_gene_rows, axis=0, ignore_index=True).to_csv(
                outdir / "pc_to_gene_backprojection_FCtop.csv", index=False
            )


        if ENABLE_DGP_FC_GENE_EXPRESSION_PSEUDOTIME:
            try:
                _posthoc_log(
                    outdir,
                    "[POSTHOC] DGP-FC prioritized raw-expression pseudotime",
                )
                run_dgp_fc_gene_expression_pseudotime_analysis(art, outdir)
            except Exception as exc:
                _posthoc_log(
                    outdir,
                    f"[POSTHOC][WARN] DGP-FC gene-expression pseudotime block failed: {type(exc).__name__}: {exc}",
                )
                _write_posthoc_traceback(outdir, "dgp_fc_gene_expression_pseudotime_exception", exc)

        try:
            _posthoc_log(
                outdir,
                "[POSTHOC] Inferred resistance-state landscapes from trained DGP latents",
            )
            run_resistance_landscape_analysis(art, outdir)
        except Exception as exc:
            _posthoc_log(
                outdir,
                f"[POSTHOC][WARN] Resistance landscape block failed: {type(exc).__name__}: {exc}",
            )
            _write_posthoc_traceback(outdir, "resistance_landscape_exception", exc)

        _posthoc_log(outdir, "[POSTHOC] Completed posthoc analysis")
    except Exception as exc:
        _write_posthoc_traceback(outdir, exc)
        raise



def _feature_strata_for_treatment_contrast(
    train_values,
    test_values,
    family,
    min_n=FEATURE_STRATIFIED_TX_MIN_N,
    include_overall=FEATURE_STRATIFIED_TX_INCLUDE_OVERALL_STRATUM,
):
    """Return feature strata for treatment-risk contrast summaries.

    Thresholds are learned from the training split within the same analysis
    scope (pooled or cancer-specific) and then applied to the held-out test
    split. This avoids defining low/high pathway strata from the test set or
    from a pooled distribution when the analysis is cancer-specific.

    Continuous/pathway variables use lower and upper training-tertile tails.
    Binary clinical indicators use value=0 versus value=1. There is no median
    split fallback for continuous/pathway features; if tertile cutpoints collapse,
    the low/high rows are skipped rather than replacing tail strata with a blunt
    median split.
    """
    vtr = np.asarray(train_values, dtype=float)
    vte = np.asarray(test_values, dtype=float)
    finite_tr = np.isfinite(vtr)
    finite_te = np.isfinite(vte)
    min_n = int(min_n)
    strata = []

    if bool(include_overall) and int(np.sum(finite_te)) >= min_n:
        strata.append((
            "overall",
            finite_te.copy(),
            "overall: all finite held-out patients in this analysis scope",
        ))

    if int(np.sum(finite_tr)) == 0 or int(np.sum(finite_te)) == 0:
        return strata

    train_unique = np.unique(vtr[finite_tr])
    if train_unique.size <= 2 and np.all(np.isin(train_unique, [0.0, 1.0])):
        for val, lab in [(0.0, "feature0"), (1.0, "feature1")]:
            m = finite_te & np.isclose(vte, val)
            if int(np.sum(m)) >= min_n:
                strata.append((
                    lab,
                    m,
                    f"{lab}: held-out value={int(val)}; cutpoint source=train-scope binary values",
                ))
        return strata

    vf = vtr[finite_tr]
    if int(vf.size) < 2 * min_n:
        return strata
    q_low, q_high = np.nanquantile(vf, [1.0 / 3.0, 2.0 / 3.0])
    if (
        not np.isfinite(q_low)
        or not np.isfinite(q_high)
        or float(q_low) >= float(q_high)
    ):
        return strata

    low = finite_te & (vte <= q_low)
    high = finite_te & (vte >= q_high)
    if int(np.sum(low)) >= min_n:
        strata.append((
            "low",
            low,
            f"low: held-out value <= train-scope tertile-1 cutpoint {q_low:.6g}",
        ))
    if int(np.sum(high)) >= min_n:
        strata.append((
            "high",
            high,
            f"high: held-out value >= train-scope tertile-2 cutpoint {q_high:.6g}",
        ))
    return strata


def _should_run_feature_stratified_tx_contrasts_for_seed(seed):
    """Return True when the expensive treatment-risk contrast block should run."""
    if not ENABLE_REPEATED_SPLIT_FEATURE_STRATIFIED_TX_CONTRASTS:
        return False
    seeds_cfg = [int(s) for s in FEATURE_STRATIFIED_TX_CONTRAST_SEEDS]
    return (-1 in seeds_cfg) or (int(seed) in set(seeds_cfg))

def _summarize_sample_vector(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "sd": np.nan, "median": np.nan, "q05": np.nan,
                "q25": np.nan, "q75": np.nan, "q95": np.nan,
                "sign_probability_positive": np.nan, "sign_probability_negative": np.nan,
                "direction_sign": 0}
    m = float(np.mean(x))
    eps = float(FEATURE_STRATIFIED_TX_EPS)
    return {
        "mean": m,
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "sign_probability_positive": float(np.mean(x > 0.0)),
        "sign_probability_negative": float(np.mean(x < 0.0)),
        "direction_sign": int(1 if m > eps else (-1 if m < -eps else 0)),
    }



def _tx_profile_label_from_vector(vec, tx_cols):
    vec = np.asarray(vec, dtype=int).reshape(-1)
    short_cols = [_treatment_label(c) for c in tx_cols]
    active = [short_cols[j] for j, v in enumerate(vec) if int(v) == 1]
    return "TX=None" if len(active) == 0 else "TX=" + "+".join(active)


def _counterfactual_exact_profile_tensor(art, tx_cols, profile_vec):
    feat_names = [str(x) for x in art["feat_names"]]
    fmap = _feature_index_map(feat_names)
    Xraw = np.asarray(art["Xte_np_sub"], float).copy()
    included_bases = {_base_treatment_col(c) for c in tx_cols}
    # If a treatment flag is intentionally excluded from the exact-profile
    # definition for this scope, set it to 0 rather than leaving observed values
    # hidden inside a profile label such as TX=None. This is most important for
    # excluding hormone from pooled/shared, Lung, and Colon treatment contrasts.
    for base in TREATMENT_FLAG_COLS:
        if base in included_bases:
            continue
        for cand in [base, f"clinbin::{base}"]:
            if cand in fmap:
                Xraw[:, fmap[cand]] = 0.0
                break
    for base_col, v in zip(tx_cols, np.asarray(profile_vec, dtype=int).reshape(-1)):
        base = _base_treatment_col(base_col)
        candidates = [str(base_col), base, f"clinbin::{base}"]
        hit = next((c for c in candidates if c in fmap), None)
        if hit is not None:
            Xraw[:, fmap[hit]] = float(v)
    return _as_torch_from_raw_standardized(Xraw, art["standardize_mu"], art["standardize_sd"])


def compute_feature_stratified_treatment_risk_contrasts(art, seed=None, cfg_tag=None):
    """All pairwise exact treatment-profile risk contrasts within feature strata.

    This is the corrected version: it is NOT only treatment-vs-no-treatment.
    It compares every observed exact treatment profile against every other observed
    exact treatment profile, then attaches DGP branch calls downstream.
    """
    if not ENABLE_REPEATED_SPLIT_FEATURE_STRATIFIED_TX_CONTRASTS:
        return pd.DataFrame()
    model = art["model"]
    alpha_dt = art["alpha_dt"]
    dt_edges = art["dt_edges"]
    g_te_t = art["g_te_t"]
    g_te = np.asarray(art["g_te"], dtype=int)
    groups = [str(x) for x in art["groups"]]
    feat_names = [str(x) for x in art["feat_names"]]
    Xraw = np.asarray(art["Xte_np_sub"], dtype=float)
    Xraw_tr = np.asarray(art["Xtr_np_sub"], dtype=float)
    g_tr = np.asarray(art["g_tr"], dtype=int)
    meta_te = art.get("meta_te")
    meta_tr = art.get("meta_tr")

    A_tx, tx_cols = _treatment_profile_from_metadata_or_design(meta_te, Xraw, feat_names)
    A_tx_tr, tx_cols_tr = _treatment_profile_from_metadata_or_design(meta_tr, Xraw_tr, feat_names)
    if (A_tx is None or A_tx.ndim != 2 or A_tx.shape[1] == 0 or
            A_tx_tr is None or A_tx_tr.ndim != 2 or A_tx_tr.shape[1] == 0):
        return pd.DataFrame()

    # Treatment profile vectors are scope-specific: pooled/shared, Lung, and
    # Colon exclude hormone; Breast-specific summaries retain hormone.
    tx_feature_cols = set()
    for c in tx_cols:
        base = _base_treatment_col(c)
        tx_feature_cols.update([str(c), base, f"clinbin::{base}"])

    allowed_families = {str(x) for x in FEATURE_STRATIFIED_TX_FAMILIES}
    eligible = []
    for j, f in enumerate(feat_names):
        fam = _feature_family_for_mechanism(f)
        if fam not in allowed_families:
            continue
        if fam == "treatment" or str(f) in tx_feature_cols:
            continue
        eligible.append((int(j), str(f), str(fam)))
    if len(eligible) == 0:
        return pd.DataFrame()

    # Keep scope for reporting, but DGP shared/private comes from dgp_branch_call,
    # not from pooled scope.
    scope_masks = [(
        "pooled",
        "All_cancers_combined",
        np.ones(Xraw.shape[0], dtype=bool),
        np.ones(Xraw_tr.shape[0], dtype=bool),
    )]
    for gi, gname in enumerate(groups):
        scope_masks.append(("group", str(gname), g_te == int(gi), g_tr == int(gi)))

    rows = []
    draw_summaries = []
    model.eval()
    for scope, group_label, scope_mask, train_scope_mask in scope_masks:
        if int(np.sum(scope_mask)) < int(FEATURE_STRATIFIED_TX_MIN_N):
            continue
        if int(np.sum(train_scope_mask)) < int(FEATURE_STRATIFIED_TX_MIN_N):
            continue
        A_scope_tr, tx_cols_scope = _treatment_profile_scope_projection(
            A_tx_tr, tx_cols_tr, scope=scope, group=group_label
        )
        if A_scope_tr is None or A_scope_tr.ndim != 2 or A_scope_tr.shape[1] == 0:
            continue
        profile_columns = [_base_treatment_col(c) for c in tx_cols_scope]
        prof_df = pd.DataFrame(A_scope_tr.astype(int), columns=profile_columns)
        prof_df["profile_key"] = prof_df.astype(str).agg("|".join, axis=1)
        prof_train = prof_df.loc[train_scope_mask].copy()
        prof_train["PFI_EVENT"] = np.asarray(art["e_tr"], int)[train_scope_mask]
        profile_support = prof_train.groupby("profile_key", dropna=False).agg(
            training_patients=("profile_key", "size"), training_events=("PFI_EVENT", "sum")
        ).reset_index()
        supported_keys = set(profile_support.loc[
            (profile_support["training_patients"] >= int(FEATURE_STRATIFIED_TX_MIN_N))
            & (profile_support["training_events"] >= 3), "profile_key"
        ].astype(str))
        profile_rows = prof_train[prof_train["profile_key"].isin(supported_keys)].drop_duplicates(
            "profile_key"
        )[profile_columns].to_numpy(dtype=int)
        profile_labels = [_tx_profile_label_from_vector(v, tx_cols_scope) for v in profile_rows]
        profile_support_map = {}
        for vec, label in zip(profile_rows, profile_labels):
            key = "|".join(map(str, np.asarray(vec, int).tolist()))
            support_row = profile_support[profile_support["profile_key"].astype(str).eq(key)]
            if len(support_row):
                profile_support_map[label] = support_row.iloc[0].to_dict()
        if len(profile_rows) < 2:
            continue
        risk_by_H_profile = {}
        for H in [float(h) for h in FEATURE_STRATIFIED_TX_H_LIST]:
            risk_by_H_profile[H] = {}
            for label, vec in zip(profile_labels, profile_rows):
                Xt = _counterfactual_exact_profile_tensor(art, tx_cols_scope, vec)
                S_H = posterior_survival_at_H(
                    model, Xt, g_te_t, H, alpha_dt, dt_edges, S=int(FEATURE_STRATIFIED_TX_MC)
                )
                risk_by_H_profile[H][label] = 1.0 - np.asarray(S_H, dtype=float)
        # One prespecified orientation per unordered pair; reversed duplicates
        # carry no additional information and would double the testing family.
        pair_list = [
            (profile_labels[i], profile_labels[j])
            for i in range(len(profile_labels))
            for j in range(i + 1, len(profile_labels))
        ]
        for H, risk_map in risk_by_H_profile.items():
            for label_a, label_b in pair_list:
                rd = risk_map[label_a] - risk_map[label_b]
                for j, feat, fam in eligible:
                    for stratum_id, stratum_mask, stratum_label in _feature_strata_for_treatment_contrast(
                        Xraw_tr[train_scope_mask, j],
                        Xraw[:, j],
                        fam,
                        min_n=int(FEATURE_STRATIFIED_TX_MIN_N),
                    ):
                        m = scope_mask & stratum_mask
                        if int(np.sum(m)) < int(FEATURE_STRATIFIED_TX_MIN_N):
                            continue
                        # Preserve posterior dependence: aggregate eligible patients
                        # within each draw, then summarize uncertainty across draws.
                        vals = np.nanmean(rd[:, m], axis=1)
                        summ = _summarize_sample_vector(vals)
                        row = {
                            "row_kind": "pairwise_exact_profile_stratum",
                            "seed": int(seed) if seed is not None else np.nan,
                            "analysis_design": "repeated_outer_split_paired_treatment_profile_contrast",
                            "prespecified_representative_split": False,
                            "repeated_split_stability_estimable": False,
                            "config_tag": str(cfg_tag) if cfg_tag is not None else "",
                            "H": float(H),
                            "scope": str(scope),
                            "group": str(group_label),
                            "contrast_type": "all_pairwise_exact_treatment_profiles",
                            "reference_profile": str(label_b),
                            "treatment_profile": str(label_a),
                            "treatment_profile_A": str(label_a),
                            "treatment_profile_B": str(label_b),
                            "contrast_label": f"{label_a} vs {label_b}",
                            "training_patients_profile_A": int(profile_support_map[label_a]["training_patients"]),
                            "training_events_profile_A": int(profile_support_map[label_a]["training_events"]),
                            "training_patients_profile_B": int(profile_support_map[label_b]["training_patients"]),
                            "training_events_profile_B": int(profile_support_map[label_b]["training_events"]),
                            "treatment_columns_for_scope": ";".join(tx_cols_scope),
                            "effect_definition": "P(event by H | profile A) - P(event by H | profile B), within feature stratum",
                            "feature": str(feat),
                            "feature_family": str(fam),
                            "feature_scope": _feature_scope_for_mechanism(feat),
                            "is_shared_across_cancers_feature": bool(_is_shared_across_cancers_feature(feat)),
                            "is_cancer_specific_feature": bool(_feature_scope_for_mechanism(feat) == "cancer_specific"),
                            "stratum_id": str(stratum_id),
                            "stratum_label": str(stratum_label),
                            "n_patients_scope_stratum": int(np.sum(m)),
                            "effect_mean": summ["mean"],
                            "effect_sd": summ["sd"],
                            "effect_median": summ["median"],
                            "effect_q05": summ["q05"],
                            "effect_q25": summ["q25"],
                            "effect_q75": summ["q75"],
                            "effect_q95": summ["q95"],
                            "sign_probability_positive": summ["sign_probability_positive"],
                            "sign_probability_negative": summ["sign_probability_negative"],
                            "effect_mean_pp": 100.0 * summ["mean"] if np.isfinite(summ["mean"]) else np.nan,
                            "effect_q05_pp": 100.0 * summ["q05"] if np.isfinite(summ["q05"]) else np.nan,
                            "effect_q95_pp": 100.0 * summ["q95"] if np.isfinite(summ["q95"]) else np.nan,
                            "direction_sign": summ["direction_sign"],
                            "posterior_mc_samples": int(FEATURE_STRATIFIED_TX_MC),
                            "is_profile_vs_none": bool((label_a == "TX=None") or (label_b == "TX=None")),
                            "interpretation": "all-pairwise exact-treatment-profile effect-modification screen using train-scope low/high cutpoints applied to held-out test patients",
                            "stratum_cutpoint_policy": "overall plus train-scope lower/upper tertile tails; no test-defined or median-split continuous cutpoints",
                        }
                        rows.append(row)
                        draw_summaries.append({"row": row, "draw_values": np.asarray(vals, float)})
    by_interaction = {}
    for item in draw_summaries:
        row = item["row"]
        if row.get("stratum_id") not in {"low", "high"}:
            continue
        key = tuple(row.get(c) for c in [
            "seed", "config_tag", "H", "scope", "group", "reference_profile",
            "treatment_profile", "feature", "feature_family",
        ])
        by_interaction.setdefault(key, {})[row["stratum_id"]] = item
    for strata in by_interaction.values():
        if "low" not in strata or "high" not in strata:
            continue
        dd = strata["high"]["draw_values"] - strata["low"]["draw_values"]
        summ = _summarize_sample_vector(dd)
        row = dict(strata["high"]["row"])
        row.update({
            "row_kind": "delta_delta_feature_treatment_interaction",
            "contrast_type": "delta_delta_high_minus_low",
            "stratum_id": "delta_delta_high_minus_low",
            "stratum_label": "(profile A - profile B)_high - (profile A - profile B)_low",
            "interaction_estimand": "DeltaDelta=(A-B)_high_feature-(A-B)_low_feature",
            "n_patients_scope_stratum": min(
                int(strata["low"]["row"]["n_patients_scope_stratum"]),
                int(strata["high"]["row"]["n_patients_scope_stratum"]),
            ),
            "effect_mean": summ["mean"], "effect_sd": summ["sd"],
            "effect_median": summ["median"], "effect_q05": summ["q05"],
            "effect_q25": summ["q25"], "effect_q75": summ["q75"], "effect_q95": summ["q95"],
            "sign_probability_positive": summ["sign_probability_positive"],
            "sign_probability_negative": summ["sign_probability_negative"],
            "effect_mean_pp": 100.0 * summ["mean"], "effect_q05_pp": 100.0 * summ["q05"],
            "effect_q95_pp": 100.0 * summ["q95"], "direction_sign": summ["direction_sign"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_feature_stratified_treatment_direction_stability(df):
    """Aggregate pairwise treatment-profile contrast directions across seeds."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    work = df.copy()
    work["effect_mean"] = pd.to_numeric(work["effect_mean"], errors="coerce")
    work["direction_sign"] = pd.to_numeric(work["direction_sign"], errors="coerce").fillna(0).astype(int)
    group_cols = [
        "config_tag", "H", "scope", "group", "contrast_type",
        "reference_profile", "treatment_profile", "contrast_label",
        "feature", "feature_family", "stratum_id", "stratum_label",
        "dgp_branch_call", "max_private_group",
    ]
    for c in group_cols:
        if c not in work.columns:
            work[c] = ""
    rows = []
    for keys, sub in work.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(sub["effect_mean"], errors="coerce").dropna().to_numpy(float)
        dirs = pd.to_numeric(sub["direction_sign"], errors="coerce").fillna(0).astype(int).to_numpy()
        dirs = dirs[dirs != 0]
        if vals.size == 0:
            continue
        n_splits = int(sub["seed"].nunique()) if "seed" in sub.columns else int(len(sub))
        pos = int(np.sum(dirs > 0))
        neg = int(np.sum(dirs < 0))
        dominant = 1 if pos >= neg else -1
        dominant_n = max(pos, neg)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_splits_with_estimate": int(n_splits),
            "repeated_split_stability_estimable": bool(n_splits >= 2),
            "effect_mean_across_splits": float(np.mean(vals)),
            "effect_sd_across_splits": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            "effect_mean_pp_across_splits": float(100.0 * np.mean(vals)),
            "direction_positive_splits": pos,
            "direction_negative_splits": neg,
            "dominant_direction": int(dominant),
            "direction_stability": float(dominant_n / n_splits) if n_splits >= 2 else np.nan,
            "signed_stability_score": float(dominant * dominant_n / n_splits) if n_splits >= 2 else np.nan,
            "abs_mean_effect": float(abs(np.mean(vals))),
        })
        rows.append(row)
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["direction_stability", "abs_mean_effect", "n_splits_with_estimate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _short_feature_label_for_plot(feature):
    s = str(feature)
    for prefix in ["act_shared::", "udp_shared::", "mut_shared::", "act::", "udp::", "mut::", "clincont::", "clinbin::"]:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def save_joined_top_wfc_treatment_contrast_outputs(fc_df, tx_df, cfg_dir, cfg_tag):
    """Join WFC-ranked shared pathways to treatment-risk contrast rows.

    This creates the final posthoc table requested for interpretation: among the
    top WFC-ranked shared pathway features, how do exact treatment-profile
    changes alter predicted risk overall and separately in low/high pathway
    activity strata?
    """
    cfg_dir = Path(cfg_dir)
    if not JOIN_TOP_WFC_TX_CONTRASTS:
        return pd.DataFrame()
    if fc_df is None or tx_df is None or len(fc_df) == 0 or len(tx_df) == 0:
        pd.DataFrame({"message": ["Missing WFC rows or treatment-risk contrast rows; joined table was not produced."]}).to_csv(
            cfg_dir / f"{cfg_tag}_joined_topWFC_txRiskContrasts_skipped.csv", index=False
        )
        return pd.DataFrame()

    fc = pd.DataFrame(fc_df).copy()
    tx = pd.DataFrame(tx_df).copy()
    # Harmonize the scope labels produced by the FC and paired-treatment blocks.
    # Without this normalization, pooled FC used group="pooled" while paired
    # contrasts used group="All_cancers_combined", so a valid join returned zero
    # rows and was incorrectly written as skipped.
    if "scope" in fc.columns:
        fc["fc_scope_original"] = fc["scope"].astype(str)
        fc["scope"] = fc["scope"].astype(str).replace({
            "group_specific": "group",
            "pooled_exact_treatment_profile": "pooled",
            "group_exact_treatment_profile": "group",
        })
    if "group" in fc.columns:
        fc["fc_group_original"] = fc["group"].astype(str)
        pooled_fc = fc.get("scope", pd.Series("", index=fc.index)).astype(str).eq("pooled")
        fc.loc[pooled_fc & fc["group"].astype(str).eq("pooled"), "group"] = "All_cancers_combined"
    if "scope" in tx.columns:
        tx["tx_scope_original"] = tx["scope"].astype(str)
    if "group" in tx.columns:
        tx["tx_group_original"] = tx["group"].astype(str)
        pooled_tx = tx.get("scope", pd.Series("", index=tx.index)).astype(str).eq("pooled")
        tx.loc[pooled_tx & tx["group"].astype(str).eq("pooled"), "group"] = "All_cancers_combined"
    if "feature" not in fc.columns or "feature" not in tx.columns:
        pd.DataFrame({"message": ["Both WFC and treatment-contrast tables must contain a feature column."]}).to_csv(
            cfg_dir / f"{cfg_tag}_joined_topWFC_txRiskContrasts_skipped.csv", index=False
        )
        return pd.DataFrame()
    if "feature_family" not in fc.columns or "feature_scope" not in fc.columns or "is_shared_across_cancers_feature" not in fc.columns:
        fc = _add_fc_feature_family_columns(fc)
    if "feature_family" not in tx.columns:
        tx["feature_family"] = tx["feature"].map(_feature_family_for_mechanism)
    if "feature_scope" not in tx.columns:
        tx["feature_scope"] = tx["feature"].map(_feature_scope_for_mechanism)
    if "is_shared_across_cancers_feature" not in tx.columns:
        tx["is_shared_across_cancers_feature"] = tx["feature"].map(_is_shared_across_cancers_feature)

    families = {str(x) for x in JOIN_TOP_WFC_TX_FAMILIES}
    fc = fc[fc["feature_family"].astype(str).isin(families)].copy()
    tx = tx[tx["feature_family"].astype(str).isin(families)].copy()
    if bool(JOIN_TOP_WFC_TX_SHARED_ONLY):
        fc = fc[fc["is_shared_across_cancers_feature"].astype(bool)].copy()
        tx = tx[tx["is_shared_across_cancers_feature"].astype(bool)].copy()

    if "FC_ranked_feature" in fc.columns:
        fc = fc[fc["FC_ranked_feature"].astype(bool)].copy()
    if "FC_score" not in fc.columns:
        pd.DataFrame({"message": ["WFC table has no FC_score column."]}).to_csv(
            cfg_dir / f"{cfg_tag}_joined_topWFC_txRiskContrasts_skipped.csv", index=False
        )
        return pd.DataFrame()
    fc["FC_score"] = pd.to_numeric(fc["FC_score"], errors="coerce")
    fc = fc[np.isfinite(fc["FC_score"].to_numpy(dtype=float))].copy()
    if "FC_rank_within_seed_scope" not in fc.columns:
        rank_cols = [c for c in ["seed", "H", "scope", "group", "treatment_profile"] if c in fc.columns]
        fc["FC_rank_within_seed_scope"] = fc.groupby(rank_cols, dropna=False)["FC_score"].rank(method="min", ascending=False) if rank_cols else fc["FC_score"].rank(method="min", ascending=False)
    fc["FC_rank_within_seed_scope"] = pd.to_numeric(fc["FC_rank_within_seed_scope"], errors="coerce")
    fc_top = fc[fc["FC_rank_within_seed_scope"] <= float(JOIN_TOP_WFC_TX_TOPN)].copy()
    if len(fc_top) == 0:
        pd.DataFrame({"message": ["No top WFC-ranked shared pathway features were available for the join."]}).to_csv(
            cfg_dir / f"{cfg_tag}_joined_topWFC_txRiskContrasts_skipped.csv", index=False
        )
        return pd.DataFrame()

    key_cols = [c for c in ["seed", "config_tag", "H", "scope", "group", "feature"] if c in fc_top.columns and c in tx.columns]
    fc_keep = [
        c for c in [
            "seed", "config_tag", "H", "scope", "group", "feature", "feature_family", "feature_scope",
            "is_shared_across_cancers_feature", "treatment_profile", "treatment_active_count",
            "is_multi_treatment_profile", "FC_score", "FC_rank_within_seed_scope",
            "FC_rank_percentile_within_seed_scope",
        ] if c in fc_top.columns
    ]
    fcj = fc_top[fc_keep].copy()
    fcj = fcj.rename(columns={
        "treatment_profile": "wfc_context_treatment_profile",
        "treatment_active_count": "wfc_context_treatment_active_count",
        "is_multi_treatment_profile": "wfc_context_is_multi_treatment_profile",
        "FC_score": "wfc_score",
        "FC_rank_within_seed_scope": "wfc_rank_within_seed_scope",
        "FC_rank_percentile_within_seed_scope": "wfc_rank_percentile_within_seed_scope",
    })
    merged = tx.merge(fcj, on=key_cols, how="inner", suffixes=("_tx", "_wfc"))
    if (
        len(merged)
        and "wfc_context_treatment_profile" in merged.columns
        and "treatment_profile_A" in merged.columns
        and "treatment_profile_B" in merged.columns
    ):
        ctx = merged["wfc_context_treatment_profile"]
        no_exact_context = ctx.isna() | ctx.astype(str).isin({"", "nan", "None"})
        context_matches_contrast = (
            ctx.astype(str).eq(merged["treatment_profile_A"].astype(str))
            | ctx.astype(str).eq(merged["treatment_profile_B"].astype(str))
        )
        merged = merged[no_exact_context | context_matches_contrast].copy()
    if len(merged) == 0:
        pd.DataFrame({"message": ["Top WFC-ranked shared pathways did not overlap treatment-risk contrast rows."]}).to_csv(
            cfg_dir / f"{cfg_tag}_joined_topWFC_txRiskContrasts_skipped.csv", index=False
        )
        return pd.DataFrame()
    sort_cols = [c for c in ["seed", "H", "scope", "group", "wfc_rank_within_seed_scope", "feature", "contrast_label", "stratum_id"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, ascending=True).reset_index(drop=True)
    merged.to_csv(cfg_dir / f"{cfg_tag}_joined_topWFC_sharedPathway_txRiskContrasts.csv", index=False)

    compact = merged.copy()
    if "is_profile_vs_none" in compact.columns:
        compact_profile = compact[compact["is_profile_vs_none"].astype(bool)].copy()
        if len(compact_profile):
            compact = compact_profile
    if "stratum_id" in compact.columns:
        compact = compact[compact["stratum_id"].astype(str).isin(["overall", "low", "high"])].copy()
    compact.to_csv(cfg_dir / f"{cfg_tag}_joined_topWFC_sharedPathway_txRiskContrasts_compact.csv", index=False)

    try:
        _plot_joined_top_wfc_tx_contrast_heatmaps(compact, cfg_dir, cfg_tag)
    except Exception as exc:
        pd.DataFrame({"plot_error": [f"{type(exc).__name__}: {exc}"]}).to_csv(
            cfg_dir / f"{cfg_tag}_joined_topWFC_txRiskContrasts_plot_error.csv", index=False
        )
    return merged


def _plot_joined_top_wfc_tx_contrast_heatmaps(df, cfg_dir, cfg_tag):
    if df is None or len(df) == 0:
        return
    need = {"seed", "H", "scope", "group", "feature", "contrast_label", "stratum_id", "effect_mean_pp"}
    if not need.issubset(set(df.columns)):
        return
    work = pd.DataFrame(df).copy()
    work["effect_mean_pp"] = pd.to_numeric(work["effect_mean_pp"], errors="coerce")
    work["wfc_rank_within_seed_scope"] = pd.to_numeric(work.get("wfc_rank_within_seed_scope", np.nan), errors="coerce")
    work = work[np.isfinite(work["effect_mean_pp"].to_numpy(dtype=float))].copy()
    work = work[(work["scope"].astype(str) == "pooled") & (work["group"].astype(str) == "All_cancers_combined")].copy()
    if bool(JOIN_TOP_WFC_TX_PROFILE_VS_NONE_ONLY_FOR_PLOTS) and "is_profile_vs_none" in work.columns:
        prof = work[work["is_profile_vs_none"].astype(bool)].copy()
        if len(prof):
            work = prof
    if len(work) == 0:
        return
    for (seed, H), sub in work.groupby(["seed", "H"], dropna=False):
        sub = sub.copy()
        feat_order = (
            sub.groupby("feature", dropna=False)["wfc_rank_within_seed_scope"]
            .min()
            .sort_values(ascending=True)
            .head(int(JOIN_TOP_WFC_TX_PLOT_TOPN))
            .index.tolist()
        )
        if len(feat_order) == 0:
            feat_order = sub["feature"].drop_duplicates().head(int(JOIN_TOP_WFC_TX_PLOT_TOPN)).tolist()
        sub = sub[sub["feature"].isin(feat_order)].copy()
        sub["feature_plot"] = sub["feature"].map(_short_feature_label_for_plot)
        sub["contrast_stratum"] = sub["contrast_label"].astype(str) + " | " + sub["stratum_id"].astype(str)
        col_score = sub.groupby("contrast_stratum")["effect_mean_pp"].apply(lambda x: float(np.nanmax(np.abs(x))))
        keep_cols = col_score.sort_values(ascending=False).head(18).index.tolist()
        sub = sub[sub["contrast_stratum"].isin(keep_cols)].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="feature_plot",
            columns="contrast_stratum",
            values="effect_mean_pp",
            aggfunc="mean",
        )
        row_labels = [_short_feature_label_for_plot(f) for f in feat_order]
        pivot = pivot.reindex([r for r in row_labels if r in pivot.index])
        pivot = pivot.reindex(columns=keep_cols)
        if pivot.shape[0] == 0 or pivot.shape[1] == 0:
            continue
        vals = pivot.to_numpy(dtype=float)
        vmax = np.nanmax(np.abs(vals)) if np.any(np.isfinite(vals)) else 1.0
        vmax = max(float(vmax), 1e-6)
        fig_w = max(9.5, 0.55 * pivot.shape[1] + 4.0)
        fig_h = max(5.0, 0.38 * pivot.shape[0] + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(list(pivot.columns), rotation=55, ha="right", fontsize=8)
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(list(pivot.index), fontsize=8.5)
        ax.set_xlabel("Treatment-profile contrast and pathway stratum")
        ax.set_ylabel("Top WFC-ranked shared pathway")
        ax.set_title(f"Top shared WFC pathways: treatment-risk shifts, seed {int(seed)}, H={int(float(H))} mo")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = vals[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
        cb.set_label("Predicted risk shift (percentage points)")
        fig.tight_layout()
        savefig(cfg_dir / f"{cfg_tag}_joined_topWFC_sharedPathway_txRiskContrast_heatmap_seed{int(seed)}_H{int(float(H))}.png")


def save_feature_stratified_treatment_contrast_outputs(rows, cfg_dir, cfg_tag):
    """Save all-pairwise treatment-profile contrasts for the current split(s)."""
    cfg_dir = Path(cfg_dir)
    if rows is None or len(rows) == 0:
        pd.DataFrame({"message": ["No pairwise feature-stratified treatment contrast rows were produced."]}).to_csv(
            cfg_dir / f"{cfg_tag}_pairwise_feature_tx_contrasts_skipped.csv", index=False
        )
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(rows, axis=0, ignore_index=True) if isinstance(rows, list) else pd.DataFrame(rows)
    df.to_csv(cfg_dir / f"{cfg_tag}_pairwise_feature_tx_contrasts_by_split.csv", index=False)
    if ENABLE_REPEATED_SPLIT_DGP_BRANCH_ATTRIBUTION:
        df.to_csv(cfg_dir / f"{cfg_tag}_pairwise_feature_tx_contrasts_by_split_WITH_DGP_BRANCH_CALLS.csv", index=False)
    n_contrast_seeds = int(df["seed"].nunique()) if "seed" in df else 0
    # Never label a representative single-split analysis as an all-ten-split result.
    if n_contrast_seeds == 10:
        df.to_csv(cfg_dir / f"{cfg_tag}_txContrasts_all10_by_split.csv", index=False)
    df.to_csv(cfg_dir / f"{cfg_tag}_feature_tx_contrasts_by_split.csv", index=False)
    stability = aggregate_feature_stratified_treatment_direction_stability(df)
    stability.to_csv(cfg_dir / f"{cfg_tag}_pairwise_feature_tx_direction_stability.csv", index=False)
    stability.to_csv(cfg_dir / f"{cfg_tag}_txContrasts_direction_stability.csv", index=False)
    stability.to_csv(cfg_dir / f"{cfg_tag}_feature_tx_direction_stability.csv", index=False)
    if len(stability):
        top = stability[
            (pd.to_numeric(stability["direction_stability"], errors="coerce") >= 0.80)
            & (pd.to_numeric(stability["n_splits_with_estimate"], errors="coerce") >= 2)
        ].copy()
        top.to_csv(cfg_dir / f"{cfg_tag}_pairwise_feature_tx_direction_stability_top.csv", index=False)
    return df, stability


def save_repeated_split_treatment_adjusted_fc_outputs(rows, cfg_dir, cfg_tag):
    """Save treatment-conditioned FC rankings aggregated across repeated splits."""
    cfg_dir = Path(cfg_dir)
    if rows is None or len(rows) == 0:
        pd.DataFrame({"message": ["No repeated-split treatment-conditioned FC rows were produced."]}).to_csv(
            cfg_dir / f"{cfg_tag}_txFC_across_splits_skipped.csv", index=False
        )
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(rows, axis=0, ignore_index=True) if isinstance(rows, list) else pd.DataFrame(rows)
    if len(df) == 0:
        pd.DataFrame({"message": ["No repeated-split treatment-conditioned FC rows were produced."]}).to_csv(
            cfg_dir / f"{cfg_tag}_txFC_across_splits_skipped.csv", index=False
        )
        return pd.DataFrame(), pd.DataFrame()
    if "FC_score" not in df.columns:
        df.to_csv(cfg_dir / f"{cfg_tag}_txFC_by_split.csv", index=False)
        return df, pd.DataFrame()
    df = df.copy()
    if "feature" in df.columns and ("feature_family" not in df.columns or "feature_scope" not in df.columns or "is_shared_across_cancers_feature" not in df.columns):
        df = _add_fc_feature_family_columns(df)
    for c in ["treatment_profile", "treatment_active_count", "is_multi_treatment_profile", "feature_family", "feature_scope", "is_shared_across_cancers_feature", "is_cancer_specific_feature", "scope", "group", "H", "seed", "feature"]:
        if c not in df.columns:
            df[c] = False if c.startswith("is_") else ""
    df["FC_score"] = pd.to_numeric(df["FC_score"], errors="coerce")
    if "FC_conditioning_feature_not_ranked" in df.columns:
        rankable_mask = ~df["FC_conditioning_feature_not_ranked"].astype(bool)
    elif "FC_skipped_feature" in df.columns:
        rankable_mask = ~df["FC_skipped_feature"].astype(bool)
    else:
        rankable_mask = np.ones(len(df), dtype=bool)
    rankable_mask = np.asarray(rankable_mask, dtype=bool) & np.isfinite(df["FC_score"].to_numpy(dtype=float))
    df["FC_ranking_method"] = "posterior_predictive_Wasserstein1_feature_collapse"
    df["FC_ranked_feature"] = rankable_mask
    df["FC_rank_within_seed_scope"] = np.nan
    df["FC_rank_percentile_within_seed_scope"] = np.nan
    rank_cols = ["seed", "H", "scope", "group", "treatment_profile"]
    if int(np.sum(rankable_mask)) > 0:
        ranks = (
            df.loc[rankable_mask]
            .groupby(rank_cols, dropna=False)["FC_score"]
            .rank(method="min", ascending=False)
            .astype(float)
        )
        df.loc[rankable_mask, "FC_rank_within_seed_scope"] = ranks.to_numpy(dtype=float)
        counts = df.loc[rankable_mask].groupby(rank_cols, dropna=False)["feature"].transform("size").to_numpy(dtype=float)
        df.loc[rankable_mask, "FC_rank_percentile_within_seed_scope"] = 1.0 - ((ranks.to_numpy(dtype=float) - 1.0) / np.maximum(counts - 1.0, 1.0))
    df.to_csv(cfg_dir / f"{cfg_tag}_txFC_by_split.csv", index=False)
    # Same data under an explicit name for downstream feature-ranking aggregation.
    df.to_csv(cfg_dir / f"{cfg_tag}_txWFC_feature_rankings_by_split.csv", index=False)
    work = df[rankable_mask].copy()
    work = work[np.isfinite(work["FC_score"].to_numpy(dtype=float))].copy()
    if len(work) == 0:
        return df, pd.DataFrame()
    work["rank_within_seed_scope"] = work["FC_rank_within_seed_scope"].astype(float)
    shared_work = work[work["is_shared_across_cancers_feature"].astype(bool)].copy()
    if len(shared_work):
        shared_work.to_csv(cfg_dir / f"{cfg_tag}_txWFC_shared_across_cancers_feature_rankings_by_split.csv", index=False)
    else:
        pd.DataFrame({"message": ["No rankable shared-across-cancers features were found in repeated-split WFC rows."]}).to_csv(
            cfg_dir / f"{cfg_tag}_txWFC_shared_across_cancers_feature_rankings_by_split_skipped.csv", index=False
        )
    group_cols = ["H", "scope", "group", "treatment_profile", "treatment_active_count", "is_multi_treatment_profile", "feature", "feature_family", "feature_scope", "is_shared_across_cancers_feature", "is_cancer_specific_feature"]
    rows_out = []
    for keys, sub in work.groupby(group_cols, dropna=False):
        vals = sub["FC_score"].to_numpy(dtype=float)
        ranks = sub["rank_within_seed_scope"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n_splits_with_estimate": int(sub["seed"].nunique()) if "seed" in sub.columns else int(len(sub)),
            "FC_score_mean_across_splits": float(np.mean(vals)),
            "FC_score_sd_across_splits": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "FC_score_median_across_splits": float(np.median(vals)),
            "FC_score_q05_across_splits": float(np.quantile(vals, 0.05)),
            "FC_score_q95_across_splits": float(np.quantile(vals, 0.95)),
            "mean_rank_across_splits": float(np.nanmean(ranks)) if ranks.size else np.nan,
            "median_rank_across_splits": float(np.nanmedian(ranks)) if ranks.size else np.nan,
            "top10_splits": int(np.sum(ranks <= 10.0)),
            "top10_fraction": float(np.mean(ranks <= 10.0)) if ranks.size else np.nan,
            "top20_splits": int(np.sum(ranks <= 20.0)),
            "top20_fraction": float(np.mean(ranks <= 20.0)) if ranks.size else np.nan,
        })
        rows_out.append(row)
    stability = pd.DataFrame(rows_out)
    if len(work):
        exact_work = work[work["scope"].astype(str).isin(["pooled_exact_treatment_profile", "group_exact_treatment_profile"])].copy()
        if len(exact_work):
            exact_work.to_csv(cfg_dir / f"{cfg_tag}_txWFC_exact_treatment_profile_rankings_by_split.csv", index=False)
    if len(stability):
        stability = stability.sort_values(
            ["top10_fraction", "FC_score_mean_across_splits", "n_splits_with_estimate"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        stability.to_csv(cfg_dir / f"{cfg_tag}_txFC_stability.csv", index=False)
        stability.to_csv(cfg_dir / f"{cfg_tag}_txWFC_feature_ranking_stability.csv", index=False)
        top = stability[pd.to_numeric(stability["n_splits_with_estimate"], errors="coerce") >= 2].head(200).copy()
        top.to_csv(cfg_dir / f"{cfg_tag}_txFC_stability_top.csv", index=False)
        top.to_csv(cfg_dir / f"{cfg_tag}_txWFC_feature_ranking_stability_top.csv", index=False)
        shared_stability = stability[stability["is_shared_across_cancers_feature"].astype(bool)].copy() if "is_shared_across_cancers_feature" in stability.columns else pd.DataFrame()
        if len(shared_stability):
            shared_stability.to_csv(cfg_dir / f"{cfg_tag}_txWFC_shared_across_cancers_feature_ranking_stability.csv", index=False)
            shared_top = shared_stability[pd.to_numeric(shared_stability["n_splits_with_estimate"], errors="coerce") >= 2].head(200).copy()
            shared_top.to_csv(cfg_dir / f"{cfg_tag}_txWFC_shared_across_cancers_feature_ranking_stability_top.csv", index=False)
        else:
            pd.DataFrame({"message": ["No shared-across-cancers features available in WFC stability output."]}).to_csv(
                cfg_dir / f"{cfg_tag}_txWFC_shared_across_cancers_feature_ranking_stability_skipped.csv", index=False
            )
    return df, stability


def fit_one_config_with_august_retries(*, split_seed, audit_path, **fit_kwargs):
    """Retry only numerical/Cholesky failures without changing the outer split."""
    ladder = [float(x) for x in os.getenv("NPJ_JITTER_LADDER", "1e-6,1e-5,1e-4,1e-3,1e-2").split(",")]
    scale = float(os.getenv("NPJ_JITTER_RELATIVE_SCALE", "1.0"))
    audit = []
    last_exc = None
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    for attempt, base_jitter in enumerate(ladder, start=1):
        jitter = float(base_jitter * max(scale, 1e-12))
        init_seed = int(split_seed) + 100000 * (attempt - 1)
        set_all_seeds(init_seed)
        rec = {"configuration": CONFIG_NAME, "split_seed": int(split_seed),
               "initialization_seed": init_seed, "initialization_attempt": attempt,
               "jitter": jitter, "jitter_relative_scale": scale,
               "jitter_ladder": ";".join(map(str, ladder)),
               "matrix_dimension": int(np.asarray(fit_kwargs["Xtr_np"]).shape[0]),
               "cancer_scope": "multitask_all_cancers", "horizon": np.nan}
        try:
            previous_active_jitter = os.environ.get("NPJ_ACTIVE_CHOLESKY_JITTER")
            os.environ["NPJ_ACTIVE_CHOLESKY_JITTER"] = str(jitter)
            try:
                with gpytorch.settings.cholesky_jitter(jitter):
                    out, art = fit_one_config_with_artifacts(**fit_kwargs)
            finally:
                if previous_active_jitter is None:
                    os.environ.pop("NPJ_ACTIVE_CHOLESKY_JITTER", None)
                else:
                    os.environ["NPJ_ACTIVE_CHOLESKY_JITTER"] = previous_active_jitter
            rec.update(status="success", exception_type="", exception_message="")
            audit.append(rec)
            pd.DataFrame(audit).to_csv(audit_path, index=False)
            art["split_seed"] = int(split_seed)
            art["initialization_seed"] = int(init_seed)
            art["successful_jitter"] = float(jitter)
            return out, art
        except Exception as exc:
            last_exc = exc
            numerical = _is_linalg_numerical_failure(exc) or any(
                token in str(exc).lower() for token in ("cholesky", "positive definite", "singular", "nan")
            )
            rec.update(status="failed_numerical" if numerical else "failed_non_numerical",
                       exception_type=type(exc).__name__, exception_message=str(exc))
            audit.append(rec)
            pd.DataFrame(audit).to_csv(audit_path, index=False)
            if not numerical:
                raise
            print(f"[AUGUST][RETRY] split={split_seed} attempt={attempt} jitter={jitter:g}: {exc}", flush=True)
    raise RuntimeError(f"All deterministic Cholesky retries failed for split {split_seed}: {last_exc}") from last_exc


def _august_patient_ids(meta, n):
    if meta is None:
        return [f"heldout_{i}" for i in range(n)], [f"heldout_{i}" for i in range(n)]
    m = meta.reset_index(drop=True)
    pcol = next((c for c in ["PATIENT_ID", "patient_id", "bcr_patient_barcode"] if c in m), None)
    scol = next((c for c in ["SAMPLE_KEY", "sample_id", "sample"] if c in m), None)
    pids = m[pcol].astype(str).tolist() if pcol else [f"heldout_{i}" for i in range(n)]
    sids = m[scol].astype(str).tolist() if scol else pids
    return pids, sids


def _august_treatment_profiles(meta, n):
    if meta is None:
        return ["not_available"] * n
    m = meta.reset_index(drop=True)
    out = []
    for _, row in m.iterrows():
        active = [c.replace("TX_", "") for c in TREATMENT_FLAG_COLS
                  if c in m.columns and pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").fillna(0).iloc[0] > 0.5]
        base = "+".join(active) if active else ("confirmed_untreated" if float(row.get("TX_EXPLICIT_NO_TREATMENT", 0) or 0) > 0.5 else "type_or_status_unknown")
        timing = str(row.get("TX_TIMING_CLASS", "timing_not_available"))
        out.append(f"{base}|{timing}")
    return out


def _august_horizon_status(t, e, H):
    if bool(e) and float(t) <= float(H):
        return "PFI_event_by_horizon"
    if float(t) >= float(H):
        return "event_free_through_horizon"
    return "censored_before_horizon"


def export_august_patient_posteriors(art, outdir, seed, cfg_tag):
    """Save exhaustive held-out posterior summaries before any selective figure."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = art.get("meta_te")
    n = len(art["t_te"])
    pids, sids = _august_patient_ids(meta, n)
    profiles = _august_treatment_profiles(meta, n)
    groups = list(art["groups"])
    horizons = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0]
    summary_rows, risk_rows, raw_rows, hist_rows = [], [], [], []
    for H in horizons:
        SH = posterior_survival_at_H(art["model"], art["Xte_t"], art["g_te_t"], H,
                                     art["alpha_dt"], art["dt_edges"], S=RISK_UQ_MC)
        arrays = {"survival_probability_S": SH,
                  "event_probability_1_minus_S": 1.0 - SH,
                  "cumulative_hazard_neglogS": -np.log(np.clip(SH, 1e-12, 1.0))}
        for i in range(n):
            base = {"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": int(seed),
                    "patient_id": pids[i], "sample_id": sids[i], "horizon_months": H,
                    "cancer": str(groups[int(art["g_te"][i])]), "treatment_profile": profiles[i],
                    "split_role": "heldout_test", "PFI_time_months": float(art["t_te"][i]),
                    "PFI_event": int(art["e_te"][i]),
                    "horizon_status": _august_horizon_status(art["t_te"][i], art["e_te"][i], H),
                    "calibration_status": "native_DGP_posterior"}
            for scale_name, A in arrays.items():
                v = np.asarray(A[:, i], float)
                summary_rows.append({**base, "risk_scale": scale_name,
                    "posterior_mean": float(np.mean(v)), "posterior_sd": float(np.std(v, ddof=1)),
                    "posterior_median": float(np.median(v)), "q05": float(np.quantile(v, .05)),
                    "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)),
                    "q95": float(np.quantile(v, .95)), "posterior_draws": int(len(v))})
            ev = arrays["event_probability_1_minus_S"][:, i]
            risk_rows.append({**base, "event_probability_mean": float(np.mean(ev)),
                              "event_probability_sd": float(np.std(ev, ddof=1)),
                              "event_probability_median": float(np.median(ev)),
                              "event_probability_q05": float(np.quantile(ev, .05)),
                              "event_probability_q95": float(np.quantile(ev, .95))})
            counts, edges = np.histogram(ev, bins=np.linspace(0.0, 1.0, 31), density=False)
            for b, count in enumerate(counts):
                hist_rows.append({**base, "bin_left": float(edges[b]), "bin_right": float(edges[b + 1]),
                                  "count": int(count), "density": float(count / max(len(ev) * (edges[b + 1] - edges[b]), 1e-12))})
            if os.getenv("NPJ_SAVE_RAW_POSTERIOR_DRAWS", "0").lower() in {"1", "true", "yes"}:
                for draw_id in range(SH.shape[0]):
                    raw_rows.append({**base, "draw_id": draw_id,
                                     "S_H": float(SH[draw_id, i]),
                                     "one_minus_S_H": float(1.0 - SH[draw_id, i]),
                                     "neglog_S_H": float(-np.log(max(SH[draw_id, i], 1e-12)))})
    summary = pd.DataFrame(summary_rows)
    risk = pd.DataFrame(risk_rows)
    summary.to_csv(outdir / "patient_posterior_summary_all_horizons.csv", index=False)
    risk.to_csv(outdir / "patient_posterior_risk_by_seed_horizon.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(outdir / "patient_posterior_histogram_bins_all_horizons.csv.gz", index=False, compression="gzip")
    pd.DataFrame(hist_rows).to_csv(outdir / "patient_posterior_density_grid_all_horizons.csv.gz", index=False, compression="gzip")
    try:
        if raw_rows:
            raw = pd.DataFrame(raw_rows)
            try:
                raw.to_parquet(outdir / "patient_posterior_draws_all_horizons.parquet", index=False, partition_cols=["horizon_months", "cancer"])
            except Exception:
                raw.to_csv(outdir / "patient_posterior_draws_all_horizons.csv.gz", index=False, compression="gzip")
        pd.DataFrame([{"configuration": CONFIG_NAME, "seed": seed, "status": "success",
                       "raw_draw_export_requested": bool(raw_rows), "summary_rows": len(summary)}]).to_csv(
            outdir / "posterior_export_status.csv", index=False)
    except Exception as exc:
        pd.DataFrame([{"configuration": CONFIG_NAME, "seed": seed, "status": "raw_draw_export_failed_summaries_preserved",
                       "error": f"{type(exc).__name__}: {exc}", "summary_rows": len(summary)}]).to_csv(
            outdir / "posterior_export_status.csv", index=False)
    monotonic = []
    for pid, sub in risk.groupby("patient_id", sort=False):
        z = sub.sort_values("horizon_months")["event_probability_mean"].to_numpy(float)
        monotonic.append({"patient_id": pid, "seed": seed, "monotone_non_decreasing": bool(np.all(np.diff(z) >= -1e-8)),
                          "largest_decrease": float(min(0.0, np.min(np.diff(z)))) if len(z) > 1 else 0.0})
    pd.DataFrame(monotonic).to_csv(outdir / "patient_posterior_monotonicity_audit.csv", index=False)
    return summary, risk


def export_august_train_defined_km(art, outdir, seed, cfg_tag):
    outdir = Path(outdir)
    assignments, cuts, support, logranks, pairwise_logranks, atrisk = [], [], [], [], [], []
    groups = list(art["groups"])
    pids, sids = _august_patient_ids(art.get("meta_te"), len(art["t_te"]))
    for H in [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0]:
        SHtr = posterior_survival_at_H(art["model"], art["Xtr_t"], art["g_tr_t"], H, art["alpha_dt"], art["dt_edges"], S=FINAL_EVAL_MC_SAMPLES)
        SHte = posterior_survival_at_H(art["model"], art["Xte_t"], art["g_te_t"], H, art["alpha_dt"], art["dt_edges"], S=FINAL_EVAL_MC_SAMPLES)
        for scope, gi in [("pooled", None)] + [("cancer", j) for j in range(len(groups))]:
            mt = np.ones(len(art["t_tr"]), bool) if gi is None else np.asarray(art["g_tr"]) == gi
            me = np.ones(len(art["t_te"]), bool) if gi is None else np.asarray(art["g_te"]) == gi
            scope_name = "Pooled" if gi is None else str(groups[gi])
            low, med, high, _, rap, st = train_defined_tertiles(SHtr[:, mt], SHte[:, me], f"{scope_name} H={H}")
            cuts.append({"configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                         "scope": scope, "cancer": scope_name, **st})
            labels = np.full(int(me.sum()), "nonestimable", object)
            labels[low], labels[med], labels[high] = "Low", "Medium", "High"
            ii = np.where(me)[0]
            for local, original in enumerate(ii):
                assignments.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                    "horizon_months": H, "scope": scope, "cancer": scope_name,
                    "patient_cancer": str(groups[int(art["g_te"][original])]),
                    "patient_id": pids[original], "sample_id": sids[original],
                    "predicted_event_risk": float(rap[local]), "KM_tertile": labels[local],
                    "PFI_time_months": float(art["t_te"][original]), "PFI_event": int(art["e_te"][original]),
                    "estimable": st["estimable"], "nonestimable_reason": st["nonestimable_reason"]})
            for label, mask in zip(["Low", "Medium", "High"], [low, med, high]):
                tt, ee = np.asarray(art["t_te"])[me][mask], np.asarray(art["e_te"])[me][mask]
                median_pfi = np.nan
                if len(tt):
                    try:
                        kmf = KaplanMeierFitter().fit(tt, ee)
                        candidate = float(kmf.median_survival_time_)
                        median_pfi = candidate if np.isfinite(candidate) else np.nan
                    except Exception:
                        median_pfi = np.nan
                support.append({"configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                                "scope": scope, "cancer": scope_name, "KM_tertile": label,
                                "patients": int(len(tt)), "events": int(np.sum(ee)),
                                "censored": int(len(tt) - np.sum(ee)), "estimable": st["estimable"],
                                "median_PFI_months": median_pfi,
                                "median_PFI_estimable": bool(np.isfinite(median_pfi)),
                                "nonestimable_reason": st["nonestimable_reason"]})
                for tau in sorted(set([0.0, H, 12.0, 24.0, 36.0, 60.0])):
                    atrisk.append({"configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                                   "scope": scope, "cancer": scope_name, "KM_tertile": label,
                                   "time_months": tau, "at_risk": int(np.sum(tt >= tau))})
            p = np.nan
            if st["estimable"]:
                try:
                    p = float(multivariate_logrank_test(np.asarray(art["t_te"])[me], labels,
                                                       np.asarray(art["e_te"])[me]).p_value)
                except Exception:
                    p = np.nan
            logranks.append({"configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                             "scope": scope, "cancer": scope_name, "logrank_p": p,
                             "estimable": st["estimable"], "nonestimable_reason": st["nonestimable_reason"]})
            group_masks = {"Low": low, "Medium": med, "High": high}
            for left_label, right_label in combinations(["Low", "Medium", "High"], 2):
                pair_mask = group_masks[left_label] | group_masks[right_label]
                pair_labels = labels[pair_mask]
                pair_t = np.asarray(art["t_te"])[me][pair_mask]
                pair_e = np.asarray(art["e_te"])[me][pair_mask]
                pair_p = np.nan
                pair_estimable = bool(group_masks[left_label].any() and group_masks[right_label].any())
                pair_reason = "" if pair_estimable else "one or both held-out comparison groups are empty"
                if pair_estimable:
                    try:
                        pair_p = float(multivariate_logrank_test(pair_t, pair_labels, pair_e).p_value)
                    except Exception as exc:
                        pair_estimable = False
                        pair_reason = f"{type(exc).__name__}: {exc}"
                pairwise_logranks.append({
                    "configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                    "scope": scope, "cancer": scope_name,
                    "left_KM_tertile": left_label, "right_KM_tertile": right_label,
                    "left_patients": int(group_masks[left_label].sum()),
                    "right_patients": int(group_masks[right_label].sum()),
                    "pairwise_logrank_p": pair_p, "estimable": pair_estimable,
                    "nonestimable_reason": pair_reason,
                })
    adf, cdf, sdf, ldf, pdf, rdf = map(
        pd.DataFrame, [assignments, cuts, support, logranks, pairwise_logranks, atrisk]
    )
    adf.to_csv(outdir / "km_patient_assignments_train_defined_tertiles.csv", index=False)
    cdf.to_csv(outdir / "km_training_cutpoints.csv", index=False)
    sdf.to_csv(outdir / "km_heldout_group_support.csv", index=False)
    ldf.to_csv(outdir / "km_logrank_results.csv", index=False)
    pdf.to_csv(outdir / "km_pairwise_logrank_results.csv", index=False)
    rdf.to_csv(outdir / "km_at_risk_counts.csv", index=False)
    # Figures are made only after the complete plotting source table is on disk.
    for (H, scope, cancer), sub in adf[adf["estimable"].astype(bool)].groupby(["horizon_months", "scope", "cancer"]):
        masks = [(sub["KM_tertile"] == x).to_numpy() for x in ["Low", "Medium", "High"]]
        src = outdir / f"KM_source_H{int(H)}_{_safe_fname(scope)}_{_safe_fname(cancer)}.csv"
        sub.to_csv(src, index=False)
        for ext in ("png", "pdf"):
            km_plot_groups(sub["PFI_time_months"].to_numpy(float), sub["PFI_event"].to_numpy(int),
                           masks, ["Low risk", "Medium risk", "High risk"],
                           f"{cancer}: held-out KM by training-defined risk tertiles (H={int(H)} mo)",
                           outdir / f"KM_trainDefined_H{int(H)}_{_safe_fname(scope)}_{_safe_fname(cancer)}.{ext}",
                           admin_cap=False)
    return adf, cdf, sdf, ldf, pdf


def _august_feature_family(name):
    s = str(name)
    if "mut" in s.lower(): return "mutation"
    if "tmb" in s.lower(): return "TMB"
    if "udp" in s.lower(): return "UDP"
    if "act_" in s.lower() or "path" in s.lower(): return "pathway"
    if "stage" in s.lower() or "grade" in s.lower() or "age" in s.lower() or "ajcc" in s.lower(): return "clinical_stage"
    if "TX_" in s: return "treatment"
    return "other"


def _august_context_feature_eligibility(feature, context):
    """Apply the prespecified Phase-1 shared-versus-specific feature policy."""
    feature = str(feature)
    context = str(context)
    family = _august_feature_family(feature)
    scope = _feature_scope_for_mechanism(feature)
    if family == "treatment":
        return False, scope, "treatment-conditioned WFC is reserved for the timing-aware Phase-2 sensitivity"
    if context == "pooled_shared":
        if scope != "shared_across_cancers":
            return False, scope, "feature is not explicitly shared across BRCA, COAD, and LUAD"
        return True, scope, ""
    if scope == "cancer_specific":
        code = {"Breast": "BRCA", "Colon": "COAD", "Lung": "LUAD"}.get(context, context.upper())
        if f"specific_{code}" not in feature:
            return False, scope, f"feature belongs to a different cancer-specific branch than {context}"
    return True, scope, ""


# LEGACY/INACTIVE: the NPJ August controller never calls this diagnosis-only
# WFC exporter. Corrected treatment-conditioned WFC runs from the saved artifact.
def export_august_primary_wasserstein(art, outdir, seed, cfg_tag):
    """Diagnosis-based Phase-1 WFC with training-defined perturbations."""
    outdir = Path(outdir)
    score_rows, repeat_rows, patient_rows, support_rows, failure_rows = [], [], [], [], []
    Xtr, Xte = np.asarray(art["Xtr_np_sub"], float), np.asarray(art["Xte_np_sub"], float)
    gtr, gte = np.asarray(art["g_tr"], int), np.asarray(art["g_te"], int)
    groups, feats = list(art["groups"]), list(art["feat_names"])
    horizons = [float(x) for x in os.getenv("AUGUST_WFC_HORIZONS", "12,18,24,36").split(",") if x.strip()]
    contexts = [("pooled_shared", None)] + [(str(groups[j]), j) for j in range(len(groups))]
    requested_contexts = {
        ("pooled_shared" if x.strip().lower() == "pooled" else x.strip().lower())
        for x in os.getenv("AUGUST_WFC_CONTEXTS", "").split(",")
        if x.strip()
    }
    if requested_contexts:
        contexts = [(name, gi) for name, gi in contexts if str(name).lower() in requested_contexts]
    R = int(os.getenv("AUGUST_WFC_REPEATS", str(FC_REPEATS)))
    S = int(os.getenv("AUGUST_WFC_DRAWS", str(FC_S)))
    save_patient = os.getenv("AUGUST_WFC_SAVE_PATIENT_LEVEL", "0").lower() in {"1", "true", "yes"}

    def record_context_failure(H, context, reason):
        for feat in feats:
            eligible, feature_scope, eligibility_reason = _august_context_feature_eligibility(feat, context)
            final_reason = reason if eligible else eligibility_reason
            rec = {"configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                   "context": context, "feature": feat, "failure_reason": final_reason,
                   "failure_type": "posterior_numerical_failure" if eligible else "prespecified_ineligible_feature"}
            if eligible:
                failure_rows.append(rec)
            score_rows.append({**rec, "config_tag": cfg_tag,
                "feature_family": _august_feature_family(feat), "feature_scope": feature_scope,
                "eligible_for_context": bool(eligible), "W1_event_probability": np.nan,
                "W1_cumulative_hazard_sensitivity": np.nan,
                "signed_delta_mean_event_probability": np.nan,
                "W1_event_probability_sd_repeats": np.nan,
                "W1_event_probability_standardized_train_IQR": np.nan,
                "estimable": False, "nonestimable_reason": final_reason})

    for H in horizons:
        for context, gi in contexts:
            mtr = np.ones(len(gtr), bool) if gi is None else gtr == gi
            mte = np.ones(len(gte), bool) if gi is None else gte == gi
            eval_tr = mtr & ((np.asarray(art["t_tr"]) >= H) | ((np.asarray(art["e_tr"]) == 1) & (np.asarray(art["t_tr"]) <= H)))
            eval_te = mte & ((np.asarray(art["t_te"]) >= H) | ((np.asarray(art["e_te"]) == 1) & (np.asarray(art["t_te"]) <= H)))
            support_rows.append({"configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                "context": context, "n_train_context": int(mtr.sum()), "n_test_context": int(mte.sum()),
                "n_train_evaluable": int(eval_tr.sum()), "n_test_evaluable": int(eval_te.sum()),
                "effective_sample_size_unweighted": float(eval_te.sum()),
                "test_number_at_risk_at_horizon": int(np.sum((np.asarray(art["t_te"]) >= H) & mte)),
                "test_censored_before_horizon": int(np.sum((np.asarray(art["e_te"]) == 0) & (np.asarray(art["t_te"]) < H) & mte)),
                "train_events_by_horizon": int(np.sum((np.asarray(art["e_tr"]) == 1) & (np.asarray(art["t_tr"]) <= H) & mtr)),
                "test_events_by_horizon": int(np.sum((np.asarray(art["e_te"]) == 1) & (np.asarray(art["t_te"]) <= H) & mte)),
                "treatment_context": False, "landmark_design": False,
                "context_role": "shared_across_cancers" if context == "pooled_shared" else "cancer_specific_evaluation",
                "feature_policy": "explicit_shared_features_only" if context == "pooled_shared" else "shared_plus_matching_cancer_specific_features"})
            if int(mte.sum()) == 0 or int(mtr.sum()) < 2:
                for feat in feats:
                    eligible, feature_scope, eligibility_reason = _august_context_feature_eligibility(feat, context)
                    score_rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                        "horizon_months": H, "context": context, "feature": feat,
                        "feature_family": _august_feature_family(feat), "feature_scope": feature_scope,
                        "eligible_for_context": bool(eligible), "W1_event_probability": np.nan,
                        "W1_cumulative_hazard_sensitivity": np.nan, "signed_delta_mean_event_probability": np.nan,
                        "estimable": False, "nonestimable_reason": eligibility_reason or "insufficient train/test context support"})
                continue
            Xte_c, gte_c = Xte[mte].copy(), gte[mte]
            gte_t = torch.tensor(gte_c, dtype=torch.long, device=device)
            mtr_t = torch.tensor(mtr, dtype=torch.bool, device=device)
            mte_t = torch.tensor(mte, dtype=torch.bool, device=device)
            # Training-derived risk scale and cached baseline posterior draws.
            try:
                set_all_seeds(int(seed) + int(H * 100))
                SHtr = posterior_survival_at_H(art["model"], art["Xtr_t"][mtr_t], art["g_tr_t"][mtr_t], H,
                                               art["alpha_dt"], art["dt_edges"], S=S)
                tr_mean_risk = np.mean(1.0 - SHtr, axis=0)
                risk_scale = float(np.subtract(*np.quantile(tr_mean_risk, [.75, .25])))
                if not np.isfinite(risk_scale) or risk_scale <= 1e-12: risk_scale = np.nan
                baseline = []
                for r in range(R):
                    crn_seed = int(seed) * 1000003 + int(H * 100) + r
                    set_all_seeds(crn_seed)
                    SH0 = posterior_survival_at_H(art["model"], art["Xte_t"][mte_t], gte_t, H,
                                                  art["alpha_dt"], art["dt_edges"], S=S)
                    baseline.append(SH0)
            except Exception as exc:
                record_context_failure(H, context, f"{type(exc).__name__}: {exc}")
                continue
            for j, feat in enumerate(feats):
                family = _august_feature_family(feat)
                eligible, feature_scope, reason = _august_context_feature_eligibility(feat, context)
                if not eligible:
                    score_rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                        "horizon_months": H, "context": context, "feature": feat, "feature_family": family,
                        "feature_scope": feature_scope, "eligible_for_context": False,
                        "W1_event_probability": np.nan, "W1_cumulative_hazard_sensitivity": np.nan,
                        "signed_delta_mean_event_probability": np.nan, "W1_event_probability_sd_repeats": np.nan,
                        "W1_event_probability_standardized_train_IQR": np.nan,
                        "estimable": False, "nonestimable_reason": reason})
                    continue
                if not np.isfinite(Xtr[mtr, j]).all():
                    reason = "nonfinite training values"
                elif np.unique(Xtr[mtr, j]).size < 2:
                    reason = "feature has fewer than two training values"
                vals_event, vals_haz, vals_delta = [], [], []
                vals_event_macro, vals_haz_macro, vals_delta_macro = [], [], []
                if not reason:
                    for r in range(R):
                        Xc = Xte_c.copy()
                        rng = np.random.default_rng(int(seed) * 1000003 + int(H * 100) + r + 7919 * (j + 1))
                        ok_perturb = True
                        perturbed_groups = 0
                        for test_group in np.unique(gte_c):
                            ii = np.where(gte_c == test_group)[0]
                            pool = Xtr[mtr & (gtr == test_group), j]
                            pool = pool[np.isfinite(pool)]
                            if pool.size < 2 or np.unique(pool).size < 2:
                                if gi is None:
                                    # A cancer-specific feature is structurally
                                    # constant outside its cancer; leave those
                                    # pooled rows unchanged.
                                    continue
                                ok_perturb = False
                                reason = f"unperturbable training cell for cancer group {test_group}"
                                break
                            Xc[ii, j] = rng.choice(pool, size=len(ii), replace=True)
                            perturbed_groups += 1
                        if perturbed_groups == 0:
                            ok_perturb = False
                            reason = "no training cancer cell had perturbable feature support"
                        if not ok_perturb: break
                        Xt = _as_torch_from_raw_standardized(Xc, art["standardize_mu"], art["standardize_sd"])
                        crn_seed = int(seed) * 1000003 + int(H * 100) + r
                        set_all_seeds(crn_seed)
                        try:
                            SH1 = posterior_survival_at_H(art["model"], Xt, gte_t, H,
                                                          art["alpha_dt"], art["dt_edges"], S=S)
                        except Exception as exc:
                            reason = f"posterior numerical failure: {type(exc).__name__}: {exc}"
                            break
                        Y0e, Y1e = 1.0 - baseline[r], 1.0 - SH1
                        Y0h, Y1h = -np.log(np.clip(baseline[r], 1e-12, 1.0)), -np.log(np.clip(SH1, 1e-12, 1.0))
                        w_event = _wasserstein1_empirical_per_row(Y0e, Y1e)
                        w_haz = _wasserstein1_empirical_per_row(Y0h, Y1h)
                        delta = np.nanmean(Y1e - Y0e, axis=0)
                        vals_event.append(float(np.nanmean(w_event)))
                        vals_haz.append(float(np.nanmean(w_haz)))
                        vals_delta.append(float(np.nanmean(delta)))
                        group_masks = [gte_c == group_index for group_index in np.unique(gte_c)]
                        vals_event_macro.append(float(np.mean([np.nanmean(w_event[z]) for z in group_masks])))
                        vals_haz_macro.append(float(np.mean([np.nanmean(w_haz[z]) for z in group_masks])))
                        vals_delta_macro.append(float(np.mean([np.nanmean(delta[z]) for z in group_masks])))
                        repeat_rows.append({"configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                            "context": context, "feature": feat, "feature_scope": feature_scope, "repeat": r,
                            "W1_event_probability": vals_event[-1], "W1_cumulative_hazard_sensitivity": vals_haz[-1],
                            "signed_delta_mean_event_probability": vals_delta[-1],
                            "W1_event_probability_equal_cancer_macro": vals_event_macro[-1],
                            "signed_delta_mean_event_probability_equal_cancer_macro": vals_delta_macro[-1]})
                        if save_patient:
                            ids = np.where(mte)[0]
                            pids, _ = _august_patient_ids(art.get("meta_te"), len(gte))
                            for k, original in enumerate(ids):
                                patient_rows.append({"configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                                    "context": context, "feature": feat, "repeat": r, "patient_id": pids[original],
                                    "cancer": str(groups[gte[original]]), "W1_event_probability": float(w_event[k]),
                                    "W1_cumulative_hazard_sensitivity": float(w_haz[k]),
                                    "signed_delta_mean_event_probability": float(delta[k])})
                if reason or not vals_event:
                    rec = {"configuration": CONFIG_NAME, "seed": seed, "horizon_months": H,
                           "context": context, "feature": feat, "feature_scope": feature_scope,
                           "eligible_for_context": True, "failure_reason": reason or "no successful repeats",
                           "failure_type": "feature_perturbation_or_posterior_failure"}
                    failure_rows.append(rec)
                    score_rows.append({**rec, "config_tag": cfg_tag, "feature_family": family,
                        "W1_event_probability": np.nan, "W1_cumulative_hazard_sensitivity": np.nan,
                        "signed_delta_mean_event_probability": np.nan, "W1_event_probability_sd_repeats": np.nan,
                        "W1_event_probability_standardized_train_IQR": np.nan,
                        "estimable": False, "nonestimable_reason": rec["failure_reason"]})
                else:
                    score_rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                        "horizon_months": H, "context": context, "feature": feat, "feature_family": family,
                        "feature_scope": feature_scope, "eligible_for_context": True,
                        "W1_event_probability": float(np.mean(vals_event)),
                        "W1_event_probability_sd_repeats": float(np.std(vals_event, ddof=1)) if len(vals_event) > 1 else 0.0,
                        "W1_cumulative_hazard_sensitivity": float(np.mean(vals_haz)),
                        "signed_delta_mean_event_probability": float(np.mean(vals_delta)),
                        "W1_event_probability_equal_cancer_macro": float(np.mean(vals_event_macro)),
                        "W1_cumulative_hazard_equal_cancer_macro": float(np.mean(vals_haz_macro)),
                        "signed_delta_mean_event_probability_equal_cancer_macro": float(np.mean(vals_delta_macro)),
                        "W1_event_probability_standardized_train_IQR": float(np.mean(vals_event) / risk_scale) if np.isfinite(risk_scale) else np.nan,
                        "successful_repeats": len(vals_event), "estimable": True, "nonestimable_reason": "",
                        "primary_scale": "event_probability_1_minus_S", "landmark_design": False,
                        "perturbation_distribution": "training_only_within_cancer_resampling"})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(outdir / "august_wasserstein_scores_all_features.csv", index=False)
    pd.DataFrame(support_rows).to_csv(outdir / "august_wasserstein_context_support.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(outdir / "august_wasserstein_failures.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(outdir / "wasserstein_repeat_level_scores.csv.gz", index=False, compression="gzip")
    if save_patient:
        pd.DataFrame(patient_rows).to_csv(outdir / "wasserstein_patient_level_w1.csv.gz", index=False, compression="gzip")
    return scores


def _august_uno_ipcw_cindex(t_train, e_train, t_test, e_test, risk, tau=60.0):
    try:
        km = KaplanMeierFitter().fit(np.asarray(t_train, float), event_observed=1 - np.asarray(e_train, int))
        def G(x):
            return max(float(km.predict(max(float(x) - 1e-9, 0.0))), 1e-6)
        num = den = 0.0
        t, e, r = np.asarray(t_test, float), np.asarray(e_test, int), np.asarray(risk, float)
        for i in range(len(t)):
            if e[i] != 1 or t[i] > tau:
                continue
            w = 1.0 / (G(t[i]) ** 2)
            js = np.where(t > t[i])[0]
            for j in js:
                den += w
                num += w * (1.0 if r[i] > r[j] else 0.5 if r[i] == r[j] else 0.0)
        return (num / den, "") if den > 0 else (np.nan, "no IPCW-comparable held-out pairs")
    except Exception as exc:
        return np.nan, f"{type(exc).__name__}: {exc}"


def export_august_performance_metrics(art, outdir, seed, cfg_tag):
    rows, horizon_rows = [], []
    groups = list(art["groups"])
    SH60 = posterior_survival_at_H(art["model"], art["Xte_t"], art["g_te_t"], 60.0,
                                   art["alpha_dt"], art["dt_edges"], S=FINAL_EVAL_MC_SAMPLES)
    risk60 = np.mean(1.0 - SH60, axis=0)
    group_cs = []
    for scope, mask in [("pooled", np.ones(len(risk60), bool))] + [
        (str(groups[j]), np.asarray(art["g_te"]) == j) for j in range(len(groups))]:
        try:
            c, _ = harrell_c(np.asarray(art["t_te"])[mask], risk60[mask],
                             np.asarray(art["e_te"])[mask], auto_orient=False)
            reason = ""
        except Exception as exc:
            c, reason = np.nan, f"{type(exc).__name__}: {exc}"
        rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                     "metric": "Harrell_C_index", "scope": scope, "horizon_months": np.nan,
                     "value": c, "patients": int(mask.sum()), "events": int(np.sum(np.asarray(art["e_te"])[mask])),
                     "estimable": bool(np.isfinite(c)), "nonestimable_reason": reason})
        if scope == "pooled":
            rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                         "metric": "pooled_patient_weighted_C_index", "scope": "pooled",
                         "horizon_months": np.nan, "value": c, "patients": int(mask.sum()),
                         "events": int(np.sum(np.asarray(art["e_te"])[mask])),
                         "estimable": bool(np.isfinite(c)), "nonestimable_reason": reason})
        if scope != "pooled":
            group_cs.append((str(scope), float(c), int(mask.sum()), str(reason)))
    complete_group_cs = [x for x in group_cs if np.isfinite(x[1])]
    complete_macro = len(complete_group_cs) == len(groups)
    macro_reason = "" if complete_macro else "one or more cancer-specific C-indices were non-estimable"
    macro_value = float(np.mean([x[1] for x in complete_group_cs])) if complete_macro else np.nan
    weighted_value = (
        float(np.average([x[1] for x in complete_group_cs], weights=[x[2] for x in complete_group_cs]))
        if complete_macro else np.nan
    )
    rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                 "metric": "equal_weight_macro_C_index", "scope": "macro", "horizon_months": np.nan,
                 "value": macro_value, "patients": len(risk60),
                 "events": int(np.sum(art["e_te"])), "estimable": complete_macro,
                 "nonestimable_reason": macro_reason})
    rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                 "metric": "cancer_stratified_patient_weighted_C_index", "scope": "cancer_stratified_pooled",
                 "horizon_months": np.nan, "value": weighted_value,
                 "patients": len(risk60), "events": int(np.sum(art["e_te"])),
                 "estimable": complete_macro, "nonestimable_reason": macro_reason})
    uno, reason = _august_uno_ipcw_cindex(art["t_tr"], art["e_tr"], art["t_te"], art["e_te"], risk60, tau=60.0)
    rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                 "metric": "Uno_IPCW_C_index", "scope": "pooled", "horizon_months": 60.0,
                 "value": uno, "patients": len(risk60), "events": int(np.sum(art["e_te"])),
                 "estimable": bool(np.isfinite(uno)), "nonestimable_reason": reason})
    brier_by_scope = {}
    for H in [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0]:
        SH = posterior_survival_at_H(art["model"], art["Xte_t"], art["g_te_t"], H,
                                     art["alpha_dt"], art["dt_edges"], S=FINAL_EVAL_MC_SAMPLES)
        risk = np.mean(1.0 - SH, axis=0)
        for scope, mask0 in [("pooled", np.ones(len(risk), bool))] + [
            (str(groups[j]), np.asarray(art["g_te"]) == j) for j in range(len(groups))]:
            evaluable = (np.asarray(art["t_te"]) >= H) | ((np.asarray(art["e_te"]) == 1) & (np.asarray(art["t_te"]) <= H))
            mask = mask0 & evaluable
            y = ((np.asarray(art["e_te"]) == 1) & (np.asarray(art["t_te"]) <= H))[mask].astype(int)
            reason = ""
            auc = ap = brier = np.nan
            if int(mask.sum()) < 2 or len(np.unique(y)) < 2:
                reason = "insufficient evaluable held-out events/non-events"
            else:
                try:
                    auc = float(roc_auc_score(y, risk[mask]))
                    ap = float(average_precision_score(y, risk[mask]))
                    brier = float(np.mean((y - risk[mask]) ** 2))
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
            brier_by_scope.setdefault(scope, []).append((H, brier))
            for metric, value in [("AUROC", auc), ("AUPRC", ap), ("Brier_score", brier)]:
                horizon_rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
                    "metric": metric, "scope": scope, "horizon_months": H, "value": value,
                    "patients": int(mask.sum()), "events": int(y.sum()), "estimable": bool(np.isfinite(value)),
                    "nonestimable_reason": "" if np.isfinite(value) else reason})
    for scope, vals in brier_by_scope.items():
        good = [(h, v) for h, v in vals if np.isfinite(v)]
        ibs = float(np.trapz([v for _, v in good], [h for h, _ in good]) / (good[-1][0] - good[0][0])) if len(good) >= 2 else np.nan
        horizon_rows.append({"configuration": CONFIG_NAME, "config_tag": cfg_tag, "seed": seed,
            "metric": "integrated_Brier_score", "scope": scope, "horizon_months": np.nan,
            "value": ibs, "patients": len(art["t_te"]), "events": int(np.sum(art["e_te"])),
            "estimable": bool(np.isfinite(ibs)), "nonestimable_reason": "" if np.isfinite(ibs) else "fewer than two estimable horizons"})
    out = pd.DataFrame(rows + horizon_rows)
    out.to_csv(Path(outdir) / "august_performance_metrics.csv", index=False)
    return out


def export_seed_c_indices_by_horizon_and_group(fit_metrics, art, outdir, seed, cfg_tag):
    """Write the exact fitted split C-indices in an explicit long-form CSV.

    The values come directly from ``evaluate_model_mc`` rather than from a
    second posterior evaluation, so this table exactly matches the C-index
    columns in the seed's fit-results row. It includes every requested horizon,
    the pooled held-out test set, and each held-out cancer group.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    groups = list(art["groups"])
    g_te = np.asarray(art["g_te"], int)
    e_te = np.asarray(art["e_te"], int)
    rows = []
    for H in [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0]:
        for scope, gi in [("pooled", None)] + [(str(g), j) for j, g in enumerate(groups)]:
            mask = np.ones(len(g_te), dtype=bool) if gi is None else (g_te == gi)
            metric_column = (
                f"c_test_H{int(H)}" if gi is None
                else f"{scope}_c_test_H{int(H)}"
            )
            value = pd.to_numeric(
                pd.Series([fit_metrics.get(metric_column, np.nan)]), errors="coerce"
            ).iloc[0]
            rows.append({
                "configuration": CONFIG_NAME,
                "config_tag": str(cfg_tag),
                "seed": int(seed),
                "dataset": "heldout_test",
                "scope": scope,
                "horizon_months": float(H),
                "metric": "Harrell_C_index_at_horizon",
                "source_metric_column": metric_column,
                "value": float(value) if np.isfinite(value) else np.nan,
                "patients": int(mask.sum()),
                "events": int(e_te[mask].sum()),
                "estimable": bool(np.isfinite(value)),
                "nonestimable_reason": "" if np.isfinite(value) else "held-out C-index was non-estimable",
            })
    cdf = pd.DataFrame(rows)
    cdf.to_csv(outdir / "c_index_by_horizon_and_group.csv", index=False)

    # Include the same explicit rows in the per-seed all-performance CSV used by
    # the master collector. Remove prior copies first so a retry is idempotent.
    perf_path = outdir / "august_performance_metrics.csv"
    perf = pd.read_csv(perf_path) if perf_path.is_file() else pd.DataFrame()
    if len(perf) and "metric" in perf.columns:
        perf = perf[~perf["metric"].astype(str).eq("Harrell_C_index_at_horizon")].copy()
    pd.concat([perf, cdf], ignore_index=True, sort=False).to_csv(perf_path, index=False)
    return cdf



def export_august_primary_outputs(art, outdir, seed, cfg_tag):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Save exhaustive posterior/KM/performance source data and the fitted artifact.
    # The only WFC workflow runs later from this artifact in the parallel WFC array.
    export_august_patient_posteriors(art, outdir, seed, cfg_tag)
    export_august_performance_metrics(art, outdir, seed, cfg_tag)
    export_august_train_defined_km(art, outdir, seed, cfg_tag)
    if os.getenv("AUGUST_DEFER_WFC", "0").lower() in {"1", "true", "yes"}:
        artifact_path = outdir.parent / f"august_fitted_artifacts_seed{int(seed)}.pt"
        torch.save(art, artifact_path)
        pd.DataFrame([{"configuration": CONFIG_NAME, "seed": seed, "status": "deferred_to_parallel_WFC_array",
                       "artifact_path": str(artifact_path)}]).to_csv(outdir / "august_wasserstein_deferred_status.csv", index=False)
    else:
        raise RuntimeError("Corrected WFC must run from the saved fitted artifact; set AUGUST_DEFER_WFC=1")


def validate_august_required_seed_outputs(cfg_dir, split_dir, seed, cfg_tag):
    """Fail a seed when a required NPJ analysis or source CSV is missing."""
    cfg_dir, split_dir = Path(cfg_dir), Path(split_dir)
    required = {
        "patient posterior risk": split_dir / "patient_posterior_risk_by_seed_horizon.csv",
        "patient posterior density data": split_dir / "patient_posterior_density_grid_all_horizons.csv.gz",
        "performance metrics": split_dir / "august_performance_metrics.csv",
        "C-index by horizon and group": split_dir / "c_index_by_horizon_and_group.csv",
        "KM patient assignments": split_dir / "km_patient_assignments_train_defined_tertiles.csv",
        "KM training cutpoints": split_dir / "km_training_cutpoints.csv",
        "KM held-out support": split_dir / "km_heldout_group_support.csv",
        "KM log-rank results": split_dir / "km_logrank_results.csv",
        "KM pairwise log-rank results": split_dir / "km_pairwise_logrank_results.csv",
        "survival metrics": split_dir / "test_survival_metric_summary.csv",
        "within-cancer survival metrics": split_dir / "test_group_survival_metric_summary.csv",
        "calibration summary": split_dir / "test_horizon_calibration_summary.csv",
        "cross-fitted calibration parameters": split_dir / "posthoc_crossfit_calibration_model_params.csv",
        "early-failure posterior predictions": split_dir / "early_failure_patient_predictions_with_posterior_intervals.csv",
        "early-failure calibration": split_dir / "early_failure_ipcw_calibration_summary.csv",
        "paired treatment contrasts by seed": cfg_dir / f"{cfg_tag}_pairwise_feature_tx_contrasts_seed{int(seed)}.csv",
        "paired treatment contrasts combined": cfg_dir / f"{cfg_tag}_pairwise_feature_tx_contrasts_by_split.csv",
    }
    frames, checks = {}, []
    for label, path in required.items():
        status, error, rows = "ok", "", 0
        try:
            if not path.is_file() or path.stat().st_size <= 1:
                raise FileNotFoundError(f"missing or empty: {path}")
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"zero data rows: {path}")
            frames[label] = df
            rows = int(len(df))
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        checks.append({
            "configuration": CONFIG_NAME,
            "config_tag": str(cfg_tag),
            "seed": int(seed),
            "required_output": label,
            "path": str(path),
            "status": status,
            "rows": rows,
            "error": error,
        })

    expected_prediction_h = {3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0}
    expected_treatment_h = {float(h) for h in FEATURE_STRATIFIED_TX_H_LIST}
    horizon_contracts = [
        ("patient posterior risk", "horizon_months", expected_prediction_h),
        ("performance metrics", "horizon_months", expected_prediction_h),
        ("C-index by horizon and group", "horizon_months", expected_prediction_h),
        ("KM training cutpoints", "horizon_months", expected_prediction_h),
        ("paired treatment contrasts by seed", "H", expected_treatment_h),
    ]
    for label, column, expected in horizon_contracts:
        if label not in frames:
            continue
        df = frames[label]
        got = set(pd.to_numeric(df.get(column), errors="coerce").dropna().astype(float).tolist())
        missing = sorted(h for h in expected if not any(np.isclose(h, x) for x in got))
        if missing:
            checks.append({
                "configuration": CONFIG_NAME,
                "config_tag": str(cfg_tag),
                "seed": int(seed),
                "required_output": f"{label} horizon coverage",
                "path": str(required[label]),
                "status": "failed",
                "rows": int(len(df)),
                "error": f"missing required horizons: {missing}; observed={sorted(got)}",
            })

    cindex_label = "C-index by horizon and group"
    if cindex_label in frames:
        cdf = frames[cindex_label]
        expected_scopes = {"pooled", "Breast", "Colon", "Lung"}
        for scope in sorted(expected_scopes):
            sub = cdf[cdf.get("scope", pd.Series("", index=cdf.index)).astype(str).eq(scope)]
            got = set(pd.to_numeric(sub.get("horizon_months"), errors="coerce").dropna().astype(float))
            missing = sorted(h for h in expected_prediction_h if not any(np.isclose(h, x) for x in got))
            if missing:
                checks.append({
                    "configuration": CONFIG_NAME,
                    "config_tag": str(cfg_tag),
                    "seed": int(seed),
                    "required_output": f"C-index horizon coverage for {scope}",
                    "path": str(required[cindex_label]),
                    "status": "failed",
                    "rows": int(len(sub)),
                    "error": f"missing required horizons: {missing}; observed={sorted(got)}",
                })

    check_df = pd.DataFrame(checks)
    check_path = cfg_dir / f"required_npj_output_contract_seed{int(seed)}.csv"
    check_df.to_csv(check_path, index=False)
    failed = check_df[check_df["status"].astype(str) != "ok"]
    if len(failed):
        details = " | ".join(
            f"{r.required_output}: {r.error}" for r in failed.itertuples()
        )
        raise RuntimeError(f"Required NPJ output contract failed for seed {seed}: {details}")
    return check_df


def main():
    set_all_seeds(SEED)
    wfc_artifact = os.getenv("AUGUST_WFC_ONLY_ARTIFACT", "").strip()
    if wfc_artifact:
        part_outdir = Path(os.environ["AUGUST_WFC_PART_OUTDIR"])
        validate_august_outdir(part_outdir)
        part_outdir.mkdir(parents=True, exist_ok=True)
        try:
            art = torch.load(wfc_artifact, map_location=device, weights_only=False)
        except TypeError:
            art = torch.load(wfc_artifact, map_location=device)
        split_seed = int(os.getenv("SEED", art.get("split_seed", SEED)))
        cfg_tag = os.getenv("AUGUST_CFG_TAG", "cfg_TOP0_S2_P2_MT2_OUT1")
        set_all_seeds(split_seed)
        horizon = float(os.environ["AUGUST_WFC_HORIZON"])
        run_corrected_wfc_partition(
            art=art, outdir=part_outdir, configuration=CONFIG_NAME,
            cfg_tag=cfg_tag, seed=split_seed, horizon=horizon,
            risk_samples_fn=neglogS_samples_at_H_mt,
            wasserstein_fn=_wasserstein1_empirical_per_row,
            family_fn=_feature_family_for_mechanism,
            scope_fn=_feature_scope_for_mechanism,
            posterior_draws=FC_S,
        )
        return
    TARGET_CONFIGS = [(0, 2, 2, 2, 1)]
    TARGET_CONFIGS = _env_target_configs("TARGET_CONFIGS", TARGET_CONFIGS)
    # Guard against accidental duplicate config entries overwriting the same cfg_dir.
    TARGET_CONFIGS = list(dict.fromkeys(TARGET_CONFIGS))
    if any(int(cfg[3]) != 2 for cfg in TARGET_CONFIGS):
        raise ValueError(f"This MT2 file requires mt_latent_rank=2. Got TARGET_CONFIGS={TARGET_CONFIGS}")
    seeds = (
        list(dict.fromkeys([int(s) for s in MULTI_SEEDS]))
        if REPEATED_SPLIT_MODE
        else [SEED]
    )

    print("\n====================== TCGA WORKFLOW C: PathClin 10RS txWFC MT2 ======================")
    print(
        f"SEED={SEED} | REPEATED_SPLIT_MODE={REPEATED_SPLIT_MODE} | RUN_OAT_SENSITIVITY={RUN_OAT_SENSITIVITY}"
    )
    print(f"Using seed-specific all-cancer preprocessing outputs from: {PREP_OUTDIR}")
    print(f"Target configs = {TARGET_CONFIGS}")
    print(
        "Workflow C output layer is group-specific; output_rank is metadata only, not multitask output rank."
    )

    all_reports = []
    for cfg in TARGET_CONFIGS:
        surv_top_k, shared_dim, private_dim, mt_latent_rank, output_rank = map(int, cfg)
        cfg_tag = f"cfg_TOP{surv_top_k}_S{shared_dim}_P{private_dim}_MT{mt_latent_rank}_OUT{output_rank}"
        cfg_dir = OUTDIR / cfg_tag
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 86)
        print(f"CONFIG {cfg_tag}")
        print("=" * 86)

        rows = []
        split_count_rows = []
        feature_tx_contrast_rows = []
        treatment_adjusted_fc_rows = []
        dgp_branch_attribution_rows = []
        groups_ref = None
        for i, seed in enumerate(seeds, start=1):
            print("\n" + "-" * 78)
            print(f"[{i}/{len(seeds)}] outer split seed={seed}")
            set_all_seeds(seed)
            d = load_inputs(seed=seed)
            Xtr_np, Xte_np = d["Xtr_np"], d["Xte_np"]
            t_tr, t_te = d["t_tr"], d["t_te"]
            e_tr, e_te = d["e_tr"], d["e_te"]
            g_tr, g_te = d["g_tr"], d["g_te"]
            meta_tr, meta_te = d["meta_tr"], d["meta_te"]
            feat_names, blocks, groups = d["feat_names"], d["blocks"], d["groups"]
            prep_dir = d.get("prep_dir", str(PREP_OUTDIR))
            groups_ref = groups

            print("\nLoaded train group sizes:")
            print(meta_tr["GROUP_LABEL"].value_counts(dropna=False).to_string())
            print("\nLoaded test group sizes:")
            print(meta_te["GROUP_LABEL"].value_counts(dropna=False).to_string())
            print("\nLoaded train event sizes:")
            print(
                meta_tr["PFI_EVENT"].value_counts(dropna=False).sort_index().to_string()
            )
            print("\nLoaded test event sizes:")
            print(
                meta_te["PFI_EVENT"].value_counts(dropna=False).sort_index().to_string()
            )

            # Repeated-split event-count audit for early-failure horizons. These
            # counts are independent of the DGP fit and document censoring/event
            # support for every seed used in the repeated-split summary.
            for _cr in _early_failure_counts(t_tr, e_tr, g_tr, groups, "train"):
                _cr["seed"] = int(seed)
                split_count_rows.append(_cr)
            for _cr in _early_failure_counts(t_te, e_te, g_te, groups, "test"):
                _cr["seed"] = int(seed)
                split_count_rows.append(_cr)

            row = {
                "seed": int(seed),
                "prep_dir": prep_dir,
                "SURV_TOP_K": surv_top_k,
                "SHARED_DIM": shared_dim,
                "PRIVATE_DIM": private_dim,
                "MT_LATENT_RANK": mt_latent_rank,
                "OUTPUT_RANK": output_rank,
            }
            try:
                out, art_seed = fit_one_config_with_august_retries(
                    split_seed=seed,
                    audit_path=cfg_dir / f"cholesky_retry_audit_seed{int(seed)}.csv",
                    Xtr_np=Xtr_np,
                    Xte_np=Xte_np,
                    t_tr=t_tr,
                    t_te=t_te,
                    e_tr=e_tr,
                    e_te=e_te,
                    g_tr=g_tr,
                    g_te=g_te,
                    groups=groups,
                    feat_names=feat_names,
                    blocks=blocks,
                    surv_top_k=surv_top_k,
                    shared_dim=shared_dim,
                    private_dim=private_dim,
                    mt_latent_rank=mt_latent_rank,
                    output_rank=output_rank,
                    meta_tr=meta_tr,
                    meta_te=meta_te,
                    prep_dir=prep_dir,
                )
                row.update(out)
                row["status"] = "ok"

                # Phase-1 authoritative exports are completed before any
                # representative/selective patient figures are attempted.
                split_posthoc_dir = cfg_dir / f"split_seed{int(seed)}_analysis"
                export_august_primary_outputs(art_seed, split_posthoc_dir, seed, cfg_tag)
                export_seed_c_indices_by_horizon_and_group(
                    out, art_seed, split_posthoc_dir, seed, cfg_tag
                )

                # May26 rescue patch: produce the full posthoc/calibration/plot packet
                # for this exact trained repeated split before deleting art_seed.
                # This prevents the old behavior where only best_split_seed*_analysis
                # received calibration and most posthoc plots.
                if ENABLE_POSTHOC_ANALYSIS_FOR_EACH_SPLIT:
                    try:
                        print(f"[POSTHOC][SPLIT] Running full posthoc + calibration for seed {seed}: {split_posthoc_dir}", flush=True)
                        set_current_split_plot_label(f"split_seed={int(seed)}")
                        try:
                            run_posthoc_analysis(art_seed, split_posthoc_dir)
                        finally:
                            set_current_split_plot_label(None)
                        pd.DataFrame([out]).to_csv(
                            split_posthoc_dir / f"split_seed{int(seed)}_fit_metrics.csv", index=False
                        )
                        print(f"[POSTHOC][SPLIT] Saved full split analysis to: {split_posthoc_dir}", flush=True)
                    except Exception as posthoc_exc:
                        split_posthoc_dir.mkdir(parents=True, exist_ok=True)
                        with open(
                            split_posthoc_dir / "posthoc_exception_summary.txt",
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(f"{type(posthoc_exc).__name__}: {posthoc_exc}\n")
                        print(f"[POSTHOC][SPLIT][ERROR] seed {seed}: {type(posthoc_exc).__name__}: {posthoc_exc}", flush=True)
                        if REQUIRE_COMPLETE_NPJ_OUTPUTS:
                            raise RuntimeError(
                                f"Required per-split posthoc analysis failed for seed {seed}: {posthoc_exc}"
                            ) from posthoc_exc

                # FC and treatment-contrast outputs run after the per-split KM/posterior-risk/calibration packet.

                df_branch_seed = pd.DataFrame()
                if ENABLE_REPEATED_SPLIT_DGP_BRANCH_ATTRIBUTION:
                    try:
                        df_branch_seed = compute_dgp_branch_feature_attribution(
                            art_seed, seed=seed, cfg_tag=cfg_tag
                        )
                        if df_branch_seed is not None and len(df_branch_seed):
                            dgp_branch_attribution_rows.append(df_branch_seed)
                            df_branch_seed.to_csv(
                                cfg_dir / f"{cfg_tag}_dgp_branch_feature_attribution_seed{int(seed)}.csv",
                                index=False,
                            )
                    except Exception as branch_exc:
                        df_branch_seed = pd.DataFrame()
                        print(
                            f"[WARN] DGP branch attribution failed for seed {seed}: {type(branch_exc).__name__}: {branch_exc}"
                        )
                if _should_run_feature_stratified_tx_contrasts_for_seed(seed):
                    try:
                        df_tx_contrast_seed = compute_feature_stratified_treatment_risk_contrasts(art_seed, seed=seed, cfg_tag=cfg_tag)
                        if len(df_tx_contrast_seed):
                            df_tx_contrast_seed = merge_dgp_branch_attribution_into_feature_tx(
                                df_tx_contrast_seed,
                                df_branch_seed,
                            )
                            feature_tx_contrast_rows.append(df_tx_contrast_seed)
                            df_tx_contrast_seed.to_csv(cfg_dir / f"{cfg_tag}_pairwise_feature_tx_contrasts_seed{int(seed)}.csv", index=False)
                            if ENABLE_REPEATED_SPLIT_DGP_BRANCH_ATTRIBUTION:
                                df_tx_contrast_seed.to_csv(cfg_dir / f"{cfg_tag}_pairwise_feature_tx_contrasts_seed{int(seed)}_WITH_DGP_BRANCH_CALLS.csv", index=False)
                            # Compatibility copy with a neutral, branch-independent name.
                            df_tx_contrast_seed.to_csv(cfg_dir / f"{cfg_tag}_feature_tx_contrasts_seed{int(seed)}.csv", index=False)
                        elif REQUIRE_COMPLETE_NPJ_OUTPUTS:
                            raise RuntimeError(
                                f"Required paired treatment-profile contrasts produced no rows for seed {seed}."
                            )
                    except Exception as tx_exc:
                        print(f"[WARN] Feature-stratified treatment contrasts failed for seed {seed}: {type(tx_exc).__name__}: {tx_exc}")
                        if REQUIRE_COMPLETE_NPJ_OUTPUTS:
                            raise RuntimeError(
                                f"Required paired treatment-profile contrasts failed for seed {seed}: {tx_exc}"
                            ) from tx_exc
                del art_seed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as ex:
                row["status"] = "fail"
                row["error"] = str(ex)
                print(f"[FAILED] {ex}")
            rows.append(row)
            pd.DataFrame(rows).to_csv(cfg_dir / f"{cfg_tag}_progress.csv", index=False)

        df = pd.DataFrame(rows)
        df.to_csv(cfg_dir / f"{cfg_tag}_results.csv", index=False)
        ok = df[df["status"] == "ok"].copy()

        # Backward-compatible C-index-only summary. Kept for old downstream scripts.
        summary_rows = []
        if len(ok):
            metric_cols = [
                c
                for c in ok.columns
                if re.search(r"(^c_test_H\d+$)|(_c_test_H\d+$)", c)
            ]
            for col in metric_cols:
                summ = _metric_summary_row(ok[col])
                summary_rows.append(
                    {
                        "metric": col,
                        "mean": summ["mean"],
                        "sd": summ["sd"],
                        "mean_sd": _fmt_mean_sd(summ["mean"], summ["sd"]),
                        "n_non_nan": summ["n_non_nan"],
                        "n_seeds": int(ok["seed"].nunique()),
                    }
                )
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(cfg_dir / f"{cfg_tag}_summary.csv", index=False)

        # Manuscript-ready repeated-split summaries for all main metrics:
        # C-index, Somers' Dxy, Brier, IBS, AUROC, and AUPRC, pooled and by cancer.
        export_repeated_split_metric_summaries(
            ok, cfg_dir=cfg_dir, cfg_tag=cfg_tag, groups=groups_ref
        )

        # Repeated-split event-count/prevalence summaries for fixed-horizon
        # early-failure endpoints, saved for train/test, pooled and by cancer.
        export_repeated_split_event_count_summaries(
            split_count_rows, cfg_dir=cfg_dir, cfg_tag=cfg_tag
        )

        # Do not execute or emit either legacy WFC workflow. The single corrected
        # treatment-conditioned WFC is executed only by the saved-artifact array.
        treatment_adjusted_fc_split_df = pd.DataFrame()
        treatment_adjusted_fc_stability_df = pd.DataFrame()
        dgp_branch_attribution_split_df, dgp_branch_attribution_stability_df = save_dgp_branch_attribution_outputs(
            dgp_branch_attribution_rows, cfg_dir=cfg_dir, cfg_tag=cfg_tag
        )
        feature_tx_contrast_split_df, feature_tx_direction_stability_df = save_feature_stratified_treatment_contrast_outputs(
            feature_tx_contrast_rows, cfg_dir=cfg_dir, cfg_tag=cfg_tag
        )
        joined_top_wfc_tx_contrast_df = pd.DataFrame()

        if REQUIRE_COMPLETE_NPJ_OUTPUTS:
            for completed_seed in seeds:
                validate_august_required_seed_outputs(
                    cfg_dir,
                    cfg_dir / f"split_seed{int(completed_seed)}_analysis",
                    completed_seed,
                    cfg_tag,
                )

        best_seed = None
        # NPJ August: never rank or select outer splits by held-out performance.
        if False and len(ok):
            ok = ok.sort_values(
                ["c_test_H60", "c_test_H36"], ascending=[False, False]
            ).reset_index(drop=True)
            ok.to_csv(cfg_dir / f"{cfg_tag}_results_sorted.csv", index=False)
            best_seed = int(ok.iloc[0]["seed"])
            print(
                f"\n[POSTHOC] Best pooled split seed for {cfg_tag} = {best_seed} | c_test_H60={ok.iloc[0]['c_test_H60']:.4f} | c_test_H36={ok.iloc[0]['c_test_H36']:.4f}"
            )
            if ENABLE_BEST_SPLIT_REFIT_POSTHOC_ANALYSIS:
                # Reproduce the same training initialization/MC stream used when this outer split
                # was fit during the sweep. Without this, the posthoc refit uses the leftover
                # RNG state from previous configs/seeds, so best_seed only controls the data split.
                set_all_seeds(best_seed)
                d_best = load_inputs(seed=best_seed)
                out_best, art_best = fit_one_config_with_artifacts(
                    Xtr_np=d_best["Xtr_np"],
                    Xte_np=d_best["Xte_np"],
                    t_tr=d_best["t_tr"],
                    t_te=d_best["t_te"],
                    e_tr=d_best["e_tr"],
                    e_te=d_best["e_te"],
                    g_tr=d_best["g_tr"],
                    g_te=d_best["g_te"],
                    groups=d_best["groups"],
                    feat_names=d_best["feat_names"],
                    blocks=d_best["blocks"],
                    surv_top_k=surv_top_k,
                    shared_dim=shared_dim,
                    private_dim=private_dim,
                    mt_latent_rank=mt_latent_rank,
                    output_rank=output_rank,
                    meta_tr=d_best.get("meta_tr"),
                    meta_te=d_best.get("meta_te"),
                    prep_dir=d_best.get("prep_dir"),
                )
                best_dir = cfg_dir / f"best_split_seed{best_seed}_analysis"
                try:
                    set_current_split_plot_label(f"best_split_seed={int(best_seed)}")
                    try:
                        run_posthoc_analysis(art_best, best_dir)
                    finally:
                        set_current_split_plot_label(None)
                except Exception as exc:
                    with open(
                        best_dir / "posthoc_exception_summary.txt", "w", encoding="utf-8"
                    ) as f:
                        f.write(f"{type(exc).__name__}: {exc}\n")
                    print(f"[POSTHOC][ERROR] {type(exc).__name__}: {exc}")
                pd.DataFrame([out_best]).to_csv(
                    best_dir / "best_split_refit_metrics.csv", index=False
                )
                print(f"[POSTHOC] Saved full best-split analysis to: {best_dir}")
            else:
                print(
                    f"[POSTHOC] Per-split posthoc is enabled; skipping duplicate best-split refit. "
                    f"Use split_seed{best_seed}_analysis for the best split outputs.",
                    flush=True,
                )

        report = {
            "seed": SEED,
            "repeated_split_mode": bool(REPEATED_SPLIT_MODE),
            "outer_split_seeds": seeds,
            "surv_top_k": surv_top_k,
            "shared_dim": shared_dim,
            "private_dim": private_dim,
            "mt_latent_rank": mt_latent_rank,
            "output_rank": output_rank,
            "groups": groups_ref,
            "prep_outdir": str(PREP_OUTDIR),
            "outdir": str(cfg_dir),
            "summary_all_metrics_csv": str(
                cfg_dir / f"{cfg_tag}_summary_all_metrics.csv"
            ),
            "summary_all_metrics_wide_csv": str(
                cfg_dir / f"{cfg_tag}_summary_all_metrics_wide.csv"
            ),
            "early_failure_event_counts_by_split_csv": str(
                cfg_dir / f"{cfg_tag}_early_failure_event_counts_by_split.csv"
            ),
            "early_failure_event_counts_summary_csv": str(
                cfg_dir / f"{cfg_tag}_early_failure_event_counts_summary.csv"
            ),
            "corrected_wfc_partition_root": str(
                cfg_dir / "split_seed*_analysis" / "corrected_wfc_parts"
            ),
            "corrected_wfc_final_shared_table": "master_corrected_wfc_final_shared_across_cancers.csv",
            "feature_tx_contrasts_by_split_csv": str(
                cfg_dir / f"{cfg_tag}_feature_tx_contrasts_by_split.csv"
            ),
            "feature_tx_direction_stability_csv": str(
                cfg_dir / f"{cfg_tag}_feature_tx_direction_stability.csv"
            ),
            "feature_tx_contrast_seeds": FEATURE_STRATIFIED_TX_CONTRAST_SEEDS,
            "feature_tx_stratum_policy": "overall plus train-scope lower/upper tertile tails applied to held-out test patients; no median-split fallback for continuous/pathway features",
            "survival_endpoint": "PFI",
            "early_failure_horizons_months": EARLY_FAILURE_H_LIST,
            "primary_early_failure_horizon_months": PRIMARY_EARLY_FAILURE_H,
            "discrete_time_edges": (
                DT_EDGES_MANUAL if USE_MANUAL_DT_EDGES else f"quantile_{DT_BINS}_bins"
            ),
            "early_bin_weighting": bool(USE_EARLY_BIN_WEIGHTS),
            "grouping": "Cancer-type task groups (Breast, Lung, Colon) from all-cancer preprocessing; treatment encoded as baseline covariates.",
            "feature_set_note": "May24 PathClin main version: pathway activity/consistency + clinical; no UDP, no TMB, no sparse mutation-profile block, no gene-expression PCs.",
            "fc_note": "One corrected treatment-conditioned WFC workflow runs from saved fitted artifacts at H=12,18,24,36 months; treatment contrasts remain a separate analysis.",
            "selection_note": (
                "Fixed outer split at seed 42."
                if not REPEATED_SPLIT_MODE
                else "Repeated outer-split sweep. Mean/SD summaries are computed across completed seeds; full posthoc plots and calibration are generated for every completed split."
            ),
            "best_pooled_seed": None,
            "best_pooled_split_analysis_dir": (
                None
            ),
            "per_split_posthoc_analysis_dir_pattern": str(cfg_dir / "split_seed<seed>_analysis"),
        }
        with open(cfg_dir / f"{cfg_tag}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        all_reports.append(report)

    with open(
        OUTDIR / "PathClin_10RS_master_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_reports, f, indent=2)
    print(
        f"\nSaved master report: {OUTDIR / 'PathClin_10RS_master_report.json'}"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
