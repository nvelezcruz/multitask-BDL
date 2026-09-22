#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCGA NPJ Precision Oncology August baseline comparisons for PathUDP_MutTMB inputs.

Uses the split-specific files written by the preprocessing scripts, including either
unsuffixed names (tcga_brca_design_train.csv) or suffixed names such as
  tcga_brca_design_train_PathClin.csv
  tcga_brca_design_train_PathUDP_MutTMB.csv
inside split_seed{seed}* folders.

Models implemented in one file:
- Paper-style fixed-horizon baselines:
  - LR, RF, SVM per cohort
  - STL-BNN per cohort
  - MTL-BNN pooled with hard parameter sharing + task descriptor
- Survival baselines:
  - MT-DeepCox pooled multitask neural Cox model
  - MT-DiscHazard pooled multitask discrete-time neural hazard model
  - DeepCox pooled and per cohort
  - CoxPH pooled and per cohort
  - optional random survival forest implementation (disabled for this analysis)

Endpoint: PFI_TIME_MONTHS / PFI_EVENT.
Default horizons: 3, 6, 12, 18, 24, 36, 60 months to match DGP summaries.
Default seeds: 42..51. Override with MULTI_SEEDS="42,43,...".
"""

import argparse
import inspect
import json
import re
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as _lifelines_c
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_int_list(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return list(default)
    vals = []
    for tok in re.split(r"[,;\s]+", str(raw).strip()):
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


def _env_optional_float(name, default=None):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    s = str(raw).strip().lower()
    if s in {"none", "null", "na", "nan", "off", "false"}:
        return None
    return float(raw)


# ------------------------ USER KNOBS ------------------------
SEED = 42
RUN_MULTI_SEEDS = _env_bool("RUN_MULTI_SEEDS", False)
MULTI_SEEDS = _env_int_list("MULTI_SEEDS", list(range(42, 52)))  # 10 seeds: 42..51
INNER_VAL_FRAC = 0.20
INNER_SPLIT_SEED = 42  # default only; run_baselines_one_seed passes the current seed
# Main comparison matches DGP by fitting neural baselines on the full outer train split.
# Set USE_INNER_VAL_FOR_NEURAL=1 only for an optional early-stopping sensitivity run.
USE_INNER_VAL_FOR_NEURAL = _env_bool("USE_INNER_VAL_FOR_NEURAL", False)
ADMIN_CENSOR_MONTHS = _env_optional_float("ADMIN_CENSOR_MONTHS", None)
USE_DISCRETE_TIME = True
DT_BINS = 12
DT_LINK = "logistic"
DGP_DT_EDGES_MONTHS = [0.0, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0, 48.0, 60.0, 84.0, 120.0]
DT_EDGES_MONTHS = _env_float_list("DT_EDGES_MONTHS", DGP_DT_EDGES_MONTHS)
POSTERIOR_H_MONTHS = 120.0
EVAL_HORIZONS_MONTHS = _env_float_list("EVAL_HORIZONS_MONTHS", [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0])
BNN_EMB_DIM = int(os.environ.get("BNN_EMB_DIM", "8"))
BNN_TASK_EMB_DIM = int(os.environ.get("BNN_TASK_EMB_DIM", "3"))
BNN_BATCH_SIZE = int(os.environ.get("BNN_BATCH_SIZE", "128"))
BNN_EPOCHS = int(os.environ.get("BNN_EPOCHS", "50"))
BNN_PATIENCE = int(os.environ.get("BNN_PATIENCE", "10"))
SURV_NN_EPOCHS = int(os.environ.get("SURV_NN_EPOCHS", "900"))
SURV_NN_PATIENCE = int(os.environ.get("SURV_NN_PATIENCE", "20"))

# Fixed non-neural baseline settings; these are not selected on test performance.
LR_MAX_ITER = 5000
RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 2
SVM_C = 1.0
SVM_GAMMA = "scale"
COX_PENALIZER = 1e-3

RUN_PAPER_BINARY_BASELINES = _env_bool("RUN_PAPER_BINARY_BASELINES", True)
RUN_SURVIVAL_BASELINES = _env_bool("RUN_SURVIVAL_BASELINES", True)
RUN_RSF_BASELINE = _env_bool("RUN_RSF_BASELINE", False)
CONFIG_NAME = os.environ.get("BASELINE_INPUT_CONFIG", "PathUDP_MutTMB")
_SUFFIX_BY_CONFIG = {
    "PathClin": "_PathClin",
    "PathUDP_MutTMB": "_PathUDP_MutTMB",
    "PathNoUDP_MutTMB": "_PathNoUDP_MutTMB",
}
EXPECTED_OUTPUT_SUFFIX = os.environ.get(
    "BASELINE_INPUT_SUFFIX",
    _SUFFIX_BY_CONFIG.get(CONFIG_NAME, f"_{CONFIG_NAME}" if CONFIG_NAME else ""),
)
ALLOW_UNSUFFIXED_PREP = _env_bool("ALLOW_UNSUFFIXED_PREP", False)

BASE = Path(os.environ["TCGA_ROOT"]).expanduser().resolve()
PREP_OUTDIR = Path(
    os.environ.get(
        "PREP_OUTDIR",
        str(
            BASE
            / "NPJ_August_2026_FULLRERUN_PathUDP_MutTMB_pathway_UDP_clinical_treatment_mutation_TMB_MT2"
            / "preprocessing"
        ),
    )
).expanduser()
OUTDIR = Path(
    os.environ.get(
        "OUTDIR", str(BASE / "NPJ_August_2026_FULLRERUN_Baselines_PathUDP_MutTMB_MT2")
    )
).expanduser()
OUTDIR.mkdir(parents=True, exist_ok=True)

P_XTR = PREP_OUTDIR / "tcga_brca_design_train.csv"
P_XTE = PREP_OUTDIR / "tcga_brca_design_test.csv"
P_META_TR = PREP_OUTDIR / "tcga_brca_metadata_train.csv"
P_META_TE = PREP_OUTDIR / "tcga_brca_metadata_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------ helpers ------------------------
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


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


def eval_cindex(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    risk = np.asarray(risk, dtype=float).reshape(-1)
    ok = np.isfinite(times) & np.isfinite(risk) & np.isfinite(events)
    times, events, risk = times[ok], events[ok], risk[ok]
    if times.size < 3 or len(np.unique(events)) < 2:
        return float("nan")
    try:
        # Baseline models return event risk/probability; lifelines C-index expects
        # larger scores to correspond to longer survival. Use a fixed sign, not
        # a test-set-dependent auto flip.
        c, _ = harrell_c(times, -risk, events, auto_orient=False)
        return float(c)
    except Exception:
        return float("nan")


def _per_group_cindex(
    t: np.ndarray, e: np.ndarray, risk: np.ndarray, g: np.ndarray, groups: list[str]
):
    out = {}
    for gi, name in enumerate(groups):
        m = g == gi
        if int(m.sum()) < 5:
            out[name] = float("nan")
            continue
        if len(np.unique(e[m])) < 2:
            out[name] = float("nan")
            continue
        out[name] = eval_cindex(t[m], e[m], risk[m])
    return out


def admin_censor_np(t: np.ndarray, e: np.ndarray, max_mo: float | None = ADMIN_CENSOR_MONTHS):
    t2 = np.asarray(t, float).copy()
    e2 = np.asarray(e, int).copy()
    if max_mo is None:
        return t2, e2
    max_mo = float(max_mo)
    if not np.isfinite(max_mo):
        return t2, e2
    over = t2 > max_mo
    e2[over] = 0
    t2[over] = max_mo
    return t2, e2


def make_inner_val_split(g_tr, e_tr, val_frac=INNER_VAL_FRAC, seed=INNER_SPLIT_SEED):
    n = len(g_tr)
    idx = np.arange(n)
    if (val_frac is None) or (val_frac <= 0.0) or (n < 10):
        return idx.copy(), np.array([], dtype=int)

    strat = pd.Series(g_tr).astype(str) + "__" + pd.Series(e_tr).astype(str)
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=int(seed)
        )
        fit_idx, val_idx = next(splitter.split(np.zeros((n, 1)), strat))
    except Exception:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(idx)
        n_val = max(1, int(round(float(val_frac) * n)))
        val_idx = np.sort(perm[:n_val])
        fit_idx = np.sort(perm[n_val:])
    return np.asarray(fit_idx, dtype=int), np.asarray(val_idx, dtype=int)


# ------------------------ all-cancer file helpers ------------------------
ALL_CANCER_CODES = ["BRCA", "LUAD", "COAD"]


def _prefix_group_specific_pc_columns(
    df: pd.DataFrame, cancer_code: str
) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        s = str(c)
        if re.match(r"^PC\d+$", s):
            rename[c] = f"{cancer_code}_{s}"
    return df.rename(columns=rename)


def _concat_align_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
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


def _concat_align_train_test_frames(
    train_frames: list[pd.DataFrame], test_frames: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align train and test to one train-defined union without changing values."""
    if len(train_frames) != len(test_frames) or len(train_frames) == 0:
        raise ValueError("Training and test frame lists must be nonempty and paired.")
    cols = []
    seen = set()
    for i, (xtr, xte) in enumerate(zip(train_frames, test_frames)):
        if xtr.columns.duplicated().any() or xte.columns.duplicated().any():
            raise ValueError(f"Duplicate design columns detected for cancer index {i}.")
        if set(map(str, xtr.columns)) != set(map(str, xte.columns)):
            missing_test = sorted(set(map(str, xtr.columns)) - set(map(str, xte.columns)))
            extra_test = sorted(set(map(str, xte.columns)) - set(map(str, xtr.columns)))
            raise ValueError(
                f"Train/test feature mismatch for cancer index {i}: "
                f"missing_test={missing_test[:10]}, extra_test={extra_test[:10]}"
            )
        for c in xtr.columns:
            if c not in seen:
                cols.append(c)
                seen.add(c)
    xtr_all = pd.concat(
        [df.reindex(columns=cols, fill_value=0.0) for df in train_frames], axis=0
    )
    xte_all = pd.concat(
        [df.reindex(columns=cols, fill_value=0.0) for df in test_frames], axis=0
    )
    return xtr_all, xte_all


def _find_split_dir(prep_root: Path, seed: int | None = None) -> Path:
    """Return the config-matched split-specific preprocessing folder."""
    prep_root = Path(prep_root).expanduser()
    if seed is None:
        return prep_root

    seed_tag = f"split_seed{int(seed)}"
    expected_name = f"{seed_tag}{EXPECTED_OUTPUT_SUFFIX}"

    if prep_root.name == expected_name:
        return prep_root
    if ALLOW_UNSUFFIXED_PREP and prep_root.name == seed_tag:
        return prep_root
    if prep_root.name.startswith(seed_tag):
        raise FileNotFoundError(
            f"PREP_OUTDIR points to split folder {prep_root.name!r}, but "
            f"BASELINE_INPUT_CONFIG={CONFIG_NAME!r} expects {expected_name!r}."
        )

    candidates = [prep_root / expected_name]
    if ALLOW_UNSUFFIXED_PREP:
        candidates.append(prep_root / seed_tag)

    for q in candidates:
        if q.exists() and q.is_dir():
            return q

    tried = ", ".join(str(q) for q in candidates)
    raise FileNotFoundError(
        f"Missing preprocessing split directory for seed {int(seed)} and "
        f"BASELINE_INPUT_CONFIG={CONFIG_NAME!r}. Tried: {tried}"
    )


def _find_split_file(prep_dir: Path, cancer_code: str, kind: str, split: str) -> Path:
    """Find the config-matched preprocessing CSV for one cancer/split."""
    code = str(cancer_code).lower()
    stem = f"tcga_{code}_{kind}_{split}"
    suffixes = [EXPECTED_OUTPUT_SUFFIX]
    if ALLOW_UNSUFFIXED_PREP:
        suffixes.append("")
    candidates = [prep_dir / f"{stem}{suf}.csv" for suf in suffixes]
    for q in candidates:
        if q.exists():
            return q
    tried = ", ".join(str(q) for q in candidates)
    raise FileNotFoundError(
        f"Missing {kind}/{split} file for {cancer_code} under {prep_dir}. "
        f"BASELINE_INPUT_CONFIG={CONFIG_NAME!r} expects suffix {EXPECTED_OUTPUT_SUFFIX!r}. "
        f"Tried: {tried}"
    )


def _first_existing_col(df: pd.DataFrame, candidates: list[str], *, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these required columns were found: {candidates}. Available: {list(df.columns)[:40]}...")
    return None


def _load_one_cancer_split(prep_dir: Path, cancer_code: str):
    p_xtr = _find_split_file(prep_dir, cancer_code, "design", "train")
    p_xte = _find_split_file(prep_dir, cancer_code, "design", "test")
    p_meta_tr = _find_split_file(prep_dir, cancer_code, "metadata", "train")
    p_meta_te = _find_split_file(prep_dir, cancer_code, "metadata", "test")

    Xtr = pd.read_csv(p_xtr, index_col=0)
    Xte = pd.read_csv(p_xte, index_col=0)
    meta_tr = pd.read_csv(p_meta_tr)
    meta_te = pd.read_csv(p_meta_te)

    meta_tr["SAMPLE_KEY"] = meta_tr["SAMPLE_KEY"].astype(str)
    meta_te["SAMPLE_KEY"] = meta_te["SAMPLE_KEY"].astype(str)

    if meta_tr["SAMPLE_KEY"].duplicated().any() or meta_te["SAMPLE_KEY"].duplicated().any():
        raise ValueError(f"Duplicate SAMPLE_KEY values detected for {cancer_code}.")
    overlap = set(meta_tr["SAMPLE_KEY"]) & set(meta_te["SAMPLE_KEY"])
    if overlap:
        raise ValueError(
            f"Train/test patient leakage detected for {cancer_code}: "
            f"{sorted(overlap)[:10]}"
        )
    missing_tr = sorted(set(meta_tr["SAMPLE_KEY"]) - set(map(str, Xtr.index)))
    missing_te = sorted(set(meta_te["SAMPLE_KEY"]) - set(map(str, Xte.index)))
    if missing_tr or missing_te:
        raise KeyError(
            f"Design/metadata SAMPLE_KEY mismatch for {cancer_code}: "
            f"missing_train={missing_tr[:10]}, missing_test={missing_te[:10]}"
        )

    Xtr = Xtr.loc[meta_tr["SAMPLE_KEY"].tolist()].copy()
    Xte = Xte.loc[meta_te["SAMPLE_KEY"].tolist()].copy()
    Xtr = _prefix_group_specific_pc_columns(Xtr, cancer_code=str(cancer_code))
    Xte = _prefix_group_specific_pc_columns(Xte, cancer_code=str(cancer_code))
    return Xtr, Xte, meta_tr, meta_te, {
        "x_train": str(p_xtr), "x_test": str(p_xte),
        "meta_train": str(p_meta_tr), "meta_test": str(p_meta_te),
    }


def _get_pfi_time_months(meta: pd.DataFrame) -> np.ndarray:
    if "PFI_TIME_MONTHS" in meta.columns:
        return meta["PFI_TIME_MONTHS"].to_numpy(dtype=float)
    if "PFI_TIME_DAYS" in meta.columns:
        return (meta["PFI_TIME_DAYS"].to_numpy(dtype=float) / 30.4375).astype(float)
    raise KeyError("PFI endpoint requested but metadata lacks PFI_TIME_MONTHS and PFI_TIME_DAYS.")


def _get_pfi_event(meta: pd.DataFrame) -> np.ndarray:
    if "PFI_EVENT" not in meta.columns:
        raise KeyError("PFI endpoint requested but metadata lacks PFI_EVENT.")
    return meta["PFI_EVENT"].to_numpy(dtype=int)


# ------------------------ data loader ------------------------
def load_inputs(seed: int | None = None):
    prep_dir = _find_split_dir(PREP_OUTDIR, seed=seed)

    Xtr_parts, Xte_parts = [], []
    meta_tr_parts, meta_te_parts = [], []
    file_manifest = {}
    for cancer_code in ALL_CANCER_CODES:
        Xtr_i, Xte_i, meta_tr_i, meta_te_i, files_i = _load_one_cancer_split(
            prep_dir, cancer_code
        )
        Xtr_parts.append(Xtr_i)
        Xte_parts.append(Xte_i)
        meta_tr_parts.append(meta_tr_i)
        meta_te_parts.append(meta_te_i)
        file_manifest[cancer_code] = files_i

    Xtr, Xte = _concat_align_train_test_frames(Xtr_parts, Xte_parts)
    meta_tr = pd.concat(meta_tr_parts, axis=0, ignore_index=True)
    meta_te = pd.concat(meta_te_parts, axis=0, ignore_index=True)

    feat_names = list(map(str, Xtr.columns.tolist()))
    groups = sorted(
        pd.concat([meta_tr["GROUP_LABEL"], meta_te["GROUP_LABEL"]])
        .astype(str)
        .unique()
        .tolist()
    )
    group_to_idx = {g: i for i, g in enumerate(groups)}

    xtr_np = Xtr.to_numpy(dtype=float)
    xte_np = Xte.to_numpy(dtype=float)
    if not np.isfinite(xtr_np).all() or not np.isfinite(xte_np).all():
        raise ValueError(
            "The August design matrices contain non-finite values. Baselines will "
            "not impute held-out data independently; correct preprocessing first."
        )

    return {
        "Xtr": xtr_np,
        "Xte": xte_np,
        "t_tr": _get_pfi_time_months(meta_tr),
        "t_te": _get_pfi_time_months(meta_te),
        "e_tr": _get_pfi_event(meta_tr),
        "e_te": _get_pfi_event(meta_te),
        "g_tr": meta_tr["GROUP_LABEL"]
        .astype(str)
        .map(group_to_idx)
        .to_numpy(dtype=int),
        "g_te": meta_te["GROUP_LABEL"]
        .astype(str)
        .map(group_to_idx)
        .to_numpy(dtype=int),
        "sample_key_tr": meta_tr["SAMPLE_KEY"].astype(str).to_numpy(),
        "sample_key_te": meta_te["SAMPLE_KEY"].astype(str).to_numpy(),
        "group_label_tr": meta_tr["GROUP_LABEL"].astype(str).to_numpy(),
        "group_label_te": meta_te["GROUP_LABEL"].astype(str).to_numpy(),
        "groups": groups,
        "feat_names": feat_names,
        "num_groups": int(len(groups)),
        "prep_dir": str(prep_dir),
        "file_manifest": file_manifest,
    }


# ==========================================================================
# Neural Cox models
# ==========================================================================
class MLP(torch.nn.Module):
    def __init__(
        self, in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [
                torch.nn.Linear(d, h),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            d = h
        layers += [torch.nn.Linear(d, out_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultitaskCoxMLP(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_groups: int,
        hidden: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(hidden) == 0:
            hidden = [32]
        num_groups = int(num_groups)
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        layers = []
        d = in_dim
        for h in hidden:
            layers += [
                torch.nn.Linear(d, h),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            d = h
        self.trunk = torch.nn.Sequential(*layers)
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(d, 1) for _ in range(num_groups)]
        )
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        g = g.reshape(-1).long()
        if g.numel() != h.size(0):
            raise ValueError(
                f"Group index length {g.numel()} does not match batch size {h.size(0)}"
            )
        if torch.any(g < 0) or torch.any(g >= self.num_groups):
            raise ValueError("Group indices out of range for multitask heads")
        all_scores = torch.cat([head(h) for head in self.heads], dim=1)
        return all_scores.gather(1, g[:, None])


def cox_ph_loss(
    log_risk: torch.Tensor, t: torch.Tensor, e: torch.Tensor
) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Breslow handling of ties."""
    eps = 1e-8
    order = torch.argsort(t, descending=True)
    f = log_risk.reshape(-1)[order]
    t = t.reshape(-1)[order]
    e = e.reshape(-1)[order].float()
    log_cum_risk = torch.logcumsumexp(f, dim=0)
    # Every member of a tied-time group uses the risk set evaluated after all
    # patients at that time have entered the descending-time cumulative sum.
    _, counts = torch.unique_consecutive(t, return_counts=True)
    group_ends = torch.cumsum(counts, dim=0) - 1
    end_for_row = torch.repeat_interleave(group_ends, counts)
    tied_log_denom = log_cum_risk[end_for_row]
    pll = torch.sum((f - tied_log_denom) * e) / (e.sum() + eps)
    return -pll


def grouped_cox_ph_loss(
    log_risk: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    losses = []
    uniq = torch.unique(g.reshape(-1))
    for gi in uniq:
        m = g.reshape(-1) == gi
        if int(m.sum().item()) < 2:
            continue
        eg = e.reshape(-1)[m]
        if int(eg.sum().item()) < 1:
            continue
        losses.append(cox_ph_loss(log_risk.reshape(-1)[m], t.reshape(-1)[m], eg))
    if not losses:
        return cox_ph_loss(log_risk, t, e)
    return torch.stack(losses).mean()


def fit_deepsurv(
    Xtr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    Xva: np.ndarray | None,
    t_va: np.ndarray | None,
    e_va: np.ndarray | None,
    *,
    hidden=(64, 32),
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=SURV_NN_EPOCHS,
    patience=SURV_NN_PATIENCE,
    device: str = "cuda",
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    _, p = Xtr.shape
    model = MLP(p, list(hidden), out_dim=1, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ttr_t = torch.tensor(t_tr, dtype=torch.float32, device=dev)
    etr_t = torch.tensor(e_tr, dtype=torch.long, device=dev)

    has_val = (
        Xva is not None and t_va is not None and e_va is not None and len(t_va) > 0
    )
    if has_val:
        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
        tva_t = torch.tensor(t_va, dtype=torch.float32, device=dev)
        eva_t = torch.tensor(e_va, dtype=torch.long, device=dev)
    else:
        Xva_t = tva_t = eva_t = None

    best_state = None
    best_val = float("inf")
    best_train = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        f = model(Xtr_t).reshape(-1)
        loss = cox_ph_loss(f, ttr_t, etr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if not has_val:
            train_loss = float(loss.detach().cpu().item())
            if train_loss + 1e-6 < best_train:
                best_train = train_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        elif ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                fva = model(Xva_t).reshape(-1)
                vloss = float(cox_ph_loss(fva, tva_t, eva_t).item())
            if vloss + 1e-6 < best_val:
                best_val = vloss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_deepsurv(
    model: torch.nn.Module, X: np.ndarray, device: str = "cuda"
) -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    return model(Xt).reshape(-1).detach().cpu().numpy()


def fit_mtdeepsurv(
    Xtr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    g_tr: np.ndarray,
    Xva: np.ndarray | None,
    t_va: np.ndarray | None,
    e_va: np.ndarray | None,
    g_va: np.ndarray | None,
    *,
    num_groups: int,
    hidden=(64, 32),
    emb_dim=8,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=SURV_NN_EPOCHS,
    patience=SURV_NN_PATIENCE,
    device: str = "cuda",
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    _, p = Xtr.shape
    model = MultitaskCoxMLP(
        p,
        num_groups=int(num_groups),
        hidden=list(hidden),
        dropout=dropout,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ttr_t = torch.tensor(t_tr, dtype=torch.float32, device=dev)
    etr_t = torch.tensor(e_tr, dtype=torch.long, device=dev)
    gtr_t = torch.tensor(g_tr, dtype=torch.long, device=dev)

    has_val = (
        Xva is not None
        and t_va is not None
        and e_va is not None
        and g_va is not None
        and len(t_va) > 0
    )
    if has_val:
        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
        tva_t = torch.tensor(t_va, dtype=torch.float32, device=dev)
        eva_t = torch.tensor(e_va, dtype=torch.long, device=dev)
        gva_t = torch.tensor(g_va, dtype=torch.long, device=dev)
    else:
        Xva_t = tva_t = eva_t = gva_t = None

    best_state = None
    best_val = float("inf")
    best_train = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        f = model(Xtr_t, gtr_t).reshape(-1)
        loss = grouped_cox_ph_loss(f, ttr_t, etr_t, gtr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if not has_val:
            train_loss = float(loss.detach().cpu().item())
            if train_loss + 1e-6 < best_train:
                best_train = train_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        elif ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                fva = model(Xva_t, gva_t).reshape(-1)
                vloss = float(grouped_cox_ph_loss(fva, tva_t, eva_t, gva_t).item())
            if vloss + 1e-6 < best_val:
                best_val = vloss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_mtdeepsurv(
    model: torch.nn.Module, X: np.ndarray, g: np.ndarray, device: str = "cuda"
) -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    gt = torch.tensor(g, dtype=torch.long, device=dev)
    return model(Xt, gt).reshape(-1).detach().cpu().numpy()


# ==========================================================================
# Multitask discrete-time neural hazard model
# ==========================================================================


def _make_dt_bin_edges(
    t_train: np.ndarray, n_bins: int = DT_BINS, max_time: float | None = ADMIN_CENSOR_MONTHS
) -> np.ndarray:
    # Match the DGP MT2 early-resolution discrete-time grid by default.  If the
    # train split extends beyond the last manual edge, append the train maximum.
    t_train = np.asarray(t_train, dtype=float)
    t = t_train[np.isfinite(t_train) & (t_train >= 0.0)]
    t_max = float(np.nanmax(t)) if t.size else 0.0
    if max_time is not None:
        try:
            cap = float(max_time)
            if np.isfinite(cap):
                t_max = max(t_max, cap)
        except Exception:
            pass
    manual = np.asarray(DT_EDGES_MONTHS, dtype=float)
    manual = np.unique(manual[np.isfinite(manual) & (manual >= 0.0)])
    if manual.size >= 2:
        mx = max(t_max, float(manual[-1]), 1.0) if t.size == 0 else max(t_max, 1.0)
        edges = manual[manual < mx]
        if edges.size == 0 or edges[0] > 0.0:
            edges = np.concatenate([[0.0], edges])
        if not np.any(np.isclose(edges, mx)):
            edges = np.concatenate([edges, [mx]])
        edges = np.unique(edges)
        if edges.size >= 2:
            return edges.astype(float)
    upper = max(t_max, 1.0)
    return np.linspace(0.0, upper, int(n_bins) + 1, dtype=float)


def _encode_discrete_time_targets(t: np.ndarray, e: np.ndarray, bin_edges: np.ndarray):
    t = np.asarray(t, dtype=float).reshape(-1)
    e = np.asarray(e, dtype=int).reshape(-1)
    upper = np.asarray(bin_edges[1:], dtype=float)
    J = upper.size
    N = t.size
    y = np.zeros((N, J), dtype=np.float32)
    mask = np.zeros((N, J), dtype=np.float32)
    for i in range(N):
        if int(e[i]) == 1:
            j = int(np.searchsorted(upper, t[i], side="left"))
            j = min(max(j, 0), J - 1)
            if j > 0:
                mask[i, :j] = 1.0
            mask[i, j] = 1.0
            y[i, j] = 1.0
        else:
            if t[i] >= bin_edges[-1]:
                mask[i, :] = 1.0
            else:
                j = int(np.searchsorted(upper, t[i], side="right"))
                j = min(max(j, 0), J)
                if j > 0:
                    mask[i, :j] = 1.0
    return y, mask


def _masked_discrete_time_loss(
    logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    per = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="none"
    )
    w = mask.float()
    denom = w.sum().clamp_min(1.0)
    return (per * w).sum() / denom


def grouped_discrete_time_loss(
    logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    losses = []
    uniq = torch.unique(g.reshape(-1))
    for gi in uniq:
        m = g.reshape(-1) == gi
        if int(m.sum().item()) < 2:
            continue
        if float(mask[m].sum().item()) <= 0.0:
            continue
        losses.append(_masked_discrete_time_loss(logits[m], y[m], mask[m]))
    if not losses:
        return _masked_discrete_time_loss(logits, y, mask)
    return torch.stack(losses).mean()


def _n_bins_for_horizon(bin_edges: np.ndarray, horizon: float) -> int:
    upper = np.asarray(bin_edges[1:], dtype=float)
    J = upper.size
    # Include only intervals whose upper boundary is at or before the horizon.
    # In particular, risk at time zero must be zero rather than the 3-month risk.
    k = int(np.searchsorted(upper, float(horizon), side="right"))
    return min(max(k, 0), J)


class MultitaskDiscreteHazardMLP(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_groups: int,
        n_bins: int,
        hidden: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(hidden) == 0:
            hidden = [32]
        num_groups = int(num_groups)
        n_bins = int(n_bins)
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        layers = []
        d = in_dim
        for h in hidden:
            layers += [
                torch.nn.Linear(d, h),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            d = h
        self.trunk = torch.nn.Sequential(*layers)
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(d, n_bins) for _ in range(num_groups)]
        )
        self.num_groups = num_groups
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        g = g.reshape(-1).long()
        if g.numel() != h.size(0):
            raise ValueError(
                f"Group index length {g.numel()} does not match batch size {h.size(0)}"
            )
        if torch.any(g < 0) or torch.any(g >= self.num_groups):
            raise ValueError("Group indices out of range for multitask hazard heads")
        all_logits = torch.stack([head(h) for head in self.heads], dim=1)
        pick = g[:, None, None].expand(-1, 1, self.n_bins)
        return all_logits.gather(1, pick).squeeze(1)


def fit_mt_discrete_hazard(
    Xtr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    g_tr: np.ndarray,
    Xva: np.ndarray | None,
    t_va: np.ndarray | None,
    e_va: np.ndarray | None,
    g_va: np.ndarray | None,
    *,
    num_groups: int,
    n_bins: int = DT_BINS,
    hidden=(64, 32),
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=SURV_NN_EPOCHS,
    patience=SURV_NN_PATIENCE,
    max_time: float = ADMIN_CENSOR_MONTHS,
    device: str = "cuda",
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    _, p = Xtr.shape
    bin_edges = _make_dt_bin_edges(t_tr, n_bins=n_bins, max_time=max_time)
    n_bins = int(max(1, len(bin_edges) - 1))
    ytr_np, mtr_np = _encode_discrete_time_targets(t_tr, e_tr, bin_edges)
    has_val = (
        Xva is not None
        and t_va is not None
        and e_va is not None
        and g_va is not None
        and len(t_va) > 0
    )
    if has_val:
        yva_np, mva_np = _encode_discrete_time_targets(t_va, e_va, bin_edges)
    else:
        yva_np = mva_np = None

    model = MultitaskDiscreteHazardMLP(
        p,
        num_groups=int(num_groups),
        n_bins=int(n_bins),
        hidden=list(hidden),
        dropout=dropout,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    gtr_t = torch.tensor(g_tr, dtype=torch.long, device=dev)
    ytr_t = torch.tensor(ytr_np, dtype=torch.float32, device=dev)
    mtr_t = torch.tensor(mtr_np, dtype=torch.float32, device=dev)

    if has_val:
        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
        gva_t = torch.tensor(g_va, dtype=torch.long, device=dev)
        yva_t = torch.tensor(yva_np, dtype=torch.float32, device=dev)
        mva_t = torch.tensor(mva_np, dtype=torch.float32, device=dev)
    else:
        Xva_t = gva_t = yva_t = mva_t = None

    best_state = None
    best_val = float("inf")
    best_train = float("inf")
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(Xtr_t, gtr_t)
        loss = grouped_discrete_time_loss(logits, ytr_t, mtr_t, gtr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if not has_val:
            train_loss = float(loss.detach().cpu().item())
            if train_loss + 1e-6 < best_train:
                best_train = train_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        elif ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                logits_va = model(Xva_t, gva_t)
                vloss = float(
                    grouped_discrete_time_loss(logits_va, yva_t, mva_t, gva_t).item()
                )
            if vloss + 1e-6 < best_val:
                best_val = vloss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, bin_edges


@torch.no_grad()
def predict_mt_discrete_hazard(
    model: torch.nn.Module,
    X: np.ndarray,
    g: np.ndarray,
    bin_edges: np.ndarray,
    *,
    horizon: float = POSTERIOR_H_MONTHS,
    device: str = "cuda",
) -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    gt = torch.tensor(g, dtype=torch.long, device=dev)
    logits = model(Xt, gt)
    haz = torch.sigmoid(logits).detach().cpu().numpy()
    k = _n_bins_for_horizon(bin_edges, horizon=float(horizon))
    if k == 0:
        return np.zeros(haz.shape[0], dtype=float)
    haz_use = np.clip(haz[:, :k], 1e-8, 1.0 - 1e-8)
    risk = -np.sum(np.log(1.0 - haz_use), axis=1)
    return risk.astype(float)



# ==========================================================================
# Paper-style fixed-horizon baselines: LR/RF/SVM and MTL-BNN
# ==========================================================================


def _standardize_fit_apply(Xfit: np.ndarray, *Xs: np.ndarray):
    Xfit = np.asarray(Xfit, dtype=float)
    mu = np.nanmean(Xfit, axis=0)
    sd = np.nanstd(Xfit, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where((np.isfinite(sd)) & (sd > 1e-8), sd, 1.0)
    outs = []
    for X in (Xfit,) + Xs:
        X = np.asarray(X, dtype=float)
        Z = (np.where(np.isfinite(X), X, mu) - mu) / sd
        Z = np.where(np.isfinite(Z), Z, 0.0)
        outs.append(Z.astype(np.float32))
    return tuple(outs)


def _split_bimodal_blocks(X: np.ndarray, feat_names: list[str]):
    """Split preprocessed features into omics-like and clinical-like blocks.

    The paper used RNA-seq + clinical modalities. Here the preprocessing outputs may
    be PathClin or PathUDP+MutTMB, so we treat pathway/UDP/mutation/TMB/latent columns
    as the omics block and clinical/treatment columns as the clinical block.
    """
    X = np.asarray(X, dtype=float)
    clin_idx = []
    for j, name in enumerate(feat_names):
        s = str(name).lower()
        is_clin = (
            s.startswith("clin")
            or "clincont::" in s
            or "clinbin::" in s
            or "tx_" in s
            or "treatment" in s
            or "stage" in s
            or "grade" in s
            or "ajcc" in s
            or s.endswith("::age")
            or s == "age"
        )
        if is_clin:
            clin_idx.append(j)
    clin_idx = np.asarray(sorted(set(clin_idx)), dtype=int)
    all_idx = np.arange(X.shape[1], dtype=int)
    omics_idx = np.asarray([j for j in all_idx if j not in set(clin_idx.tolist())], dtype=int)
    if omics_idx.size == 0:
        omics_idx = all_idx
        clin_idx = np.asarray([], dtype=int)
    return omics_idx, clin_idx


def _take_block(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if idx is None or len(idx) == 0:
        return np.zeros((X.shape[0], 1), dtype=np.float32)
    return X[:, idx].astype(np.float32)


class PaperFeatureExtractor(torch.nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 8, dropout: float = 0.05):
        super().__init__()
        in_dim = int(max(1, in_dim))
        emb_dim = int(max(2, emb_dim))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.Softsign(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.Softsign(),
        )

    def forward(self, x):
        return self.net(x)


class PaperBimodalBNN(torch.nn.Module):
    """Bimodal neural classifier following the NPJ paper's implementation pattern.

    It has separate feature extractors for omics-like and clinical-like blocks,
    intermediate fusion, optional task descriptor embedding, batch norm, Softsign,
    and a sigmoid/logit binary classifier.
    """

    def __init__(
        self,
        omics_dim: int,
        clinical_dim: int,
        *,
        num_tasks: int | None = None,
        emb_dim: int = 8,
        task_emb_dim: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.omics_fe = PaperFeatureExtractor(omics_dim, emb_dim=emb_dim, dropout=dropout)
        self.clinical_fe = PaperFeatureExtractor(clinical_dim, emb_dim=emb_dim, dropout=dropout)
        self.use_task = num_tasks is not None and int(num_tasks) > 1
        if self.use_task:
            self.task_embedding = torch.nn.Embedding(int(num_tasks), int(task_emb_dim))
            in_dim = 2 * int(emb_dim) + int(task_emb_dim)
        else:
            self.task_embedding = None
            in_dim = 2 * int(emb_dim)
        hidden = max(4, in_dim // 2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.Softsign(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, xo, xc, g=None):
        h = [self.omics_fe(xo), self.clinical_fe(xc)]
        if self.use_task:
            if g is None:
                raise ValueError("Task indices are required for multitask BNN.")
            h.append(self.task_embedding(g.reshape(-1).long()))
        z = torch.cat(h, dim=1)
        return self.classifier(z).reshape(-1)


def _make_binary_training_arrays(
    Xo: np.ndarray,
    Xc: np.ndarray,
    y: np.ndarray,
    g: np.ndarray | None = None,
    keep: np.ndarray | None = None,
):
    if keep is None:
        keep = np.ones(len(y), dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    Xo2 = np.asarray(Xo, dtype=np.float32)[keep]
    Xc2 = np.asarray(Xc, dtype=np.float32)[keep]
    y2 = np.asarray(y, dtype=np.float32).reshape(-1)[keep]
    if g is None:
        return Xo2, Xc2, y2, None
    return Xo2, Xc2, y2, np.asarray(g, dtype=int).reshape(-1)[keep]


def fit_paper_bimodal_bnn(
    Xo_fit: np.ndarray,
    Xc_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    g_fit: np.ndarray | None = None,
    Xo_val: np.ndarray | None = None,
    Xc_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    g_val: np.ndarray | None = None,
    num_tasks: int | None = None,
    mtl_weighted_sampler: bool = False,
    emb_dim: int = BNN_EMB_DIM,
    task_emb_dim: int = BNN_TASK_EMB_DIM,
    epochs: int = BNN_EPOCHS,
    patience: int = BNN_PATIENCE,
    lr: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = BNN_BATCH_SIZE,
    device: str = "cuda",
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Xo_fit = np.asarray(Xo_fit, dtype=np.float32)
    Xc_fit = np.asarray(Xc_fit, dtype=np.float32)
    y_fit = np.asarray(y_fit, dtype=np.float32).reshape(-1)
    if y_fit.size < 4 or len(np.unique(y_fit.astype(int))) < 2:
        raise ValueError("BNN binary training set has too few samples or one class.")

    model = PaperBimodalBNN(
        Xo_fit.shape[1],
        Xc_fit.shape[1],
        num_tasks=num_tasks,
        emb_dim=emb_dim,
        task_emb_dim=task_emb_dim,
        dropout=0.05,
    ).to(dev)

    xo_t = torch.tensor(Xo_fit, dtype=torch.float32)
    xc_t = torch.tensor(Xc_fit, dtype=torch.float32)
    y_t = torch.tensor(y_fit, dtype=torch.float32)
    if g_fit is None:
        g_t = torch.zeros(len(y_fit), dtype=torch.long)
    else:
        g_t = torch.tensor(np.asarray(g_fit, dtype=int), dtype=torch.long)
    ds = torch.utils.data.TensorDataset(xo_t, xc_t, y_t, g_t)

    if mtl_weighted_sampler and g_fit is not None:
        counts = pd.Series(np.asarray(g_fit, dtype=int)).value_counts().to_dict()
        # Paper states inverse square of task sample size.
        w = np.asarray([1.0 / max(float(counts[int(gi)]) ** 2, 1.0) for gi in g_fit], dtype=float)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(w, dtype=torch.double),
            num_samples=len(w),
            replacement=True,
        )
        drop_last = (len(ds) > int(batch_size)) and (len(ds) % int(batch_size) == 1)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=drop_last)
    else:
        drop_last = (len(ds) > int(batch_size)) and (len(ds) % int(batch_size) == 1)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    opt = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=float(momentum))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    has_val = (
        Xo_val is not None and Xc_val is not None and y_val is not None
        and len(y_val) >= 4 and len(np.unique(np.asarray(y_val, dtype=int))) >= 2
    )
    if has_val:
        Xo_val_t = torch.tensor(np.asarray(Xo_val, dtype=np.float32), dtype=torch.float32, device=dev)
        Xc_val_t = torch.tensor(np.asarray(Xc_val, dtype=np.float32), dtype=torch.float32, device=dev)
        y_val_np = np.asarray(y_val, dtype=int).reshape(-1)
        if g_val is None:
            g_val_t = torch.zeros(len(y_val_np), dtype=torch.long, device=dev)
        else:
            g_val_t = torch.tensor(np.asarray(g_val, dtype=int), dtype=torch.long, device=dev)
    else:
        Xo_val_t = Xc_val_t = g_val_t = None
        y_val_np = None

    best_state = None
    best_score = -np.inf
    bad = 0
    for ep in range(1, int(epochs) + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_n = 0
        for xb_o, xb_c, yb, gb in loader:
            xb_o = xb_o.to(dev)
            xb_c = xb_c.to(dev)
            yb = yb.to(dev)
            gb = gb.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(xb_o, xb_c, gb if model.use_task else None)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = int(yb.numel())
            epoch_loss_sum += float(loss.detach().cpu().item()) * max(bs, 1)
            epoch_n += max(bs, 1)

        if not has_val:
            # The published bimodal-network experiment used 50 epochs and the
            # final checkpoint.  Do not choose an epoch from held-out outcomes.
            pass
        elif ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xo_val_t, Xc_val_t, g_val_t if model.use_task else None)
                val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score
                score = float(average_precision_score(y_val_np, val_prob))
                if not np.isfinite(score):
                    score = float(roc_auc_score(y_val_np, val_prob))
            except Exception:
                score = -float(loss.item())
            if score > best_score + 1e-6:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= int(patience):
                break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_paper_bimodal_bnn(model: PaperBimodalBNN, Xo: np.ndarray, Xc: np.ndarray, g: np.ndarray | None = None, *, device: str = "cuda") -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    xo = torch.tensor(np.asarray(Xo, dtype=np.float32), dtype=torch.float32, device=dev)
    xc = torch.tensor(np.asarray(Xc, dtype=np.float32), dtype=torch.float32, device=dev)
    if g is None:
        gt = torch.zeros(xo.shape[0], dtype=torch.long, device=dev)
    else:
        gt = torch.tensor(np.asarray(g, dtype=int), dtype=torch.long, device=dev)
    logits = model(xo, xc, gt if model.use_task else None)
    return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).astype(float)


def _fit_predict_sklearn_binary(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=int).reshape(-1)
    if y_train.size < 4 or len(np.unique(y_train)) < 2:
        raise ValueError(f"{model_name} binary training set has too few samples or one class.")
    if model_name == "LR":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=LR_MAX_ITER, class_weight="balanced", solver="liblinear", random_state=int(seed)),
        )
    elif model_name == "RF":
        clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            class_weight="balanced_subsample",
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_jobs=-1,
            random_state=int(seed),
        )
    elif model_name == "SVM":
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, class_weight="balanced", probability=True, random_state=int(seed)),
        )
    else:
        raise ValueError(f"Unknown sklearn binary model: {model_name}")
    clf.fit(np.asarray(X_train, dtype=float), y_train)
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(np.asarray(X_test, dtype=float))[:, 1]
    else:
        score = clf.decision_function(np.asarray(X_test, dtype=float))
        prob = 1.0 / (1.0 + np.exp(-np.asarray(score, dtype=float)))
    return np.asarray(prob, dtype=float).reshape(-1)


# ==========================================================================
# CoxPH baselines
# ==========================================================================
def fit_coxph_lifelines(
    Xtr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    *,
    penalizer: float = COX_PENALIZER,
    var_thresh: float = 1e-10,
):
    Xtr = np.asarray(Xtr, dtype=float)
    t_tr = np.asarray(t_tr, dtype=float)
    e_tr = np.asarray(e_tr, dtype=int)

    cols_all = [f"x{j}" for j in range(Xtr.shape[1])]
    dfX = pd.DataFrame(Xtr, columns=cols_all)

    # Drop non-finite columns
    finite_mask = np.isfinite(dfX.to_numpy()).all(axis=0)
    keep_cols = [c for c, keep in zip(cols_all, finite_mask) if keep]

    dfX = dfX[keep_cols].copy()

    # Drop zero / near-zero variance columns on the training set
    var = dfX.var(axis=0, ddof=0)
    keep_cols = var.index[var > float(var_thresh)].tolist()
    dfX = dfX[keep_cols].copy()

    if len(keep_cols) == 0:
        raise ValueError("No usable CoxPH columns remain after variance filtering.")

    df = dfX.copy()
    df["time"] = t_tr
    df["event"] = e_tr

    cph = CoxPHFitter(penalizer=float(penalizer))
    cph.fit(df, duration_col="time", event_col="event", show_progress=False)
    return cph, keep_cols


def predict_coxph_lifelines(
    cph: CoxPHFitter,
    X: np.ndarray,
    cols: list[str],
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    all_cols = [f"x{j}" for j in range(X.shape[1])]
    df = pd.DataFrame(X, columns=all_cols)

    # Keep only training-time columns, in the same order
    df = df.reindex(columns=cols, fill_value=0.0)

    try:
        s = cph.predict_log_partial_hazard(df).to_numpy().reshape(-1)
    except Exception:
        s = np.log(
            np.maximum(cph.predict_partial_hazard(df).to_numpy().reshape(-1), 1e-12)
        )
    return s.astype(float)


def _fit_rsf(
    Xtr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    *,
    n_estimators: int = 300,
    min_samples_leaf: int = 15,
    min_samples_split: int = 10,
    max_features="sqrt",
    seed: int = 0,
):
    Xtr = np.asarray(Xtr, dtype=float)
    t_tr = np.asarray(t_tr, dtype=float)
    e_tr = np.asarray(e_tr, dtype=int)

    RSF_cls = None
    try:
        from lifelines import RandomSurvivalForestFitter as RSF_cls  # type: ignore
    except Exception:
        try:
            from lifelines.fitters import RandomSurvivalForestFitter as RSF_cls  # type: ignore
        except Exception:
            RSF_cls = None

    if RSF_cls is not None:
        feat_cols = [f"x{j}" for j in range(Xtr.shape[1])]
        df = pd.DataFrame(Xtr, columns=feat_cols)
        df["time"] = t_tr
        df["event"] = e_tr
        kwargs = {
            "n_estimators": int(n_estimators),
            "random_state": int(seed),
            "min_samples_leaf": int(min_samples_leaf),
            "min_samples_split": int(min_samples_split),
            "max_features": max_features,
        }
        try:
            sig = inspect.signature(RSF_cls)
            allowed = set(sig.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        except Exception:
            pass
        model = RSF_cls(**kwargs)
        model.fit(df, duration_col="time", event_col="event")
        return "lifelines", model, feat_cols

    try:
        from sksurv.ensemble import RandomSurvivalForest  # type: ignore
        from sksurv.util import Surv  # type: ignore

        y = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr.astype(float))
        model = RandomSurvivalForest(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            min_samples_split=int(min_samples_split),
            max_features=max_features,
            n_jobs=-1,
            random_state=int(seed),
        )
        model.fit(Xtr, y)
        return "sksurv", model, None
    except Exception as ex:
        raise ImportError(
            "RSF baseline requested, but no RSF backend is available. "
            "Install lifelines with RandomSurvivalForestFitter or install scikit-survival."
        ) from ex


def _integrate_survival(
    time_grid: np.ndarray, surv: np.ndarray, horizon: float
) -> float:
    t = np.asarray(time_grid, dtype=float)
    s = np.asarray(surv, dtype=float)
    if t.size == 0 or s.size == 0 or t.size != s.size:
        return float("nan")
    order = np.argsort(t)
    t = t[order]
    s = s[order]
    H = float(horizon)
    m = (t >= 0.0) & (t <= H)
    t = t[m]
    s = s[m]
    if t.size < 2:
        return float("nan")
    if t[0] > 0.0:
        t = np.concatenate([[0.0], t])
        s = np.concatenate([[1.0], s])
    if t[-1] < H:
        t = np.concatenate([t, [H]])
        s = np.concatenate([s, [s[-1]]])
    return float(np.trapz(s, t))


def _predict_rsf_risk(
    backend: str,
    model,
    X: np.ndarray,
    feat_cols=None,
    horizon: float = ADMIN_CENSOR_MONTHS,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if backend == "lifelines":
        if feat_cols is None:
            feat_cols = [f"x{j}" for j in range(X.shape[1])]
        dfX = pd.DataFrame(X, columns=list(feat_cols))
        sf = model.predict_survival_function(dfX)
        if isinstance(sf, pd.DataFrame):
            t = sf.index.to_numpy(dtype=float)
            out = []
            for j in range(sf.shape[1]):
                s = sf.iloc[:, j].to_numpy(dtype=float)
                out.append(-_integrate_survival(t, s, horizon=horizon))
            return np.asarray(out, dtype=float)
        out = []
        for sfi in sf:
            if isinstance(sfi, pd.Series):
                t = sfi.index.to_numpy(dtype=float)
                s = sfi.to_numpy(dtype=float)
                out.append(-_integrate_survival(t, s, horizon=horizon))
            else:
                out.append(float("nan"))
        return np.asarray(out, dtype=float)

    if backend == "sksurv":
        sf_list = model.predict_survival_function(X.astype(float), return_array=False)
        out = []
        for sfi in sf_list:
            try:
                t = np.asarray(sfi.x, dtype=float)
                s = np.asarray(sfi.y, dtype=float)
            except Exception:
                out.append(float("nan"))
                continue
            out.append(-_integrate_survival(t, s, horizon=horizon))
        return np.asarray(out, dtype=float)

    raise ValueError(f"Unknown RSF backend: {backend}")


# ==========================================================================


def _breslow_baseline_cumulative_hazard(
    log_risk: np.ndarray, times: np.ndarray, events: np.ndarray
):
    log_risk = np.asarray(log_risk, dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)

    event_times = np.sort(np.unique(times[events == 1]))
    if event_times.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    exp_lp = np.exp(np.clip(log_risk, -50.0, 50.0))
    cumhaz = []
    running = 0.0
    for tj in event_times:
        d_j = float(np.sum((times == tj) & (events == 1)))
        risk_set = times >= tj
        denom = float(np.sum(exp_lp[risk_set]))
        inc = 0.0 if denom <= 0.0 else d_j / denom
        running += inc
        cumhaz.append(running)
    return event_times.astype(float), np.asarray(cumhaz, dtype=float)


def _step_value_at_time(event_times: np.ndarray, values: np.ndarray, horizon: float) -> float:
    event_times = np.asarray(event_times, dtype=float)
    values = np.asarray(values, dtype=float)
    if event_times.size == 0 or values.size == 0:
        return 0.0
    idx = int(np.searchsorted(event_times, float(horizon), side="right") - 1)
    if idx < 0:
        return 0.0
    return float(values[min(idx, values.size - 1)])


def _ph_survival_from_log_risk(
    train_log_risk: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    pred_log_risk: np.ndarray,
    horizon: float,
) -> np.ndarray:
    evt_times, base_cumhaz = _breslow_baseline_cumulative_hazard(
        train_log_risk, t_train, e_train
    )
    H0 = _step_value_at_time(evt_times, base_cumhaz, float(horizon))
    pred_log_risk = np.asarray(pred_log_risk, dtype=float).reshape(-1)
    surv = np.exp(-H0 * np.exp(np.clip(pred_log_risk, -50.0, 50.0)))
    return np.clip(surv, 1e-12, 1.0)


def predict_deepsurv_risk_at_horizon(
    model: torch.nn.Module,
    X_train: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    X: np.ndarray,
    *,
    horizon: float,
    device: str = "cuda",
) -> np.ndarray:
    train_lp = predict_deepsurv(model, X_train, device=device)
    pred_lp = predict_deepsurv(model, X, device=device)
    surv = _ph_survival_from_log_risk(train_lp, t_train, e_train, pred_lp, horizon)
    return (1.0 - surv).astype(float)


def _ph_survival_from_log_risk_by_group(
    train_log_risk: np.ndarray,
    g_train: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    pred_log_risk: np.ndarray,
    g_pred: np.ndarray,
    horizon: float,
) -> np.ndarray:
    train_log_risk = np.asarray(train_log_risk, dtype=float).reshape(-1)
    g_train = np.asarray(g_train, dtype=int).reshape(-1)
    t_train = np.asarray(t_train, dtype=float).reshape(-1)
    e_train = np.asarray(e_train, dtype=int).reshape(-1)
    pred_log_risk = np.asarray(pred_log_risk, dtype=float).reshape(-1)
    g_pred = np.asarray(g_pred, dtype=int).reshape(-1)

    if not (
        train_log_risk.size == g_train.size == t_train.size == e_train.size
        and pred_log_risk.size == g_pred.size
    ):
        raise ValueError("Grouped PH survival inputs have inconsistent lengths.")

    out = np.ones(pred_log_risk.shape[0], dtype=float)
    all_groups = np.unique(np.concatenate([g_train, g_pred]))
    for gi in all_groups:
        m_tr = g_train == gi
        m_pr = g_pred == gi
        if not np.any(m_pr):
            continue
        if not np.any(m_tr):
            out[m_pr] = 1.0
            continue
        evt_times, base_cumhaz = _breslow_baseline_cumulative_hazard(
            train_log_risk[m_tr], t_train[m_tr], e_train[m_tr]
        )
        H0 = _step_value_at_time(evt_times, base_cumhaz, float(horizon))
        out[m_pr] = np.exp(
            -H0 * np.exp(np.clip(pred_log_risk[m_pr], -50.0, 50.0))
        )
    return np.clip(out, 1e-12, 1.0)



def predict_mtdeepsurv_risk_at_horizon(
    model: torch.nn.Module,
    X_train: np.ndarray,
    g_train: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    X: np.ndarray,
    g: np.ndarray,
    *,
    horizon: float,
    device: str = "cuda",
) -> np.ndarray:
    train_lp = predict_mtdeepsurv(model, X_train, g_train, device=device)
    pred_lp = predict_mtdeepsurv(model, X, g, device=device)
    surv = _ph_survival_from_log_risk_by_group(
        train_lp, g_train, t_train, e_train, pred_lp, g, horizon
    )
    return (1.0 - surv).astype(float)


def predict_coxph_risk_at_horizon(
    cph: CoxPHFitter,
    X: np.ndarray,
    cols: list[str],
    *,
    horizon: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    all_cols = [f"x{j}" for j in range(X.shape[1])]
    df = pd.DataFrame(X, columns=all_cols).reindex(columns=cols, fill_value=0.0)
    sf = cph.predict_survival_function(df, times=[float(horizon)])
    if isinstance(sf, pd.DataFrame):
        surv = sf.iloc[0].to_numpy(dtype=float)
    else:
        surv = np.asarray([float(s.iloc[0]) for s in sf], dtype=float)
    return np.clip(1.0 - surv, 0.0, 1.0)


def _step_survival_at_horizon(time_grid: np.ndarray, surv: np.ndarray, horizon: float) -> float:
    t = np.asarray(time_grid, dtype=float)
    s = np.asarray(surv, dtype=float)
    if t.size == 0 or s.size == 0 or t.size != s.size:
        return float("nan")
    order = np.argsort(t)
    t = t[order]
    s = s[order]
    idx = int(np.searchsorted(t, float(horizon), side="right") - 1)
    if idx < 0:
        return 1.0
    return float(np.clip(s[min(idx, s.size - 1)], 0.0, 1.0))


def _predict_rsf_risk_at_horizon(
    backend: str,
    model,
    X: np.ndarray,
    feat_cols=None,
    *,
    horizon: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = []
    if backend == "lifelines":
        if feat_cols is None:
            feat_cols = [f"x{j}" for j in range(X.shape[1])]
        dfX = pd.DataFrame(X, columns=list(feat_cols))
        sf = model.predict_survival_function(dfX)
        if isinstance(sf, pd.DataFrame):
            t = sf.index.to_numpy(dtype=float)
            for j in range(sf.shape[1]):
                s = sf.iloc[:, j].to_numpy(dtype=float)
                out.append(1.0 - _step_survival_at_horizon(t, s, horizon))
        else:
            for sfi in sf:
                if isinstance(sfi, pd.Series):
                    t = sfi.index.to_numpy(dtype=float)
                    s = sfi.to_numpy(dtype=float)
                    out.append(1.0 - _step_survival_at_horizon(t, s, horizon))
                else:
                    out.append(float("nan"))
        return np.asarray(out, dtype=float)

    if backend == "sksurv":
        sf_list = model.predict_survival_function(X.astype(float), return_array=False)
        for sfi in sf_list:
            try:
                t = np.asarray(sfi.x, dtype=float)
                s = np.asarray(sfi.y, dtype=float)
                out.append(1.0 - _step_survival_at_horizon(t, s, horizon))
            except Exception:
                out.append(float("nan"))
        return np.asarray(out, dtype=float)

    raise ValueError(f"Unknown RSF backend: {backend}")


@torch.no_grad()
def predict_mt_discrete_hazard_risk_at_horizon(
    model: torch.nn.Module,
    X: np.ndarray,
    g: np.ndarray,
    bin_edges: np.ndarray,
    *,
    horizon: float,
    device: str = "cuda",
) -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    gt = torch.tensor(g, dtype=torch.long, device=dev)
    logits = model(Xt, gt)
    haz = torch.sigmoid(logits).detach().cpu().numpy()
    k = _n_bins_for_horizon(bin_edges, horizon=float(horizon))
    if k == 0:
        return np.zeros(haz.shape[0], dtype=float)
    haz_use = np.clip(haz[:, :k], 1e-8, 1.0 - 1e-8)
    surv = np.prod(1.0 - haz_use, axis=1)
    return np.clip(1.0 - surv, 0.0, 1.0).astype(float)


def _somers_dxy_from_c(c: float) -> float:
    return float(2.0 * c - 1.0) if np.isfinite(c) else float("nan")


def _binary_event_by_horizon(times: np.ndarray, events: np.ndarray, horizon: float):
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    pos = (events == 1) & (times <= float(horizon))
    neg = times > float(horizon)
    keep = pos | neg
    y = pos.astype(int)
    return keep, y


def _km_censor_survival(times: np.ndarray, events: np.ndarray):
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    cens = (events == 0).astype(int)
    event_times = np.sort(np.unique(times[cens == 1]))
    if event_times.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    surv = []
    s = 1.0
    for tj in event_times:
        at_risk = np.sum(times >= tj)
        d = np.sum((times == tj) & (cens == 1))
        if at_risk > 0:
            s *= (1.0 - float(d) / float(at_risk))
        surv.append(max(float(s), 1e-8))
    return event_times.astype(float), np.asarray(surv, dtype=float)


def _step_survival_at(event_times: np.ndarray, surv_values: np.ndarray, t: float, *, left: bool = False) -> float:
    event_times = np.asarray(event_times, dtype=float)
    surv_values = np.asarray(surv_values, dtype=float)
    if event_times.size == 0 or surv_values.size == 0:
        return 1.0
    side = "left" if left else "right"
    idx = int(np.searchsorted(event_times, float(t), side=side) - 1)
    if idx < 0:
        return 1.0
    return float(max(surv_values[min(idx, surv_values.size - 1)], 1e-8))


def _ipcw_brier_at_horizon(
    times: np.ndarray,
    events: np.ndarray,
    risk_prob: np.ndarray,
    horizon: float,
    *,
    censor_times: np.ndarray,
    censor_events: np.ndarray,
) -> float:
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    risk_prob = np.clip(np.asarray(risk_prob, dtype=float).reshape(-1), 0.0, 1.0)
    if not (times.size == events.size == risk_prob.size) or times.size == 0:
        return float("nan")
    # Estimate the censoring distribution from outer-training patients only.
    g_times, g_surv = _km_censor_survival(censor_times, censor_events)
    n = float(times.size)
    total = 0.0
    for ti, ei, pi in zip(times, events, risk_prob):
        if ei == 1 and ti <= float(horizon):
            g = _step_survival_at(g_times, g_surv, ti, left=True)
            total += ((1.0 - pi) ** 2) / max(g, 1e-8)
        elif ti > float(horizon):
            g = _step_survival_at(g_times, g_surv, float(horizon), left=False)
            total += (pi ** 2) / max(g, 1e-8)
    return float(total / n)


def _binary_auc_metrics(
    times: np.ndarray,
    events: np.ndarray,
    risk_prob: np.ndarray,
    horizon: float,
    *,
    censor_times: np.ndarray,
    censor_events: np.ndarray,
):
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    risk_prob = np.asarray(risk_prob, dtype=float).reshape(-1)
    keep, y_all = _binary_event_by_horizon(times, events, float(horizon))
    keep = np.asarray(keep, bool) & np.isfinite(risk_prob)
    y = np.asarray(y_all, dtype=int)[keep]
    s = risk_prob[keep]
    if y.size < 3 or int(np.sum(y == 1)) < 2 or int(np.sum(y == 0)) < 2:
        return float("nan"), float("nan")
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        g_times, g_surv = _km_censor_survival(censor_times, censor_events)
        t_keep = times[keep]
        e_keep = events[keep]
        pos = (e_keep == 1) & (t_keep <= float(horizon))
        neg = t_keep > float(horizon)
        w = np.zeros(y.size, dtype=float)
        for i in np.flatnonzero(pos):
            w[i] = 1.0 / max(
                _step_survival_at(g_times, g_surv, t_keep[i], left=True), 1e-8
            )
        g_h = max(
            _step_survival_at(g_times, g_surv, float(horizon), left=False), 1e-8
        )
        w[neg] = 1.0 / g_h
        if not np.isfinite(w).all() or float(np.sum(w)) <= 0.0:
            return float("nan"), float("nan")
        auroc = float(roc_auc_score(y, s, sample_weight=w))
        auprc = float(average_precision_score(y, s, sample_weight=w))
        return auroc, auprc
    except Exception:
        return float("nan"), float("nan")


def _uno_ipcw_cindex(
    times: np.ndarray,
    events: np.ndarray,
    risk: np.ndarray,
    *,
    tau: float,
    censor_times: np.ndarray,
    censor_events: np.ndarray,
) -> float:
    """Uno's IPCW concordance using the outer-training censoring distribution."""
    t = np.asarray(times, dtype=float).reshape(-1)
    e = np.asarray(events, dtype=int).reshape(-1)
    r = np.asarray(risk, dtype=float).reshape(-1)
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    t, e, r = t[ok], e[ok], r[ok]
    if t.size < 3:
        return float("nan")
    g_times, g_surv = _km_censor_survival(censor_times, censor_events)
    num = 0.0
    den = 0.0
    for i in np.flatnonzero((e == 1) & (t <= float(tau))):
        comparable = t > t[i]
        n_comp = int(np.sum(comparable))
        if n_comp == 0:
            continue
        g_i = max(_step_survival_at(g_times, g_surv, t[i], left=True), 1e-8)
        w_i = 1.0 / (g_i * g_i)
        diff = r[i] - r[comparable]
        concordant = float(np.sum(diff > 0.0) + 0.5 * np.sum(diff == 0.0))
        num += w_i * concordant
        den += w_i * n_comp
    return float(num / den) if den > 0.0 else float("nan")


def _make_ibs_grid(horizon: float, n_points: int = 21) -> np.ndarray:
    H = float(horizon)
    if H <= 0.0:
        return np.asarray([0.0], dtype=float)
    n_points = int(max(5, n_points))
    return np.linspace(0.0, H, n_points, dtype=float)


def _ibs_from_risk_predictions(
    times: np.ndarray,
    events: np.ndarray,
    grid: np.ndarray,
    risks: np.ndarray,
    *,
    censor_times: np.ndarray,
    censor_events: np.ndarray,
) -> float:
    grid = np.asarray(grid, dtype=float).reshape(-1)
    risks = np.asarray(risks, dtype=float)
    if grid.size == 0 or risks.size == 0 or risks.shape[0] != grid.size:
        return float("nan")
    bs = []
    for j, h in enumerate(grid):
        if h <= 0.0:
            bs.append(0.0)
        else:
            bs.append(
                _ipcw_brier_at_horizon(
                    times,
                    events,
                    risks[j],
                    float(h),
                    censor_times=censor_times,
                    censor_events=censor_events,
                )
            )
    bs = np.asarray(bs, dtype=float)
    if grid.size < 2 or float(grid[-1]) <= 0.0 or not np.isfinite(bs).any():
        return float("nan")
    return float(np.trapz(bs, grid) / float(grid[-1]))


def _evaluate_horizon_metrics(
    times: np.ndarray,
    events: np.ndarray,
    risk_prob: np.ndarray,
    horizon: float,
    ibs_grid: np.ndarray | None = None,
    ibs_risks: np.ndarray | None = None,
    *,
    censor_times: np.ndarray,
    censor_events: np.ndarray,
) -> dict:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    events_arr = np.asarray(events, dtype=int).reshape(-1)
    c_test = eval_cindex(times, events, risk_prob)
    uno_c = _uno_ipcw_cindex(
        times,
        events,
        risk_prob,
        tau=float(horizon),
        censor_times=censor_times,
        censor_events=censor_events,
    )
    somers = _somers_dxy_from_c(c_test)
    brier = _ipcw_brier_at_horizon(
        times,
        events,
        risk_prob,
        float(horizon),
        censor_times=censor_times,
        censor_events=censor_events,
    )
    auroc, auprc = _binary_auc_metrics(
        times,
        events,
        risk_prob,
        float(horizon),
        censor_times=censor_times,
        censor_events=censor_events,
    )
    ibs = (
        _ibs_from_risk_predictions(
            times,
            events,
            ibs_grid,
            ibs_risks,
            censor_times=censor_times,
            censor_events=censor_events,
        )
        if (ibs_grid is not None and ibs_risks is not None)
        else float("nan")
    )
    return {
        "c_test": float(c_test) if np.isfinite(c_test) else float("nan"),
        "uno_c_ipcw": float(uno_c) if np.isfinite(uno_c) else float("nan"),
        "somers_dxy_test": float(somers) if np.isfinite(somers) else float("nan"),
        "brier_at_H": float(brier) if np.isfinite(brier) else float("nan"),
        "ibs_up_to_H": float(ibs) if np.isfinite(ibs) else float("nan"),
        "auroc": float(auroc) if np.isfinite(auroc) else float("nan"),
        "auprc": float(auprc) if np.isfinite(auprc) else float("nan"),
        "n_eval": int(times_arr.size),
        "n_events_total": int(np.sum(events_arr == 1)),
        "n_early_events_by_H": int(
            np.sum((events_arr == 1) & (times_arr <= float(horizon)))
        ),
        "n_event_free_at_H": int(np.sum(times_arr > float(horizon))),
        "n_censored_before_H": int(
            np.sum((events_arr == 0) & (times_arr <= float(horizon)))
        ),
    }


def _print_metric_line(model_name: str, train_mode: str, eval_group: str, horizon: float, metrics: dict):
    print(
        f"[{model_name}][{train_mode}][{eval_group}] H={int(float(horizon)):>3d} | "
        f"c={metrics.get('c_test', float('nan')):.4f} | "
        f"uno={metrics.get('uno_c_ipcw', float('nan')):.4f} | "
        f"dxy={metrics.get('somers_dxy_test', float('nan')):.4f} | "
        f"brier={metrics.get('brier_at_H', float('nan')):.4f} | "
        f"ibs={metrics.get('ibs_up_to_H', float('nan')):.4f} | "
        f"auroc={metrics.get('auroc', float('nan')):.4f} | "
        f"auprc={metrics.get('auprc', float('nan')):.4f}"
    )


def _append_metric_row(out_rows: list[dict], *, seed: int, model: str, train_mode: str, train_group: str, eval_group: str, horizon: float, metrics: dict, prep_dir: str, error: str | None = None):
    row = {
        "seed": int(seed),
        "input_config": str(CONFIG_NAME),
        "model": str(model),
        "train_mode": str(train_mode),
        "train_group": str(train_group),
        "eval_group": str(eval_group),
        "eval_horizon_months": float(horizon),
        "c_test": metrics.get("c_test", float("nan")),
        "uno_c_ipcw": metrics.get("uno_c_ipcw", float("nan")),
        "somers_dxy_test": metrics.get("somers_dxy_test", float("nan")),
        "brier_at_H": metrics.get("brier_at_H", float("nan")),
        "ibs_up_to_H": metrics.get("ibs_up_to_H", float("nan")),
        "auroc": metrics.get("auroc", float("nan")),
        "auprc": metrics.get("auprc", float("nan")),
        "n_eval": metrics.get("n_eval", float("nan")),
        "n_events_total": metrics.get("n_events_total", float("nan")),
        "n_early_events_by_H": metrics.get("n_early_events_by_H", float("nan")),
        "n_event_free_at_H": metrics.get("n_event_free_at_H", float("nan")),
        "n_censored_before_H": metrics.get("n_censored_before_H", float("nan")),
        "prep_dir": prep_dir,
    }
    if error is not None:
        row["error"] = str(error)
    out_rows.append(row)




def run_baselines_one_seed(seed: int, *, device: str = "cuda"):
    set_all_seeds(seed)
    data = load_inputs(seed=seed)

    Xtr = data["Xtr"]
    Xte = data["Xte"]
    t_tr, e_tr = admin_censor_np(data["t_tr"], data["e_tr"], max_mo=ADMIN_CENSOR_MONTHS)
    t_te, e_te = admin_censor_np(data["t_te"], data["e_te"], max_mo=ADMIN_CENSOR_MONTHS)
    g_tr = data["g_tr"]
    g_te = data["g_te"]
    groups = data["groups"]
    num_groups = int(data["num_groups"])

    if USE_INNER_VAL_FOR_NEURAL:
        fit_idx, val_idx = make_inner_val_split(g_tr, e_tr, val_frac=INNER_VAL_FRAC, seed=seed)
    else:
        fit_idx = np.arange(len(g_tr), dtype=int)
        val_idx = np.asarray([], dtype=int)
    Xfit, t_fit, e_fit, g_fit = Xtr[fit_idx], t_tr[fit_idx], e_tr[fit_idx], g_tr[fit_idx]
    has_val = bool(USE_INNER_VAL_FOR_NEURAL and len(val_idx) > 0)
    if has_val:
        Xval, t_val, e_val, g_val = Xtr[val_idx], t_tr[val_idx], e_tr[val_idx], g_tr[val_idx]
    else:
        Xval = t_val = e_val = g_val = None
    protocol = "training-only validation early stopping" if has_val else "full train split with fixed pre-specified epochs"
    print(f"[Training protocol] neural baselines use {protocol}")

    out_rows = []
    prediction_rows = []
    prep_dir = data.get("prep_dir", str(PREP_OUTDIR))

    def _record_patient_predictions(
        model_name: str,
        train_mode: str,
        train_group: str,
        eval_group: str,
        horizon: float,
        risks_eval: np.ndarray,
    ):
        if str(eval_group) != "ALL":
            return
        risks_eval = np.asarray(risks_eval, dtype=float).reshape(-1)
        if risks_eval.size != len(t_te):
            raise ValueError(
                f"Pooled prediction length mismatch for {model_name}: "
                f"{risks_eval.size} versus {len(t_te)}"
            )
        for i in range(len(t_te)):
            prediction_rows.append(
                {
                    "seed": int(seed),
                    "input_config": str(CONFIG_NAME),
                    "model": str(model_name),
                    "train_mode": str(train_mode),
                    "train_group": str(train_group),
                    "eval_group": "ALL",
                    "eval_horizon_months": float(horizon),
                    "SAMPLE_KEY": str(data["sample_key_te"][i]),
                    "GROUP_LABEL": str(data["group_label_te"][i]),
                    "PFI_TIME_MONTHS": float(t_te[i]),
                    "PFI_EVENT": int(e_te[i]),
                    "predicted_event_probability": float(risks_eval[i]),
                }
            )

    def _censor_reference(eval_group: str):
        if str(eval_group) == "ALL":
            return t_tr, e_tr
        if str(eval_group) not in groups:
            raise KeyError(f"Unknown evaluation group for censoring reference: {eval_group}")
        gi = groups.index(str(eval_group))
        m = g_tr == gi
        return t_tr[m], e_tr[m]

    def _record_model_metrics(model_name: str, train_mode: str, train_group: str, eval_group: str,
                              times_eval: np.ndarray, events_eval: np.ndarray, risks_eval: np.ndarray,
                              horizon: float, ibs_grid: np.ndarray, ibs_risks_eval: np.ndarray):
        censor_times, censor_events = _censor_reference(eval_group)
        metrics = _evaluate_horizon_metrics(
            times_eval,
            events_eval,
            risks_eval,
            float(horizon),
            ibs_grid=ibs_grid,
            ibs_risks=ibs_risks_eval,
            censor_times=censor_times,
            censor_events=censor_events,
        )
        _append_metric_row(
            out_rows,
            seed=seed,
            model=model_name,
            train_mode=train_mode,
            train_group=train_group,
            eval_group=eval_group,
            horizon=float(horizon),
            metrics=metrics,
            prep_dir=prep_dir,
        )
        _record_patient_predictions(
            model_name,
            train_mode,
            train_group,
            eval_group,
            float(horizon),
            risks_eval,
        )
        _print_metric_line(model_name, train_mode, eval_group, float(horizon), metrics)

    def _append_error_rows(model_name: str, train_mode: str, train_group: str, eval_groups: list[str], ex: Exception):
        err = f"{type(ex).__name__}: {ex}"
        print(f"[ERROR][{model_name}][{train_mode}][{train_group}] {err}")
        for horizon in EVAL_HORIZONS_MONTHS:
            nan_metrics = {
                "c_test": float("nan"),
                "uno_c_ipcw": float("nan"),
                "somers_dxy_test": float("nan"),
                "brier_at_H": float("nan"),
                "ibs_up_to_H": float("nan"),
                "auroc": float("nan"),
                "auprc": float("nan"),
            }
            for eg in eval_groups:
                _append_metric_row(
                    out_rows,
                    seed=seed,
                    model=model_name,
                    train_mode=train_mode,
                    train_group=train_group,
                    eval_group=eg,
                    horizon=float(horizon),
                    metrics=nan_metrics,
                    prep_dir=prep_dir,
                    error=err,
                )

    def _record_fixed_horizon_metrics(model_name: str, train_mode: str, train_group: str, eval_group: str,
                                      times_eval: np.ndarray, events_eval: np.ndarray, risks_eval: np.ndarray,
                                      horizon: float):
        # A fixed-horizon classifier does not supply a coherent survival curve.
        # Brier score at H is valid; IBS is deliberately left non-estimable.
        censor_times, censor_events = _censor_reference(eval_group)
        metrics = _evaluate_horizon_metrics(
            times_eval,
            events_eval,
            risks_eval,
            float(horizon),
            ibs_grid=None,
            ibs_risks=None,
            censor_times=censor_times,
            censor_events=censor_events,
        )
        _append_metric_row(
            out_rows,
            seed=seed,
            model=model_name,
            train_mode=train_mode,
            train_group=train_group,
            eval_group=eval_group,
            horizon=float(horizon),
            metrics=metrics,
            prep_dir=prep_dir,
        )
        _record_patient_predictions(
            model_name,
            train_mode,
            train_group,
            eval_group,
            float(horizon),
            risks_eval,
        )
        _print_metric_line(model_name, train_mode, eval_group, float(horizon), metrics)

    def _append_fixed_error_row(model_name: str, train_mode: str, train_group: str, eval_group: str, horizon: float, ex: Exception):
        err = f"{type(ex).__name__}: {ex}"
        print(f"[ERROR][{model_name}][{train_mode}][{eval_group}] H={float(horizon):.1f} {err}")
        nan_metrics = {
            "c_test": float("nan"),
            "uno_c_ipcw": float("nan"),
            "somers_dxy_test": float("nan"),
            "brier_at_H": float("nan"),
            "ibs_up_to_H": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
        }
        _append_metric_row(
            out_rows,
            seed=seed,
            model=model_name,
            train_mode=train_mode,
            train_group=train_group,
            eval_group=eval_group,
            horizon=float(horizon),
            metrics=nan_metrics,
            prep_dir=prep_dir,
            error=err,
        )

    # ---------------------- PAPER-STYLE FIXED-HORIZON MODELS ----------------------
    # These implement the comparisons from the NPJ MTL paper under our PFI endpoint:
    # LR/RF/SVM are fitted separately by cancer; MTL-BNN is fitted pooled
    # with hard parameter sharing and a task descriptor embedding.
    if RUN_PAPER_BINARY_BASELINES:
        feat_names = data.get("feat_names", [f"x{j}" for j in range(Xtr.shape[1])])
        omics_idx, clinical_idx = _split_bimodal_blocks(Xtr, feat_names)
        print(f"[Paper-style BNN] omics-like features={len(omics_idx)} | clinical-like features={len(clinical_idx)}")

        for horizon in EVAL_HORIZONS_MONTHS:
            keep_tr, y_tr_bin = _binary_event_by_horizon(t_tr, e_tr, float(horizon))
            keep_te, y_te_bin = _binary_event_by_horizon(t_te, e_te, float(horizon))
            n_pos = int(np.sum(y_tr_bin[keep_tr] == 1))
            n_neg = int(np.sum(y_tr_bin[keep_tr] == 0))
            print(f"[Paper-style binary] seed={seed} H={float(horizon):.1f} train usable={int(np.sum(keep_tr))} pos={n_pos} neg={n_neg}")
            stl_all_risks = {
                mdl: np.full(len(t_te), np.nan, dtype=float)
                for mdl in ("LR", "RF", "SVM", "STL-BNN")
            }

            # Single-task paper baselines: per-cohort LR/RF/SVM.
            for gi, gname in enumerate(groups):
                mtr = (g_tr == gi) & keep_tr
                mte = (g_te == gi)
                if int(np.sum(mte)) == 0:
                    continue

                # LR/RF/SVM use the whole preprocessed design matrix.
                for mdl in ("LR", "RF", "SVM"):
                    try:
                        risk = _fit_predict_sklearn_binary(
                            mdl,
                            Xtr[mtr],
                            y_tr_bin[mtr],
                            Xte[mte],
                            seed=seed + 1000 + gi,
                        )
                        stl_all_risks[mdl][mte] = risk
                        _record_fixed_horizon_metrics(
                            mdl,
                            "paper_stl_per_cohort",
                            gname,
                            gname,
                            t_te[mte],
                            e_te[mte],
                            risk,
                            float(horizon),
                        )
                    except Exception as ex:
                        _append_fixed_error_row(mdl, "paper_stl_per_cohort", gname, gname, float(horizon), ex)

                # Single-task bimodal neural network, fitted separately by cancer.
                try:
                    stl_pool_idx = np.flatnonzero(mtr)
                    if USE_INNER_VAL_FOR_NEURAL:
                        rel_fit_idx, rel_val_idx = make_inner_val_split(
                            np.zeros(len(stl_pool_idx), dtype=int),
                            y_tr_bin[stl_pool_idx],
                            val_frac=INNER_VAL_FRAC,
                            seed=seed + int(round(float(horizon) * 10.0)) + 2000 + gi,
                        )
                        stl_fit_idx = stl_pool_idx[rel_fit_idx]
                        stl_val_idx = stl_pool_idx[rel_val_idx]
                    else:
                        stl_fit_idx = stl_pool_idx
                        stl_val_idx = np.asarray([], dtype=int)

                    xo_fit_raw = _take_block(Xtr[stl_fit_idx], omics_idx)
                    xc_fit_raw = _take_block(Xtr[stl_fit_idx], clinical_idx)
                    xo_test_raw = _take_block(Xte[mte], omics_idx)
                    xc_test_raw = _take_block(Xte[mte], clinical_idx)
                    if len(stl_val_idx) > 0:
                        xo_val_raw = _take_block(Xtr[stl_val_idx], omics_idx)
                        xc_val_raw = _take_block(Xtr[stl_val_idx], clinical_idx)
                        xo_fit, xo_val, xo_test = _standardize_fit_apply(
                            xo_fit_raw, xo_val_raw, xo_test_raw
                        )
                        xc_fit, xc_val, xc_test = _standardize_fit_apply(
                            xc_fit_raw, xc_val_raw, xc_test_raw
                        )
                        y_val_stl = y_tr_bin[stl_val_idx]
                    else:
                        xo_fit, xo_test = _standardize_fit_apply(
                            xo_fit_raw, xo_test_raw
                        )
                        xc_fit, xc_test = _standardize_fit_apply(
                            xc_fit_raw, xc_test_raw
                        )
                        xo_val = xc_val = y_val_stl = None

                    set_all_seeds(seed + 2000 + gi)
                    stl_bnn = fit_paper_bimodal_bnn(
                        xo_fit,
                        xc_fit,
                        y_tr_bin[stl_fit_idx],
                        Xo_val=xo_val,
                        Xc_val=xc_val,
                        y_val=y_val_stl,
                        num_tasks=None,
                        mtl_weighted_sampler=False,
                        device=device,
                    )
                    stl_risk = predict_paper_bimodal_bnn(
                        stl_bnn, xo_test, xc_test, device=device
                    )
                    stl_all_risks["STL-BNN"][mte] = stl_risk
                    _record_fixed_horizon_metrics(
                        "STL-BNN",
                        "paper_stl_per_cohort",
                        gname,
                        gname,
                        t_te[mte],
                        e_te[mte],
                        stl_risk,
                        float(horizon),
                    )
                except Exception as ex:
                    _append_fixed_error_row(
                        "STL-BNN",
                        "paper_stl_per_cohort",
                        gname,
                        gname,
                        float(horizon),
                        ex,
                    )

            # Pooled evaluation of the three separately fitted cancer models.
            # Require predictions for every held-out patient; never pool a
            # selectively successful subset of cancers.
            for mdl, pooled_risk in stl_all_risks.items():
                if np.isfinite(pooled_risk).all():
                    _record_fixed_horizon_metrics(
                        mdl,
                        "paper_stl_per_cohort",
                        "SEPARATE_CANCER_MODELS",
                        "ALL",
                        t_te,
                        e_te,
                        pooled_risk,
                        float(horizon),
                    )
                else:
                    _append_fixed_error_row(
                        mdl,
                        "paper_stl_per_cohort",
                        "SEPARATE_CANCER_MODELS",
                        "ALL",
                        float(horizon),
                        RuntimeError(
                            "pooled result unavailable because one or more "
                            "cancer-specific model fits failed"
                        ),
                    )


            # Main paper-style MTL-BNN: pooled training with task descriptor.
            try:
                mtr_all = keep_tr
                train_pool_idx = np.flatnonzero(mtr_all)
                if USE_INNER_VAL_FOR_NEURAL:
                    rel_fit_idx, rel_val_idx = make_inner_val_split(
                        g_tr[train_pool_idx],
                        y_tr_bin[train_pool_idx],
                        val_frac=INNER_VAL_FRAC,
                        seed=seed + int(round(float(horizon) * 10.0)) + 3000,
                    )
                    bnn_fit_idx = train_pool_idx[rel_fit_idx]
                    bnn_val_idx = train_pool_idx[rel_val_idx]
                else:
                    bnn_fit_idx = train_pool_idx
                    bnn_val_idx = np.asarray([], dtype=int)

                Xo_fit_raw = _take_block(Xtr[bnn_fit_idx], omics_idx)
                Xc_fit_raw = _take_block(Xtr[bnn_fit_idx], clinical_idx)
                Xo_test_raw = _take_block(Xte, omics_idx)
                Xc_test_raw = _take_block(Xte, clinical_idx)
                if len(bnn_val_idx) > 0:
                    Xo_val_raw = _take_block(Xtr[bnn_val_idx], omics_idx)
                    Xc_val_raw = _take_block(Xtr[bnn_val_idx], clinical_idx)
                    Xo_fit, Xo_val, Xo_test = _standardize_fit_apply(Xo_fit_raw, Xo_val_raw, Xo_test_raw)
                    Xc_fit, Xc_val, Xc_test = _standardize_fit_apply(Xc_fit_raw, Xc_val_raw, Xc_test_raw)
                    y_val_bnn = y_tr_bin[bnn_val_idx]
                    g_val_bnn = g_tr[bnn_val_idx]
                else:
                    Xo_fit, Xo_test = _standardize_fit_apply(Xo_fit_raw, Xo_test_raw)
                    Xc_fit, Xc_test = _standardize_fit_apply(Xc_fit_raw, Xc_test_raw)
                    Xo_val = Xc_val = y_val_bnn = g_val_bnn = None

                set_all_seeds(seed + 3000)
                mtbnn = fit_paper_bimodal_bnn(
                    Xo_fit,
                    Xc_fit,
                    y_tr_bin[bnn_fit_idx],
                    g_fit=g_tr[bnn_fit_idx],
                    Xo_val=Xo_val,
                    Xc_val=Xc_val,
                    y_val=y_val_bnn,
                    g_val=g_val_bnn,
                    num_tasks=num_groups,
                    mtl_weighted_sampler=True,
                    device=device,
                )
                risk_all = predict_paper_bimodal_bnn(mtbnn, Xo_test, Xc_test, g_te, device=device)
                _record_fixed_horizon_metrics(
                    "MTL-BNN",
                    "paper_mtl_pooled",
                    "ALL",
                    "ALL",
                    t_te,
                    e_te,
                    risk_all,
                    float(horizon),
                )
                for gi, gname in enumerate(groups):
                    m = (g_te == gi)
                    if int(np.sum(m)) == 0:
                        continue
                    _record_fixed_horizon_metrics(
                        "MTL-BNN",
                        "paper_mtl_pooled",
                        "ALL",
                        gname,
                        t_te[m],
                        e_te[m],
                        risk_all[m],
                        float(horizon),
                    )
            except Exception as ex:
                
                _append_fixed_error_row("MTL-BNN", "paper_mtl_pooled", "ALL", "ALL", float(horizon), ex)
                for _gname in groups:
                    _append_fixed_error_row("MTL-BNN", "paper_mtl_pooled", "ALL", _gname, float(horizon), ex)

    if not RUN_SURVIVAL_BASELINES:
        return out_rows, prediction_rows

    # ---------------------- POOLED models ----------------------

    # MT-DeepCox pooled
    try:
        set_all_seeds(seed + 101)
        mtds = fit_mtdeepsurv(
            Xfit, t_fit, e_fit, g_fit, Xval, t_val, e_val, g_val, num_groups=num_groups, device=device
        )
        def _mtds_predict(x, g, h):
            return predict_mtdeepsurv_risk_at_horizon(
                mtds, Xfit, g_fit, t_fit, e_fit, x, g, horizon=float(h), device=device
            )
        for horizon in EVAL_HORIZONS_MONTHS:
            ibs_grid = _make_ibs_grid(float(horizon))
            ibs_risks_full = np.vstack([_mtds_predict(Xte, g_te, h) for h in ibs_grid])
            risk_te = ibs_risks_full[-1]
            _record_model_metrics("MT-DeepCox", "pooled", "ALL", "ALL", t_te, e_te, risk_te, float(horizon), ibs_grid, ibs_risks_full)
            for gi, gname in enumerate(groups):
                m = (g_te == gi)
                if int(np.sum(m)) == 0:
                    continue
                _record_model_metrics("MT-DeepCox", "pooled", "ALL", gname, t_te[m], e_te[m], risk_te[m], float(horizon), ibs_grid, ibs_risks_full[:, m])
    except Exception as ex:
        _append_error_rows("MT-DeepCox", "pooled", "ALL", ["ALL"] + list(groups), ex)

    # MT-DiscHazard pooled
    try:
        set_all_seeds(seed + 151)
        mtdh, dt_bin_edges = fit_mt_discrete_hazard(
            Xfit, t_fit, e_fit, g_fit, Xval, t_val, e_val, g_val, num_groups=num_groups, n_bins=DT_BINS, device=device
        )
        def _mtdh_predict(x, g, h):
            return predict_mt_discrete_hazard_risk_at_horizon(
                mtdh, x, g, dt_bin_edges, horizon=float(h), device=device
            )
        for horizon in EVAL_HORIZONS_MONTHS:
            ibs_grid = _make_ibs_grid(float(horizon))
            ibs_risks_full = np.vstack([_mtdh_predict(Xte, g_te, h) for h in ibs_grid])
            risk_te = ibs_risks_full[-1]
            _record_model_metrics("MT-DiscHazard", "pooled", "ALL", "ALL", t_te, e_te, risk_te, float(horizon), ibs_grid, ibs_risks_full)
            for gi, gname in enumerate(groups):
                m = (g_te == gi)
                if int(np.sum(m)) == 0:
                    continue
                _record_model_metrics("MT-DiscHazard", "pooled", "ALL", gname, t_te[m], e_te[m], risk_te[m], float(horizon), ibs_grid, ibs_risks_full[:, m])
    except Exception as ex:
        _append_error_rows("MT-DiscHazard", "pooled", "ALL", ["ALL"] + list(groups), ex)

    # DeepCox pooled
    try:
        set_all_seeds(seed + 201)
        ds_pooled = fit_deepsurv(Xfit, t_fit, e_fit, Xval, t_val, e_val, device=device)
        def _ds_pooled_predict(x, h):
            return predict_deepsurv_risk_at_horizon(ds_pooled, Xfit, t_fit, e_fit, x, horizon=float(h), device=device)
        for horizon in EVAL_HORIZONS_MONTHS:
            ibs_grid = _make_ibs_grid(float(horizon))
            ibs_risks_full = np.vstack([_ds_pooled_predict(Xte, h) for h in ibs_grid])
            risk_te = ibs_risks_full[-1]
            _record_model_metrics("DeepCox", "pooled", "ALL", "ALL", t_te, e_te, risk_te, float(horizon), ibs_grid, ibs_risks_full)
            for gi, gname in enumerate(groups):
                m = (g_te == gi)
                if int(np.sum(m)) == 0:
                    continue
                _record_model_metrics("DeepCox", "pooled", "ALL", gname, t_te[m], e_te[m], risk_te[m], float(horizon), ibs_grid, ibs_risks_full[:, m])
    except Exception as ex:
        _append_error_rows("DeepCox", "pooled", "ALL", ["ALL"] + list(groups), ex)

    # CoxPH pooled
    try:
        cph, cph_cols = fit_coxph_lifelines(Xtr, t_tr, e_tr, penalizer=COX_PENALIZER)
        def _cox_pooled_predict(x, h):
            return predict_coxph_risk_at_horizon(cph, x, cph_cols, horizon=float(h))
        for horizon in EVAL_HORIZONS_MONTHS:
            ibs_grid = _make_ibs_grid(float(horizon))
            ibs_risks_full = np.vstack([_cox_pooled_predict(Xte, h) for h in ibs_grid])
            risk_te = ibs_risks_full[-1]
            _record_model_metrics("CoxPH", "pooled", "ALL", "ALL", t_te, e_te, risk_te, float(horizon), ibs_grid, ibs_risks_full)
            for gi, gname in enumerate(groups):
                m = (g_te == gi)
                if int(np.sum(m)) == 0:
                    continue
                _record_model_metrics("CoxPH", "pooled", "ALL", gname, t_te[m], e_te[m], risk_te[m], float(horizon), ibs_grid, ibs_risks_full[:, m])
    except Exception as ex:
        _append_error_rows("CoxPH", "pooled", "ALL", ["ALL"] + list(groups), ex)

    # Random survival forest pooled
    if RUN_RSF_BASELINE:
        try:
            rsf_backend, rsf_model, rsf_cols = _fit_rsf(
                Xtr, t_tr, e_tr, seed=seed + 251
            )

            def _rsf_pooled_predict(x, h):
                return _predict_rsf_risk_at_horizon(
                    rsf_backend,
                    rsf_model,
                    x,
                    rsf_cols,
                    horizon=float(h),
                )

            for horizon in EVAL_HORIZONS_MONTHS:
                ibs_grid = _make_ibs_grid(float(horizon))
                ibs_risks_full = np.vstack(
                    [_rsf_pooled_predict(Xte, h) for h in ibs_grid]
                )
                risk_te = ibs_risks_full[-1]
                _record_model_metrics(
                    "RSF",
                    "pooled",
                    "ALL",
                    "ALL",
                    t_te,
                    e_te,
                    risk_te,
                    float(horizon),
                    ibs_grid,
                    ibs_risks_full,
                )
                for gi, gname in enumerate(groups):
                    m = g_te == gi
                    if int(np.sum(m)) == 0:
                        continue
                    _record_model_metrics(
                        "RSF",
                        "pooled",
                        "ALL",
                        gname,
                        t_te[m],
                        e_te[m],
                        risk_te[m],
                        float(horizon),
                        ibs_grid,
                        ibs_risks_full[:, m],
                    )
        except Exception as ex:
            _append_error_rows("RSF", "pooled", "ALL", ["ALL"] + list(groups), ex)

    # ---------------------- PER-COHORT models ----------------------
    per_cohort_surv_curves = {
        mdl: {
            float(h): np.full(
                (len(_make_ibs_grid(float(h))), len(t_te)), np.nan, dtype=float
            )
            for h in EVAL_HORIZONS_MONTHS
        }
        for mdl in (["DeepCox", "CoxPH"] + (["RSF"] if RUN_RSF_BASELINE else []))
    }
    for gi, gname in enumerate(groups):
        mtr = g_tr == gi
        mte = g_te == gi
        if (
            int(mtr.sum()) < 20
            or int(mte.sum()) < 10
            or len(np.unique(e_tr[mtr])) < 2
            or len(np.unique(e_te[mte])) < 2
        ):
            skip_reason = (
                "non-estimable: insufficient per-cohort train/test sample size or "
                "event variation"
            )
            nan_metrics = {
                "c_test": float("nan"),
                "somers_dxy_test": float("nan"),
                "brier_at_H": float("nan"),
                "ibs_up_to_H": float("nan"),
                "auroc": float("nan"),
                "auprc": float("nan"),
            }
            skipped_models = ["DeepCox", "CoxPH"]
            if RUN_RSF_BASELINE:
                skipped_models.append("RSF")
            for mdl in skipped_models:
                for horizon in EVAL_HORIZONS_MONTHS:
                    _append_metric_row(
                        out_rows,
                        seed=seed,
                        model=mdl,
                        train_mode="per_cohort",
                        train_group=gname,
                        eval_group=gname,
                        horizon=float(horizon),
                        metrics=nan_metrics,
                        prep_dir=prep_dir,
                        error=skip_reason,
                    )
            continue

        # DeepCox per-cohort
        try:
            set_all_seeds(seed + 301 + gi)
            group_train_idx = np.flatnonzero(mtr)
            if USE_INNER_VAL_FOR_NEURAL:
                rel_fit_idx, rel_val_idx = make_inner_val_split(
                    np.zeros(len(group_train_idx), dtype=int),
                    e_tr[group_train_idx],
                    val_frac=INNER_VAL_FRAC,
                    seed=seed + 301 + gi,
                )
                ds_fit_idx = group_train_idx[rel_fit_idx]
                ds_val_idx = group_train_idx[rel_val_idx]
            else:
                ds_fit_idx = group_train_idx
                ds_val_idx = np.asarray([], dtype=int)
            if len(ds_val_idx) > 0:
                ds = fit_deepsurv(Xtr[ds_fit_idx], t_tr[ds_fit_idx], e_tr[ds_fit_idx], Xtr[ds_val_idx], t_tr[ds_val_idx], e_tr[ds_val_idx], device=device)
            else:
                ds = fit_deepsurv(Xtr[ds_fit_idx], t_tr[ds_fit_idx], e_tr[ds_fit_idx], None, None, None, device=device)
            def _ds_group_predict(x, h):
                return predict_deepsurv_risk_at_horizon(ds, Xtr[ds_fit_idx], t_tr[ds_fit_idx], e_tr[ds_fit_idx], x, horizon=float(h), device=device)
            for horizon in EVAL_HORIZONS_MONTHS:
                ibs_grid = _make_ibs_grid(float(horizon))
                ibs_risks = np.vstack([_ds_group_predict(Xte[mte], h) for h in ibs_grid])
                risk = ibs_risks[-1]
                per_cohort_surv_curves["DeepCox"][float(horizon)][:, mte] = ibs_risks
                _record_model_metrics("DeepCox", "per_cohort", gname, gname, t_te[mte], e_te[mte], risk, float(horizon), ibs_grid, ibs_risks)
        except Exception as ex:
            _append_error_rows("DeepCox", "per_cohort", gname, [gname], ex)

        # CoxPH per-cohort
        try:
            cph, cph_cols = fit_coxph_lifelines(Xtr[mtr], t_tr[mtr], e_tr[mtr], penalizer=COX_PENALIZER)
            def _cox_group_predict(x, h):
                return predict_coxph_risk_at_horizon(cph, x, cph_cols, horizon=float(h))
            for horizon in EVAL_HORIZONS_MONTHS:
                ibs_grid = _make_ibs_grid(float(horizon))
                ibs_risks = np.vstack([_cox_group_predict(Xte[mte], h) for h in ibs_grid])
                risk = ibs_risks[-1]
                per_cohort_surv_curves["CoxPH"][float(horizon)][:, mte] = ibs_risks
                _record_model_metrics("CoxPH", "per_cohort", gname, gname, t_te[mte], e_te[mte], risk, float(horizon), ibs_grid, ibs_risks)
        except Exception as ex:
            _append_error_rows("CoxPH", "per_cohort", gname, [gname], ex)

        # Random survival forest per cohort
        if RUN_RSF_BASELINE:
            try:
                rsf_backend, rsf_model, rsf_cols = _fit_rsf(
                    Xtr[mtr], t_tr[mtr], e_tr[mtr], seed=seed + 401 + gi
                )

                def _rsf_group_predict(x, h):
                    return _predict_rsf_risk_at_horizon(
                        rsf_backend,
                        rsf_model,
                        x,
                        rsf_cols,
                        horizon=float(h),
                    )

                for horizon in EVAL_HORIZONS_MONTHS:
                    ibs_grid = _make_ibs_grid(float(horizon))
                    ibs_risks = np.vstack(
                        [_rsf_group_predict(Xte[mte], h) for h in ibs_grid]
                    )
                    risk = ibs_risks[-1]
                    per_cohort_surv_curves["RSF"][float(horizon)][:, mte] = ibs_risks
                    _record_model_metrics(
                        "RSF",
                        "per_cohort",
                        gname,
                        gname,
                        t_te[mte],
                        e_te[mte],
                        risk,
                        float(horizon),
                        ibs_grid,
                        ibs_risks,
                    )
            except Exception as ex:
                _append_error_rows("RSF", "per_cohort", gname, [gname], ex)


    # Pooled evaluation of the separately fitted survival models. Every cancer
    # must have succeeded so the pooled row always represents the full test set.
    for mdl, curves_by_h in per_cohort_surv_curves.items():
        for horizon, ibs_risks in curves_by_h.items():
            if np.isfinite(ibs_risks).all():
                ibs_grid = _make_ibs_grid(float(horizon))
                _record_model_metrics(
                    mdl,
                    "per_cohort",
                    "SEPARATE_CANCER_MODELS",
                    "ALL",
                    t_te,
                    e_te,
                    ibs_risks[-1],
                    float(horizon),
                    ibs_grid,
                    ibs_risks,
                )
            else:
                _append_metric_row(
                    out_rows,
                    seed=seed,
                    model=mdl,
                    train_mode="per_cohort",
                    train_group="SEPARATE_CANCER_MODELS",
                    eval_group="ALL",
                    horizon=float(horizon),
                    metrics={
                        "c_test": float("nan"),
                        "uno_c_ipcw": float("nan"),
                        "somers_dxy_test": float("nan"),
                        "brier_at_H": float("nan"),
                        "ibs_up_to_H": float("nan"),
                        "auroc": float("nan"),
                        "auprc": float("nan"),
                    },
                    prep_dir=prep_dir,
                    error=(
                        "pooled result unavailable because one or more "
                        "cancer-specific model fits failed"
                    ),
                )


    return out_rows, prediction_rows



# ==========================================================================

def summary_table(df_in: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = df_in.copy()
    metric_cols = [
        "c_test",
        "uno_c_ipcw",
        "somers_dxy_test",
        "brier_at_H",
        "ibs_up_to_H",
        "auroc",
        "auprc",
    ]
    for col in metric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    if "error" not in d.columns:
        d["error"] = ""
    d["failed"] = d["error"].fillna("").astype(str).str.len() > 0
    g = d.groupby(group_cols, dropna=False)
    out = g.agg(
        c_mean=("c_test", "mean"),
        c_sd=("c_test", "std"),
        uno_c_mean=("uno_c_ipcw", "mean"),
        uno_c_sd=("uno_c_ipcw", "std"),
        dxy_mean=("somers_dxy_test", "mean"),
        dxy_sd=("somers_dxy_test", "std"),
        brier_mean=("brier_at_H", "mean"),
        brier_sd=("brier_at_H", "std"),
        ibs_mean=("ibs_up_to_H", "mean"),
        ibs_sd=("ibs_up_to_H", "std"),
        auroc_mean=("auroc", "mean"),
        auroc_sd=("auroc", "std"),
        auprc_mean=("auprc", "mean"),
        auprc_sd=("auprc", "std"),
        n_non_nan=("c_test", lambda x: int(pd.Series(x).notna().sum())),
        n_uno_non_nan=("uno_c_ipcw", lambda x: int(pd.Series(x).notna().sum())),
        n_failed=("failed", "sum"),
        n_rows=("seed", "size"),
        n_seeds=("seed", lambda x: int(pd.Series(x).nunique())),
    ).reset_index()
    out = out.sort_values([c for c in group_cols if c in out.columns]).reset_index(drop=True)
    return out



def main():
    ap = argparse.ArgumentParser(
        description="TCGA NPJ all baseline comparisons under PFI endpoint"
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--multi", action="store_true", help="Run MULTI_SEEDS instead of single seed"
    )
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    use_multi = args.multi or RUN_MULTI_SEEDS
    seeds = MULTI_SEEDS if use_multi else [args.seed]
    seeds = list(dict.fromkeys([int(s) for s in seeds]))

    print("\n====================== TCGA BASELINES ======================")
    print(f"Using precomputed TCGA preprocessing outputs from base dir: {PREP_OUTDIR}")
    print(
        "If split_seed{seed}/ folders exist, each seed will load its matching outer split."
    )
    print(f"Saving outputs to: {OUTDIR}")
    print(
        "[Legend] train_mode=pooled: fit one pooled model on all groups; eval_group rows report test C-index within each group."
    )
    print(
        "[Legend] train_mode=per_cohort: fit a separate model inside each GROUP_LABEL; eval_group identifies that group."
    )

    rows = []
    prediction_rows = []
    for s in seeds:
        print(f"\n[Baselines] ===== Seed {s} =====")
        seed_rows, seed_prediction_rows = run_baselines_one_seed(
            s, device=args.device
        )
        rows.extend(seed_rows)
        prediction_rows.extend(seed_prediction_rows)

    df = pd.DataFrame(rows)
    if "mode" not in df.columns:
        df["mode"] = df["train_mode"]
    if "group" not in df.columns:
        df["group"] = df["eval_group"]

    out_path = OUTDIR / "tcga_baselines_npj_allmodels_pfi.csv"
    df.to_csv(out_path, index=False)
    prediction_path = OUTDIR / "tcga_baselines_patient_predictions.csv.gz"
    pd.DataFrame(prediction_rows).to_csv(
        prediction_path, index=False, compression="gzip"
    )
    manifest_path = OUTDIR / "baseline_run_manifest.json"
    manifest = {
        "input_config": str(CONFIG_NAME),
        "preprocessing_root": str(PREP_OUTDIR),
        "output_root": str(OUTDIR),
        "seeds": [int(s) for s in seeds],
        "evaluation_horizons_months": [
            float(h) for h in EVAL_HORIZONS_MONTHS
        ],
        "discrete_time_edges_months": [float(h) for h in DT_EDGES_MONTHS],
        "models": [
            "LR",
            "RF",
            "SVM",
            "STL-BNN",
            "MTL-BNN",
            "CoxPH",
            "DeepCox",
            "MT-DeepCox",
            "MT-DiscHazard",
        ]
        + (["RSF"] if RUN_RSF_BASELINE else []),
        "bnn_epochs": int(BNN_EPOCHS),
        "survival_neural_epochs": int(SURV_NN_EPOCHS),
        "use_inner_validation_for_neural": bool(USE_INNER_VAL_FOR_NEURAL),
        "censoring_weight_source": "matching outer-training split",
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    summ_all = summary_table(
        df[df["eval_group"] == "ALL"], ["input_config", "model", "train_mode", "eval_horizon_months"]
    )
    summ_all_path = OUTDIR / "tcga_baselines_summary_all.csv"
    summ_all.to_csv(summ_all_path, index=False)

    summ_by_group = summary_table(
        df, ["input_config", "model", "train_mode", "eval_group", "eval_horizon_months"]
    )
    summ_by_group_path = OUTDIR / "tcga_baselines_summary_by_group.csv"
    summ_by_group.to_csv(summ_by_group_path, index=False)

    print("\nSaved:")
    print(f"  {out_path}")
    print(f"  {prediction_path}")
    print(f"  {manifest_path}")
    print(f"  {summ_all_path}")
    print(f"  {summ_by_group_path}")

    print("\nSummary across seeds (ALL only):")
    print(summ_all.to_string(index=False))

    print("\nSummary across seeds (by group):")
    print(summ_by_group.to_string(index=False))


if __name__ == "__main__":
    main()
