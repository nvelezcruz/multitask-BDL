#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash run_one_seed.sh {PathClin|PathNoUDP_MutTMB|PathUDP_MutTMB} SEED" >&2
    exit 64
fi

config="$1"
seed="$2"
case "$config" in
    PathClin|PathNoUDP_MutTMB|PathUDP_MutTMB) ;;
    *) echo "Unknown configuration: $config" >&2; exit 64 ;;
esac
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be an integer" >&2; exit 64; }
: "${TCGA_ROOT:?Set TCGA_ROOT to the directory containing TCGA_BRCA2, LUAD, and COAD}"
: "${RUN_ROOT:?Set RUN_ROOT to a new output directory containing August and the configuration name}"

code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/code" && pwd)"
release_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
preprocess_script="$code_dir/TCGA_preprocess_${config}_NPJ_August.py"
model_script="$code_dir/TCGA_dgp_${config}_NPJ_August.py"
baseline_script="$code_dir/TCGA_baselines_${config}_NPJ_August.py"
prep_dir="$RUN_ROOT/preprocessing"
model_dir="$RUN_ROOT/seed$seed"
baseline_dir="$RUN_ROOT/baselines/seed$seed"
manifest_dir="$model_dir/manifests"
cfg_tag="cfg_TOP0_S2_P2_MT2_OUT1"

mkdir -p "$prep_dir" "$model_dir" "$baseline_dir" "$manifest_dir"

write_dgp_status() {
    local status="$1" rc="$2"
    printf 'configuration,seed,status,exit_code,error,finished_utc\n%s,%s,%s,%s,%s,%s\n' \
        "$config" "$seed" "$status" "$rc" "" "$(date -u +%FT%TZ)" \
        > "$manifest_dir/seed_status.csv"
}

write_baseline_status() {
    local status="$1" rc="$2"
    printf 'configuration,seed,status,exit_code,error,finished_utc\n%s,%s,%s,%s,%s,%s\n' \
        "$config" "$seed" "$status" "$rc" "" "$(date -u +%FT%TZ)" \
        > "$baseline_dir/seed_status.csv"
}

write_wfc_status() {
    local horizon="$1" status="$2" rc="$3"
    printf 'configuration,seed,horizon,status,exit_code,finished_utc\n%s,%s,%s,%s,%s,%s\n' \
        "$config" "$seed" "$horizon" "$status" "$rc" "$(date -u +%FT%TZ)" \
        > "$manifest_dir/corrected_wfc_status_H${horizon}_all_cancers.csv"
}

CANONICAL_ONLY=1 OUTDIR="$prep_dir" "$python_bin" "$preprocess_script"
CANONICAL_ONLY=0 OUTDIR="$prep_dir" SPLIT_SEED="$seed" RUN_MULTI_SEEDS=0 \
    "$python_bin" "$preprocess_script"

if OUTDIR="$model_dir" PREP_OUTDIR="$prep_dir" SEED="$seed" \
    MULTI_SEEDS="$seed" REPEATED_SPLIT_MODE=1 TARGET_CONFIGS="0,2,2,2,1" \
    POSTHOC_CALIBRATION_ALLOW_PARTIAL_OOF=0 \
    POSTHOC_CALIBRATION_MIN_OOF_FRACTION=1.0 \
    ENABLE_POSTHOC_ANALYSIS_FOR_EACH_SPLIT=1 REQUIRE_COMPLETE_NPJ_OUTPUTS=1 \
    ENABLE_REPEATED_SPLIT_TREATMENT_ADJUSTED_FC=0 \
    REPEATED_SPLIT_TREATMENT_ADJUSTED_FC_SAVE_PLOTS=0 \
    ENABLE_REPEATED_SPLIT_FEATURE_STRATIFIED_TX_CONTRASTS=1 \
    FEATURE_STRATIFIED_TX_CONTRAST_SEEDS=all \
    TREATMENT_ADJUSTED_FC_H_LIST="12,18,24,36" \
    FEATURE_STRATIFIED_TX_H_LIST="3,6,12,18,24,36" \
    ENABLE_REPEATED_SPLIT_DGP_BRANCH_ATTRIBUTION=0 \
    ENABLE_BEST_SPLIT_REFIT_POSTHOC_ANALYSIS=0 \
    NPJ_SAVE_RAW_POSTERIOR_DRAWS=0 AUGUST_WFC_SAVE_PATIENT_LEVEL=0 \
    AUGUST_DEFER_WFC=1 "$python_bin" "$model_script" && \
    "$python_bin" "$release_dir/validate_seed_outputs.py" dgp \
        --seed "$seed" --output-dir "$model_dir"; then
    write_dgp_status completed 0
else
    rc=$?
    write_dgp_status failed "$rc"
    exit "$rc"
fi

if BASELINE_INPUT_CONFIG="$config" BASELINE_INPUT_SUFFIX="_$config" \
    PREP_OUTDIR="$prep_dir" OUTDIR="$baseline_dir" RUN_MULTI_SEEDS=0 \
    MULTI_SEEDS="$seed" EVAL_HORIZONS_MONTHS="3,6,12,18,24,36,60" \
    BNN_EPOCHS=50 SURV_NN_EPOCHS=900 RUN_PAPER_BINARY_BASELINES=1 \
    RUN_SURVIVAL_BASELINES=1 RUN_RSF_BASELINE=0 \
    USE_INNER_VAL_FOR_NEURAL=0 \
    "$python_bin" "$baseline_script" --seed "$seed" --device "${BASELINE_DEVICE:-cpu}" && \
    "$python_bin" "$release_dir/validate_seed_outputs.py" baseline \
        --seed "$seed" --output-dir "$baseline_dir"; then
    write_baseline_status complete 0
else
    rc=$?
    write_baseline_status failed "$rc"
    exit "$rc"
fi

artifact="$model_dir/$cfg_tag/august_fitted_artifacts_seed$seed.pt"
[[ -s "$artifact" ]] || { echo "Missing fitted DGP artifact: $artifact" >&2; exit 66; }
for horizon in 12 18 24 36; do
    part_dir="$model_dir/$cfg_tag/split_seed${seed}_analysis/corrected_wfc_parts/H${horizon}_all_cancers"
    mkdir -p "$part_dir"
    if OUTDIR="$model_dir" PREP_OUTDIR="$prep_dir" SEED="$seed" \
        AUGUST_WFC_ONLY_ARTIFACT="$artifact" AUGUST_WFC_PART_OUTDIR="$part_dir" \
        AUGUST_WFC_HORIZON="$horizon" AUGUST_CFG_TAG="$cfg_tag" \
        "$python_bin" "$model_script" && \
        "$python_bin" "$release_dir/validate_seed_outputs.py" wfc \
            --seed "$seed" --horizon "$horizon" --output-dir "$part_dir"; then
        write_wfc_status "$horizon" completed 0
    else
        rc=$?
        write_wfc_status "$horizon" failed "$rc"
        exit "$rc"
    fi
done
