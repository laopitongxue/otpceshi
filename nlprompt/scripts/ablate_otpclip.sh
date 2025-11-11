#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/otpclip_caltech101.yaml}
DATASET=${DATASET:-caltech101}
FEWSHOT=${FEWSHOT:-16}

run_variant() {
  local name=$1
  shift
  OUTPUT_DIR="runs/${name}" EXTRA_ARGS="--output-dir ${OUTPUT_DIR} $*" \
    CONFIG=${CONFIG} DATASET=${DATASET} FEWSHOT=${FEWSHOT} \
    ROBUST=${ROBUST:-mae} ENABLE_MTA=${ENABLE_MTA:-true} ENABLE_OT=${ENABLE_OT:-true} \
    bash "$(dirname "$0")/train_otpclip.sh"
}

run_variant "full" ""
run_variant "no_mta" "--no-mta"
run_variant "no_ot" "--no-ot"
run_variant "no_prompt_noise" "--no-prompt-noise"
run_variant "robust_gce" "--robust gce"
