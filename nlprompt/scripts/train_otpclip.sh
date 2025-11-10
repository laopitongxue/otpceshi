#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/otpclip_caltech101.yaml}
DATASET=${DATASET:-caltech101}
FEWSHOT=${FEWSHOT:-16}
ENABLE_MTA=${ENABLE_MTA:-true}
ENABLE_OT=${ENABLE_OT:-true}
ROBUST=${ROBUST:-mae}
EXTRA_ARGS=${EXTRA_ARGS:-}

python train.py \
  --config-file "${CONFIG}" \
  --trainer OTPCLIPTrainer \
  --dataset-config-file "configs/datasets/${DATASET}.yaml" \
  --few-shot ${FEWSHOT} \
  --robust ${ROBUST} \
  $( [ "${ENABLE_MTA}" = true ] && echo "--enable-mta" ) \
  $( [ "${ENABLE_OT}" = true ] && echo "--enable-ot" ) \
  ${EXTRA_ARGS}
