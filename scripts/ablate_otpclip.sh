#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
CONFIG=${CONFIG:-otpclip/configs/otpclip_caltech101.yaml}
DATA_ROOT=${DATA_ROOT:-/data/caltech101}

ablations=(
  ""
  "MODEL.MTA.TRAIN_GRAD true"
  "MODEL.MTA.ENTROPY false"
  "MODEL.OT.SINKHORN 0"
  "MODEL.PROMPT_NOISE.GRANULARITY class"
  "MODEL.PROMPT_NOISE.GRANULARITY global"
  "LOSS.ROBUST_TYPE GCE"
  "MODEL.PROMPT_NOISE.ANNEAL.START 0 MODEL.PROMPT_NOISE.ANNEAL.END 0"
)

for opts in "${ablations[@]}"; do
  echo "Running OTP-CLIP with options: ${opts}"
  python "${ROOT_DIR}/Dassl.pytorch/tools/train.py" \
    --config-file "${ROOT_DIR}/${CONFIG}" \
    --trainer OTPCLIPTrainer \
    DATASET.ROOT "${DATA_ROOT}" ${opts}
done
