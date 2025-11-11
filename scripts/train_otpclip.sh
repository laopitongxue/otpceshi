#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATA_ROOT=${DATA_ROOT:-/data/caltech101}
CONFIG=${CONFIG:-otpclip/configs/otpclip_caltech101.yaml}

python "${ROOT_DIR}/Dassl.pytorch/tools/train.py" \
  --config-file "${ROOT_DIR}/${CONFIG}" \
  --trainer OTPCLIPTrainer \
  DATASET.ROOT "${DATA_ROOT}" "$@" 
