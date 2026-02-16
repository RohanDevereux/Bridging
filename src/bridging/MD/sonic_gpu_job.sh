#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source .venv/bin/activate

export PYTHONPATH=src

DATASET="${DATASET:-src/bridging/processedData/PRODIGY_Data.csv}"
OUT_ROOT="${OUT_ROOT:-src/bridging/generatedData/MD_datasets/$(basename "$DATASET" .csv)}"

python -m bridging.MD.run_dataset --dataset "$DATASET" --out "$OUT_ROOT"
