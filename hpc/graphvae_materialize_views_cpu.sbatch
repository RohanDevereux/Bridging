#!/bin/bash -l

#SBATCH --job-name=graphvae_views
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=shared
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ucd.ie

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -f "$SLURM_SUBMIT_DIR/.venv/bin/activate" ]; then
  source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
fi

PYTHON="${PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || command -v python)"
fi
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1

echo "[ENV] host=$(hostname)"
echo "[ENV] python=$PYTHON"
"$PYTHON" -V

if [ -z "${RECORDS:-}" ]; then
  echo "[FAIL] RECORDS is empty."
  exit 1
fi
if [ -z "${DATASET:-}" ]; then
  echo "[FAIL] DATASET is empty."
  exit 1
fi

OUT_DIR="${OUT_DIR:-src/bridging/generatedData/graphvae/views/default}"
GRAPH_VIEWS="${GRAPH_VIEWS:-full,interface}"
INTERFACE_POLICIES="${INTERFACE_POLICIES:-md_closest_pair_patch}"

ARGS=(
  --records "$RECORDS"
  --dataset "$DATASET"
  --out-dir "$OUT_DIR"
  --graph-views "$GRAPH_VIEWS"
  --interface-policies "$INTERFACE_POLICIES"
  --progress-every "${VIEW_PROGRESS_EVERY:-25}"
)

if [ "${MATCH_INTERFACE_SUBSET:-true}" = "true" ]; then
  ARGS+=(--match-interface-subset)
else
  ARGS+=(--no-match-interface-subset)
fi

if [ -n "${PDB_CACHE_ROOT:-}" ]; then
  ARGS+=(--pdb-cache-root "$PDB_CACHE_ROOT")
fi
if [ -n "${MD_ROOT:-}" ]; then
  ARGS+=(--md-root "$MD_ROOT")
fi

"$PYTHON" -m bridging.graphvae.runners.materialize_views "${ARGS[@]}"

echo "[DONE] view materialization summary: $OUT_DIR/view_materialization_summary.json"
