#!/bin/bash -l

#SBATCH --job-name=graphvae_prep_shard
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=shared
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
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

DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
MD_ROOT="${MD_ROOT:-$SCRATCH_ROOT/MD_datasets/$DATASET_STEM}"
PDB_CACHE_ROOT="${PDB_CACHE_ROOT:-$SCRATCH_ROOT/pdb_cache}"
DEEPRANK_HDF5="${DEEPRANK_HDF5:-}"

if [ -z "$DEEPRANK_HDF5" ]; then
  echo "[FAIL] DEEPRANK_HDF5 is empty."
  exit 1
fi

SHARD_DIR="${SHARD_DIR:-tmp/graphvae_prepare_shards/$DATASET_STEM}"
OUT_ROOT="${OUT_ROOT:-src/bridging/generatedData/graphvae/${DATASET_STEM}_parallel_prepare}"
SHARD_ID="$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"
SHARD_CSV="$SHARD_DIR/shard_${SHARD_ID}.csv"
OUT_DIR="$OUT_ROOT/shard_${SHARD_ID}/prepared"

if [ ! -f "$SHARD_CSV" ]; then
  echo "[FAIL] missing shard CSV: $SHARD_CSV"
  echo "[HINT] generate shards before submitting this array job."
  exit 1
fi

ROWS=$(( $(wc -l < "$SHARD_CSV") - 1 ))
if [ "$ROWS" -lt 0 ]; then
  ROWS=0
fi
echo "[PREP] shard_id=$SHARD_ID rows=$ROWS csv=$SHARD_CSV out=$OUT_DIR"
if [ "$ROWS" -eq 0 ]; then
  echo "[PREP] shard is empty; exiting."
  exit 0
fi

# shellcheck disable=SC2206
H5_LIST=( $DEEPRANK_HDF5 )

ARGS=(
  --dataset "$SHARD_CSV"
  --md-root "$MD_ROOT"
  --out-dir "$OUT_DIR"
  --pdb-cache-root "$PDB_CACHE_ROOT"
  --graph-source "${GRAPH_SOURCE:-md_topology_protein}"
  --deep-rank-hdf5 "${H5_LIST[@]}"
  --frames-per-complex "${FRAMES_PER_COMPLEX:-120}"
  --traj-cache-size "${PREP_TRAJ_CACHE_SIZE:-1}"
  --checkpoint-every "${PREP_CHECKPOINT_EVERY:-25}"
  --progress-every "${PREP_PROGRESS_EVERY:-25}"
  --per-complex-timeout-minutes "${PREP_COMPLEX_TIMEOUT_MINUTES:-30}"
  --split-seed "${SPLIT_SEED:-42}"
)
if [ "${INCLUDE_DYNAMIC_DIST_STATS:-1}" = "0" ]; then
  ARGS+=(--no-include-dynamic-dist-stats)
fi
if [ "${ALLOW_PARTIAL_NODE_COVERAGE:-0}" = "1" ]; then
  ARGS+=(--allow-partial-node-coverage)
fi
if [ -n "${MAX_EDGE_LENGTH:-}" ]; then
  ARGS+=(--max-edge-length "$MAX_EDGE_LENGTH")
fi

"$PYTHON" -m bridging.graphvae.prepare "${ARGS[@]}"

echo "[DONE] prepare shard_id=$SHARD_ID out=$OUT_DIR"
