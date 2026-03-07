#!/bin/bash -l

#SBATCH --job-name=graphvae_full_gpu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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

echo "[ENV] host=$(hostname)"
echo "[ENV] python=$PYTHON"
"$PYTHON" -V
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
MD_ROOT="${MD_ROOT:-$SCRATCH_ROOT/MD_datasets/$DATASET_STEM}"
PDB_CACHE_ROOT="${PDB_CACHE_ROOT:-$SCRATCH_ROOT/pdb_cache}"
OUT_DIR="${OUT_DIR:-src/bridging/generatedData/graphvae/$DATASET_STEM}"

BUILD_DEEPRANK="${BUILD_DEEPRANK:-0}"
REUSE_PREPARED="${REUSE_PREPARED:-0}"
DEEPRANK_PREFIX="${DEEPRANK_PREFIX:-$SCRATCH_ROOT/deeprank/$DATASET_STEM/graphs}"
DEEPRANK_HDF5="${DEEPRANK_HDF5:-}"

COMMON_ARGS=(
  --dataset "$DATASET"
  --md-root "$MD_ROOT"
  --out-dir "$OUT_DIR"
  --pdb-cache-root "$PDB_CACHE_ROOT"
  --graph-source "${GRAPH_SOURCE:-md_topology_protein}"
  --influence-radius "${INFLUENCE_RADIUS:-1000000}"
  --frames-per-complex "${FRAMES_PER_COMPLEX:-120}"
  --prepare-progress-every "${PREP_PROGRESS_EVERY:-25}"
  --split-seed "${SPLIT_SEED:-42}"
  --max-epochs "${MAX_EPOCHS:-120}"
  --patience "${PATIENCE:-20}"
  --batch-size "${BATCH_SIZE:-16}"
  --latent-dim "${LATENT_DIM:-8}"
  --device "${DEVICE:-cuda}"
  --num-workers "${NUM_WORKERS:-4}"
  --alpha-grid "${ALPHA_GRID:-1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100}"
)
if [ "${RUN_SUP_BASELINES:-0}" = "1" ]; then
  COMMON_ARGS+=(--run-supervised-baselines)
fi
if [ "${ALLOW_PARTIAL_NODE_COVERAGE:-0}" = "1" ]; then
  COMMON_ARGS+=(--allow-partial-node-coverage)
fi

if [ "$REUSE_PREPARED" = "1" ]; then
  "$PYTHON" -m bridging.graphvae.run_full \
    "${COMMON_ARGS[@]}" \
    --reuse-prepared
elif [ "$BUILD_DEEPRANK" = "1" ]; then
  "$PYTHON" -m bridging.graphvae.run_full \
    "${COMMON_ARGS[@]}" \
    --build-deeprank \
    --deeprank-prefix "$DEEPRANK_PREFIX"
else
  if [ -z "$DEEPRANK_HDF5" ]; then
    echo "[FAIL] BUILD_DEEPRANK=0 but DEEPRANK_HDF5 is empty."
    echo "[HINT] set DEEPRANK_HDF5 to one or more space-separated hdf5 paths."
    exit 1
  fi
  # shellcheck disable=SC2206
  H5_LIST=( $DEEPRANK_HDF5 )
  "$PYTHON" -m bridging.graphvae.run_full \
    "${COMMON_ARGS[@]}" \
    --deep-rank-hdf5 "${H5_LIST[@]}"
fi

echo "[DONE] compare summary: $OUT_DIR/compare_S_vs_SD.json"
