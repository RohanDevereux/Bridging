#!/bin/bash -l

#SBATCH --job-name=graphvae_full_gpu
#SBATCH --time=10-00:00:00
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
export PYTHONUNBUFFERED=1

echo "[ENV] host=$(hostname)"
echo "[ENV] python=$PYTHON"
"$PYTHON" -V
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

if [ -n "${MSMS_MODULE:-}" ]; then
  module load "$MSMS_MODULE"
fi
if [ -n "${MSMS_BIN_DIR:-}" ]; then
  export PATH="${MSMS_BIN_DIR}:$PATH"
fi
if ! command -v msms >/dev/null 2>&1; then
  echo "[FAIL] msms not found on PATH. DeepRank exposure features require msms."
  echo "[HINT] set MSMS_MODULE (if available) or install msms on this cluster."
  exit 1
fi
echo "[ENV] msms=$(command -v msms)"

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
DEEPRANK_QUERY_MODE="${DEEPRANK_QUERY_MODE:-full_complex}"
DEEPRANK_CPU_COUNT="${DEEPRANK_CPU_COUNT:-4}"

PREFLIGHT_ARGS=()
H5_LIST=()
if [ "$BUILD_DEEPRANK" = "0" ] && [ -n "$DEEPRANK_HDF5" ]; then
  # shellcheck disable=SC2206
  H5_LIST=( $DEEPRANK_HDF5 )
  PREFLIGHT_ARGS+=(--deep-rank-hdf5 "${H5_LIST[@]}")
fi

"$PYTHON" -m bridging.graphvae.preflight \
  --dataset "$DATASET" \
  --md-root "$MD_ROOT" \
  --expected-deeprank2-version "${DEEPRANK2_VERSION:-3.1.0}" \
  --sample-done "${PREFLIGHT_SAMPLE_DONE:-20}" \
  --require-done "${PREFLIGHT_REQUIRE_DONE:-50}" \
  --validate-hdf5-sample "${PREFLIGHT_VALIDATE_HDF5_SAMPLE:-40}" \
  "${PREFLIGHT_ARGS[@]}"

COMMON_ARGS=(
  --dataset "$DATASET"
  --md-root "$MD_ROOT"
  --out-dir "$OUT_DIR"
  --pdb-cache-root "$PDB_CACHE_ROOT"
  --graph-source "${GRAPH_SOURCE:-md_topology_protein}"
  --influence-radius "${INFLUENCE_RADIUS:-1000000}"
  --deeprank-query-mode "$DEEPRANK_QUERY_MODE"
  --deeprank-cpu-count "$DEEPRANK_CPU_COUNT"
  --frames-per-complex "${FRAMES_PER_COMPLEX:-120}"
  --prepare-traj-cache-size "${PREP_TRAJ_CACHE_SIZE:-1}"
  --prepare-progress-every "${PREP_PROGRESS_EVERY:-25}"
  --prepare-checkpoint-every "${PREP_CHECKPOINT_EVERY:-25}"
  --split-seed "${SPLIT_SEED:-42}"
  --max-epochs "${MAX_EPOCHS:-120}"
  --patience "${PATIENCE:-20}"
  --batch-size "${BATCH_SIZE:-16}"
  --latent-dim "${LATENT_DIM:-8}"
  --hidden-dim "${HIDDEN_DIM:-128}"
  --num-layers "${NUM_LAYERS:-3}"
  --mask-ratio "${MASK_RATIO:-0.30}"
  --lr "${LR:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-1e-5}"
  --beta-start "${BETA_START:-0.0}"
  --beta-end "${BETA_END:-1.0}"
  --beta-anneal-fraction "${BETA_ANNEAL_FRACTION:-0.30}"
  --corr-weight "${CORR_WEIGHT:-0.01}"
  --device "${DEVICE:-cuda}"
  --num-workers "${NUM_WORKERS:-4}"
  --train-checkpoint-every "${TRAIN_CHECKPOINT_EVERY:-1}"
  --alpha-grid "${ALPHA_GRID:-1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100}"
  --ridge-cv-folds "${RIDGE_CV_FOLDS:-0}"
  --ridge-cv-repeats "${RIDGE_CV_REPEATS:-0}"
  --ridge-cv-inner-folds "${RIDGE_CV_INNER_FOLDS:-5}"
  --vae-cv-folds "${VAE_CV_FOLDS:-0}"
  --vae-cv-repeats "${VAE_CV_REPEATS:-0}"
  --vae-cv-val-fraction "${VAE_CV_VAL_FRACTION:-0.15}"
)
if [ -n "${MMGBSA_CSV:-}" ]; then
  COMMON_ARGS+=(--mmgbsa-csv "$MMGBSA_CSV")
fi
if [ "${RUN_SUP_BASELINES:-0}" = "1" ]; then
  COMMON_ARGS+=(--run-supervised-baselines)
fi
if [ "${ALLOW_PARTIAL_NODE_COVERAGE:-0}" = "1" ]; then
  COMMON_ARGS+=(--allow-partial-node-coverage)
fi
if [ "${INCLUDE_DYNAMIC_DIST_STATS:-1}" = "0" ]; then
  COMMON_ARGS+=(--no-include-dynamic-dist-stats)
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
  if [ "${#H5_LIST[@]}" -eq 0 ]; then
    # shellcheck disable=SC2206
    H5_LIST=( $DEEPRANK_HDF5 )
  fi
  "$PYTHON" -m bridging.graphvae.run_full \
    "${COMMON_ARGS[@]}" \
    --deep-rank-hdf5 "${H5_LIST[@]}"
fi

echo "[DONE] compare summary: $OUT_DIR/compare_S_vs_SD.json"
