#!/bin/bash -l

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || command -v python)"
fi
export PYTHONPATH="${PYTHONPATH:-src}"

DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
MD_ROOT="${MD_ROOT:-$SCRATCH_ROOT/MD_datasets/$DATASET_STEM}"
PDB_CACHE_ROOT="${PDB_CACHE_ROOT:-$SCRATCH_ROOT/pdb_cache}"
DEEPRANK_HDF5="${DEEPRANK_HDF5:-$SCRATCH_ROOT/deeprank/$DATASET_STEM/graphs_fullcomplex_v1.hdf5}"

N_SHARDS="${N_SHARDS:-3}"
CPUS_PER_SHARD="${CPUS_PER_SHARD:-16}"
TOTAL_CPUS=$(( N_SHARDS * CPUS_PER_SHARD ))
if [ "$TOTAL_CPUS" -gt 48 ]; then
  echo "[FAIL] requested total CPUs $TOTAL_CPUS exceeds shared-user cap 48."
  echo "[HINT] reduce N_SHARDS or CPUS_PER_SHARD."
  exit 1
fi

PARALLEL_OUT_ROOT="${PARALLEL_OUT_ROOT:-src/bridging/generatedData/graphvae/${DATASET_STEM}_parallel_prepare}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-$PARALLEL_OUT_ROOT/prepared/checkpoints}"
SHARD_DIR="${SHARD_DIR:-tmp/graphvae_prepare_shards/$DATASET_STEM}"
REMAINING_CSV="${REMAINING_CSV:-tmp/${DATASET_STEM}_remaining_for_prepare.csv}"

mkdir -p logs tmp "$SHARD_DIR" "$PARALLEL_OUT_ROOT"

if [ -n "${CANCEL_JOBID:-}" ]; then
  echo "[INFO] cancelling current job: $CANCEL_JOBID"
  scancel "$CANCEL_JOBID" || true
  while squeue -j "$CANCEL_JOBID" -h | grep -q .; do
    sleep 3
  done
fi

echo "[STEP] build remaining CSV (preserve checkpointed progress)"
"$PYTHON" -m bridging.graphvae.tools.build_remaining_dataset \
  --dataset "$DATASET" \
  --md-root "$MD_ROOT" \
  --deep-rank-hdf5 "$DEEPRANK_HDF5" \
  --checkpoint-dir "$BASE_CHECKPOINT_DIR" \
  --out-csv "$REMAINING_CSV" \
  --require-done \
  --require-graph

echo "[STEP] shard remaining CSV into $N_SHARDS"
"$PYTHON" -m bridging.dataProcessing.preshard_dataset \
  --dataset "$REMAINING_CSV" \
  --shard-dir "$SHARD_DIR" \
  --num-shards "$N_SHARDS"

echo "[STEP] submit shard prepare array"
PREP_JID=$(
  sbatch \
    --array="0-$((N_SHARDS - 1))" \
    --cpus-per-task="$CPUS_PER_SHARD" \
    --export=ALL,DATASET="$DATASET",SCRATCH_ROOT="$SCRATCH_ROOT",MD_ROOT="$MD_ROOT",PDB_CACHE_ROOT="$PDB_CACHE_ROOT",DEEPRANK_HDF5="$DEEPRANK_HDF5",SHARD_DIR="$SHARD_DIR",OUT_ROOT="$PARALLEL_OUT_ROOT" \
    hpc/graphvae_prepare_sharded_cpu.sbatch \
    | awk '{print $4}'
)
echo "[SUBMIT] PREP_JID=$PREP_JID"

echo "[STEP] submit merge job after shard prepare"
MERGE_JID=$(
  sbatch \
    --dependency="afterok:${PREP_JID}" \
    --export=ALL,DATASET="$DATASET",BASE_CHECKPOINT_DIR="$BASE_CHECKPOINT_DIR",SHARD_ROOT="$PARALLEL_OUT_ROOT",MERGED_PREPARED_DIR="$PARALLEL_OUT_ROOT/prepared" \
    hpc/graphvae_merge_prepared_cpu.sbatch \
    | awk '{print $4}'
)
echo "[SUBMIT] MERGE_JID=$MERGE_JID"

echo
echo "Monitor:"
echo "  squeue -j ${PREP_JID},${MERGE_JID} -o '%.18i %.2t %.10M %.10l %.20R'"
echo "  tail -f logs/graphvae_prep_shard_${PREP_JID}_*.out logs/graphvae_merge_prep_${MERGE_JID}.out"
echo
echo "Merged prepared output (for later GPU training):"
echo "  $PARALLEL_OUT_ROOT/prepared/graph_records.pt"
