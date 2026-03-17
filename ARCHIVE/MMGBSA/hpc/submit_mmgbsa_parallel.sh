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
DATASET_LABEL="$DATASET"
DATASET_STEM="$(basename "$DATASET" .csv)"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
MD_ROOT="${MD_ROOT:-$SCRATCH_ROOT/MD_datasets/$DATASET_STEM}"

N_SHARDS="${N_SHARDS:-8}"
CPUS_PER_SHARD="${CPUS_PER_SHARD:-1}"
TOTAL_CPUS=$(( N_SHARDS * CPUS_PER_SHARD ))
if [ "$TOTAL_CPUS" -gt 48 ]; then
  echo "[FAIL] requested total CPUs $TOTAL_CPUS exceeds shared-user cap 48."
  echo "[HINT] reduce N_SHARDS or CPUS_PER_SHARD."
  exit 1
fi

MMGBSA_PRESET="${MMGBSA_PRESET:-balanced_gb}"
case "$MMGBSA_PRESET" in
  balanced_gb)
    : "${SOLVATION_MODEL:=gb}"
    : "${START_FRAME:=21}"
    : "${INTERVAL:=5}"
    : "${SOURCE_MODE:=protein_traj}"
    : "${IGB:=5}"
    : "${SALTCON:=0.150}"
    : "${ISTRNG:=0.150}"
    ;;
  full_gb)
    : "${SOLVATION_MODEL:=gb}"
    : "${START_FRAME:=21}"
    : "${INTERVAL:=1}"
    : "${SOURCE_MODE:=full_traj}"
    : "${IGB:=5}"
    : "${SALTCON:=0.150}"
    : "${ISTRNG:=0.150}"
    ;;
  pb_pilot)
    : "${SOLVATION_MODEL:=pb}"
    : "${START_FRAME:=21}"
    : "${INTERVAL:=2}"
    : "${SOURCE_MODE:=full_traj}"
    : "${IGB:=5}"
    : "${SALTCON:=0.150}"
    : "${ISTRNG:=0.150}"
    ;;
  full_pb)
    : "${SOLVATION_MODEL:=pb}"
    : "${START_FRAME:=21}"
    : "${INTERVAL:=1}"
    : "${SOURCE_MODE:=full_traj}"
    : "${IGB:=5}"
    : "${SALTCON:=0.150}"
    : "${ISTRNG:=0.150}"
    ;;
  *)
    echo "[FAIL] unsupported MMGBSA_PRESET=$MMGBSA_PRESET"
    echo "[HINT] use one of: balanced_gb, full_gb, pb_pilot, full_pb"
    exit 1
    ;;
esac

END_FRAME="${END_FRAME:-}"

RUN_TAG="${RUN_TAG:-${SOLVATION_MODEL}_sf${START_FRAME}_int${INTERVAL}}"
SHARD_DIR="${MMGBSA_SHARD_DIR:-${SHARD_DIR:-tmp/mmgbsa_shards/${DATASET_STEM}_${RUN_TAG}}}"
OUT_ROOT="${MMGBSA_OUT_ROOT:-${OUT_ROOT:-src/bridging/generatedData/MMGBSA/${DATASET_STEM}_${RUN_TAG}}}"
WORK_ROOT="${MMGBSA_WORK_ROOT:-${WORK_ROOT:-$SCRATCH_ROOT/mmgbsa_work/${DATASET_STEM}_${RUN_TAG}}}"
MERGED_CSV="${MMGBSA_MERGED_CSV:-${MERGED_CSV:-$OUT_ROOT/${DATASET_STEM}_mmgbsa_estimates.csv}}"

mkdir -p logs tmp "$SHARD_DIR" "$OUT_ROOT"

SHARD_DATASET="$DATASET"
if [ -n "${LIMIT_ROWS:-}" ]; then
  SHARD_DATASET="tmp/${DATASET_STEM}_${RUN_TAG}_head${LIMIT_ROWS}.csv"
  echo "[STEP] build limited dataset rows=$LIMIT_ROWS -> $SHARD_DATASET"
  "$PYTHON" - "$DATASET" "$SHARD_DATASET" "$LIMIT_ROWS" <<'PY'
import sys
import pandas as pd
src, out, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(src).head(limit)
df.to_csv(out, index=False)
print(f"[MMGBSA] limited_rows={len(df)} out={out}")
PY
fi

echo "[STEP] shard dataset into $N_SHARDS"
"$PYTHON" -m bridging.dataProcessing.preshard_dataset \
  --dataset "$SHARD_DATASET" \
  --shard-dir "$SHARD_DIR" \
  --num-shards "$N_SHARDS" \
  --no-dedup-by-pdb

echo "[STEP] submit MMGBSA shard array"
MMGBSA_JID=$(
  sbatch \
    --array="0-$((N_SHARDS - 1))" \
    --cpus-per-task="$CPUS_PER_SHARD" \
    --export=ALL,DATASET="$SHARD_DATASET",DATASET_LABEL="$DATASET_LABEL",SCRATCH_ROOT="$SCRATCH_ROOT",MD_ROOT="$MD_ROOT",MMGBSA_SHARD_DIR="$SHARD_DIR",MMGBSA_OUT_ROOT="$OUT_ROOT",MMGBSA_WORK_ROOT="$WORK_ROOT",SOLVATION_MODEL="$SOLVATION_MODEL",START_FRAME="$START_FRAME",INTERVAL="$INTERVAL",END_FRAME="$END_FRAME",IGB="$IGB",SALTCON="$SALTCON",ISTRNG="$ISTRNG",SOURCE_MODE="$SOURCE_MODE" \
    hpc/mmgbsa_prefetch_sharded_cpu.sbatch \
    | awk '{print $4}'
)
echo "[SUBMIT] MMGBSA_JID=$MMGBSA_JID"

echo "[MMGBSA] preset=$MMGBSA_PRESET model=$SOLVATION_MODEL start=$START_FRAME end=${END_FRAME:-NA} interval=$INTERVAL source_mode=$SOURCE_MODE igb=$IGB saltcon=$SALTCON istrng=$ISTRNG"

echo "[STEP] submit merge job"
MERGE_JID=$(
  sbatch \
    --dependency="afterok:${MMGBSA_JID}" \
    --export=ALL,SHARD_ROOT="$OUT_ROOT",OUT_CSV="$MERGED_CSV" \
    hpc/mmgbsa_merge_cpu.sbatch \
    | awk '{print $4}'
)
echo "[SUBMIT] MERGE_JID=$MERGE_JID"

echo
echo "Monitor:"
echo "  squeue -j ${MMGBSA_JID},${MERGE_JID} -o '%.18i %.2t %.10M %.10l %.20R'"
echo "  tail -f logs/mmgbsa_shard_${MMGBSA_JID}_*.out logs/mmgbsa_merge_${MERGE_JID}.out"
echo
echo "Merged MMGBSA CSV:"
echo "  $MERGED_CSV"
