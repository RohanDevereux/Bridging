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

INPUT_BASE_CHECKPOINT_DIR="${INPUT_BASE_CHECKPOINT_DIR:-src/bridging/generatedData/graphvae/${DATASET_STEM}_torsion_input_v1/base/prepared/checkpoints}"
INPUT_SHARD_ROOT="${INPUT_SHARD_ROOT:-src/bridging/generatedData/graphvae/${DATASET_STEM}_torsion_input_v1}"
OUT_ROOT="${FORCE_FEATURE_OUT_ROOT:-${OUT_ROOT:-src/bridging/generatedData/graphvae/${DATASET_STEM}_torsion_force_input_v1}}"
MANIFEST="${FORCE_FEATURE_MANIFEST:-${MANIFEST:-tmp/graphvae_force_feature_manifest_${DATASET_STEM}.tsv}}"

mkdir -p logs tmp "$OUT_ROOT"

: > "$MANIFEST"

if [ -d "$INPUT_BASE_CHECKPOINT_DIR" ] && find "$INPUT_BASE_CHECKPOINT_DIR" -maxdepth 1 -type f -name 'records_*.pt' | grep -q .; then
  echo -e "base\t$INPUT_BASE_CHECKPOINT_DIR\t$OUT_ROOT/base/prepared/checkpoints\t$OUT_ROOT/base/prepared/force_feature_report.json" >> "$MANIFEST"
fi

while IFS= read -r SHARD_DIR; do
  SHARD_NAME="$(basename "$(dirname "$(dirname "$SHARD_DIR")")")"
  case "$SHARD_NAME" in
    base|shard_*)
      echo -e "${SHARD_NAME}\t${SHARD_DIR}\t$OUT_ROOT/${SHARD_NAME}/prepared/checkpoints\t$OUT_ROOT/${SHARD_NAME}/prepared/force_feature_report.json" >> "$MANIFEST"
      ;;
  esac
done < <(find "$INPUT_SHARD_ROOT" -type d -path '*/shard_*/prepared/checkpoints' | sort)

PARTS=$(wc -l < "$MANIFEST" | awk '{print $1}')
if [ "$PARTS" -lt 1 ]; then
  echo "[FAIL] no force-feature input checkpoint parts found."
  echo "[HINT] checked base=$INPUT_BASE_CHECKPOINT_DIR and shard_root=$INPUT_SHARD_ROOT"
  exit 1
fi

if [ "${CPUS_PER_TASK:-1}" -gt 48 ]; then
  echo "[FAIL] CPUS_PER_TASK exceeds shared-user cap 48."
  exit 1
fi

echo "[STEP] build force-feature manifest parts=$PARTS -> $MANIFEST"
cat "$MANIFEST"

echo "[STEP] submit force-feature array"
FORCE_JID=$(
  sbatch \
    --array="0-$((PARTS - 1))" \
    --cpus-per-task="${CPUS_PER_TASK:-1}" \
    --export=ALL,MANIFEST="$MANIFEST",DATASET="$DATASET",MD_ROOT="$MD_ROOT",PDB_CACHE_ROOT="$PDB_CACHE_ROOT",MAX_FRAMES="${MAX_FRAMES:-120}",TRAJ_CACHE_SIZE="${TRAJ_CACHE_SIZE:-1}",PROGRESS_EVERY="${PROGRESS_EVERY:-10}" \
    hpc/graphvae_augment_force_features_part_cpu.sbatch \
    | awk '{print $4}'
)
echo "[SUBMIT] FORCE_JID=$FORCE_JID"

MERGE_BASE=""
if grep -q '^base[[:space:]]' "$MANIFEST"; then
  MERGE_BASE="$OUT_ROOT/base/prepared/checkpoints"
fi

echo "[STEP] submit merge job"
MERGE_JID=$(
  sbatch \
    --dependency="afterok:${FORCE_JID}" \
    --export=ALL,DATASET="$DATASET",BASE_CHECKPOINT_DIR="$MERGE_BASE",SHARD_ROOT="$OUT_ROOT",MERGED_PREPARED_DIR="$OUT_ROOT/prepared" \
    hpc/graphvae_merge_prepared_cpu.sbatch \
    | awk '{print $4}'
)
echo "[SUBMIT] MERGE_JID=$MERGE_JID"

echo
echo "Monitor:"
echo "  squeue -j ${FORCE_JID},${MERGE_JID} -o '%.18i %.2t %.10M %.10l %.20R'"
echo "  tail -f logs/graphvae_force_${FORCE_JID}_*.out logs/graphvae_merge_prep_${MERGE_JID}.out"
echo
echo "Merged force-feature prepared output:"
echo "  $OUT_ROOT/prepared/graph_records.pt"
