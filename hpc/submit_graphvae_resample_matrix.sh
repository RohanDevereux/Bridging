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
if [ ! -x "$PYTHON" ] && [ -x "$HOME/Bridging/.venv/bin/python" ]; then
  PYTHON="$HOME/Bridging/.venv/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || command -v python)"
fi
export PYTHONPATH="${PYTHONPATH:-src}"

DATASET="${DATASET:-src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv}"
DATASET_STEM="$(basename "$DATASET" .csv)"
VIEW_ROOT="${VIEW_ROOT:-src/bridging/generatedData/graphvae/${DATASET_STEM}_views_force_mdiface_v3}"
FULL_RECORDS="${FULL_RECORDS:-$VIEW_ROOT/md_closest_pair_patch/graph_records_full.pt}"
IFACE_RECORDS="${IFACE_RECORDS:-$VIEW_ROOT/md_closest_pair_patch/graph_records_interface.pt}"
OUT_ROOT="${OUT_ROOT:-src/bridging/generatedData/graphvae/${DATASET_STEM}_resample_matrix_v1}"

GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-8}"
FOLDS="${FOLDS:-5}"
REPEATS="${REPEATS:-2}"
if [ -z "${VAL_FRACTION_OF_TRAINVAL:-}" ]; then
  VAL_FRACTION_OF_TRAINVAL="$(awk -v k="$FOLDS" 'BEGIN { printf "%.6f", 0.15 / (1.0 - 1.0 / k) }')"
fi
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"

MAX_EPOCHS="${MAX_EPOCHS:-120}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-3}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
MASK_RATIO="${MASK_RATIO:-0.30}"
BETA_START="${BETA_START:-0.0}"
BETA_END="${BETA_END:-1.0}"
BETA_ANNEAL_FRACTION="${BETA_ANNEAL_FRACTION:-0.30}"
CORR_WEIGHT="${CORR_WEIGHT:-0.01}"
AFFINITY_WEIGHT="${AFFINITY_WEIGHT:-1.0}"
ALPHA_GRID="${ALPHA_GRID:-1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100}"
RIDGE_CV_FOLDS="${RIDGE_CV_FOLDS:-0}"
RIDGE_CV_REPEATS="${RIDGE_CV_REPEATS:-0}"
RIDGE_CV_INNER_FOLDS="${RIDGE_CV_INNER_FOLDS:-5}"
INCLUDE_BASELINES="${INCLUDE_BASELINES:-true}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p logs "$OUT_ROOT"
MANIFEST="$OUT_ROOT/submitted_jobs.tsv"
echo -e "job_id\ttag\tmodel_family\tview\tmode\tsupervision\ttarget_policy\tlatent_dim\tout_dir" > "$MANIFEST"

if [ ! -f "$FULL_RECORDS" ]; then
  echo "[FAIL] Missing full records: $FULL_RECORDS"
  exit 1
fi
if [ ! -f "$IFACE_RECORDS" ]; then
  echo "[FAIL] Missing interface records: $IFACE_RECORDS"
  exit 1
fi

declare -a JOB_IDS=()

submit_job() {
  local tag="$1"
  local model_family="$2"
  local view="$3"
  local mode="$4"
  local supervision="$5"
  local target_policy="$6"
  local latent_dim="$7"
  local records="$8"
  local out_dir="$OUT_ROOT/$tag"

  local jid="DRYRUN"
  if [ "$DRY_RUN" != "true" ]; then
    jid=$(
      env \
        PYTHON="$PYTHON" \
        DATASET="$DATASET" \
        RECORDS="$records" \
        OUT_DIR="$out_dir" \
        MODEL_FAMILY="$model_family" \
        MODE="$mode" \
        SUPERVISION_MODE="$supervision" \
        TARGET_POLICY="$target_policy" \
        LATENT_DIM="$latent_dim" \
        TAG="$tag" \
        REPEATS="$REPEATS" \
        FOLDS="$FOLDS" \
        VAL_FRACTION_OF_TRAINVAL="$VAL_FRACTION_OF_TRAINVAL" \
        SEED="$SEED" \
        DEVICE="$DEVICE" \
        MAX_EPOCHS="$MAX_EPOCHS" \
        PATIENCE="$PATIENCE" \
        BATCH_SIZE="$BATCH_SIZE" \
        NUM_WORKERS="$NUM_WORKERS" \
        HIDDEN_DIM="$HIDDEN_DIM" \
        NUM_LAYERS="$NUM_LAYERS" \
        LR="$LR" \
        WEIGHT_DECAY="$WEIGHT_DECAY" \
        MASK_RATIO="$MASK_RATIO" \
        BETA_START="$BETA_START" \
        BETA_END="$BETA_END" \
        BETA_ANNEAL_FRACTION="$BETA_ANNEAL_FRACTION" \
        CORR_WEIGHT="$CORR_WEIGHT" \
        AFFINITY_WEIGHT="$AFFINITY_WEIGHT" \
        ALPHA_GRID="$ALPHA_GRID" \
        RIDGE_CV_FOLDS="$RIDGE_CV_FOLDS" \
        RIDGE_CV_REPEATS="$RIDGE_CV_REPEATS" \
        RIDGE_CV_INNER_FOLDS="$RIDGE_CV_INNER_FOLDS" \
        sbatch \
        --cpus-per-task="$GPU_CPUS_PER_TASK" \
        --export=ALL \
        hpc/graphvae_resample_gpu.sbatch \
        | awk '{print $4}'
    )
    JOB_IDS+=("$jid")
  fi

  echo -e "${jid}\t${tag}\t${model_family}\t${view}\t${mode}\t${supervision}\t${target_policy}\t${latent_dim}\t${out_dir}" >> "$MANIFEST"
  echo "[SUBMIT] jid=$jid tag=$tag"
}

for view in full interface; do
  if [ "$view" = "full" ]; then
    records="$FULL_RECORDS"
    view_short="full"
  else
    records="$IFACE_RECORDS"
    view_short="iface"
  fi

  for supervision in unsupervised semi_supervised; do
    if [ "$supervision" = "unsupervised" ]; then
      sup_short="unsup"
    else
      sup_short="semi"
    fi

    for latent_dim in 8 16 32; do
      submit_job "${view_short}_S_${sup_short}_shared_static_z${latent_dim}" "vae_ridge" "$view" "S" "$supervision" "shared_static" "$latent_dim" "$records"
      submit_job "${view_short}_SD_${sup_short}_shared_static_z${latent_dim}" "vae_ridge" "$view" "SD" "$supervision" "shared_static" "$latent_dim" "$records"
      submit_job "${view_short}_SD_${sup_short}_sd_dynamic_all_z${latent_dim}" "vae_ridge" "$view" "SD" "$supervision" "sd_dynamic_all" "$latent_dim" "$records"
      submit_job "${view_short}_SD_${sup_short}_sd_static_plus_dynamic_all_z${latent_dim}" "vae_ridge" "$view" "SD" "$supervision" "sd_static_plus_dynamic_all" "$latent_dim" "$records"
    done
  done
done

if [ "$INCLUDE_BASELINES" = "true" ]; then
  submit_job "full_base_S" "supervised_baseline" "full" "S" "supervised" "" "0" "$FULL_RECORDS"
  submit_job "full_base_SD" "supervised_baseline" "full" "SD" "supervised" "" "0" "$FULL_RECORDS"
  submit_job "iface_base_S" "supervised_baseline" "interface" "S" "supervised" "" "0" "$IFACE_RECORDS"
  submit_job "iface_base_SD" "supervised_baseline" "interface" "SD" "supervised" "" "0" "$IFACE_RECORDS"
fi

echo
echo "[INFO] manifest=$MANIFEST"
if [ "${#JOB_IDS[@]}" -gt 0 ]; then
  joined="$(IFS=,; echo "${JOB_IDS[*]}")"
  echo "[INFO] monitor: squeue -j $joined -o '%.18i %.2t %.10M %.10l %.20R %.30j'"
else
  echo "[INFO] dry run only. No jobs submitted."
fi
