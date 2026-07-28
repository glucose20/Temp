#!/bin/bash
# DAVIS: four jobs per dataset, one per running setting.
# Each array job runs all five folds sequentially.
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=16
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=s226181148@deakin.edu.au
#SBATCH --array=0-23

set -u

module purge
module load Anaconda3
source activate
conda activate esm_thuy
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=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

DATASET="davis"
RUNNING_SETS=("warm" "novel-drug" "novel-pair" "novel-prot")
MODELS=("fnet_moe" "moe" "baseline")
LEARNING_RATES=("1e-3" "5e-5")
LEARNING_RATE=${LEARNING_RATES[$((SLURM_ARRAY_TASK_ID / 12))]}
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 4 % 3))]}
RUNNING_SET=${RUNNING_SETS[$((SLURM_ARRAY_TASK_ID % 4))]}
BATCH_SIZE=256
NUM_FOLDS=5
EPOCHS=500
MAX_PATIENCE=30
NUM_EXPERTS=4
TOP_K=2
MOL_EMBED_TYPE="molformer"
USE_ESMC=true
ESMC_MODEL="esm3"

RESULTS_ROOT="fnet_ab_results_esm3_dm/${LEARNING_RATE}/${DATASET}/${RUNNING_SET}/${MODEL}"
LOG_DIR="fnet_ab_logs_esm3_dm/${LEARNING_RATE}/${DATASET}/${RUNNING_SET}/${MODEL}"
mkdir -p "$LOG_DIR"

# Each fold runs the three models sequentially because this job owns one GPU.
for ((fold=0; fold<NUM_FOLDS; fold++)); do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="${LOG_DIR}/${timestamp}_fold${fold}_jid${SLURM_ARRAY_JOB_ID}_tid${SLURM_ARRAY_TASK_ID}.log"
    echo "Starting ${DATASET}/${RUNNING_SET}, fold ${fold}, model ${MODEL}"
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        python scripts/ab_testing_fnet.py \
        --dataset "$DATASET" \
        --running_set "$RUNNING_SET" \
        --fold "$fold" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --max_patience "$MAX_PATIENCE" \
        --learning_rate "$LEARNING_RATE" \
        --num_experts "$NUM_EXPERTS" \
        --top_k "$TOP_K" \
        --cuda 0 \
        --results_root "$RESULTS_ROOT" \
        --models "$MODEL" \
        --mol_embed_type "$MOL_EMBED_TYPE" \
        --use_esmc \
        --esmc_model "$ESMC_MODEL" \
        > "$log_file" 2>&1
    status=$?
    if ((status != 0)); then
        echo "Fold ${fold} failed with status ${status}; see ${log_file}"
        exit "$status"
    fi
    echo "Completed fold ${fold}; log: ${log_file}"
done
