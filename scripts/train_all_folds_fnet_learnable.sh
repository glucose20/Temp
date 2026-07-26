#!/bin/bash
# DAVIS + METZ: eight jobs total, one per (dataset, running_set) pair.
# Each array job runs all five folds sequentially on a single A100.
#SBATCH --job-name=dm_fnetl_%a
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=s226181148@deakin.edu.au
#SBATCH --array=0-7

set -u

module purge
module load Anaconda3
source activate
conda activate esm_thuy
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=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

DATASETS=("davis" "metz")
RUNNING_SETS=("warm" "novel-drug" "novel-pair" "novel-prot")
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID / 4))]}
RUNNING_SET=${RUNNING_SETS[$((SLURM_ARRAY_TASK_ID % 4))]}
LEARNING_RATE="5e-4"
BATCH_SIZE=256
NUM_FOLDS=5
EPOCHS=500
MAX_PATIENCE=30
NUM_EXPERTS=4
TOP_K=2
RESULTS_ROOT="fnet_learnable_ab_results_4jobs_per_dataset/${LEARNING_RATE}/${DATASET}/${RUNNING_SET}"
LOG_DIR="fnet_learnable_ab_logs_4jobs_per_dataset/${LEARNING_RATE}/${DATASET}/${RUNNING_SET}"
mkdir -p "$LOG_DIR"

# Each fold runs sequentially because this job owns one GPU.
for ((fold=0; fold<NUM_FOLDS; fold++)); do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="${LOG_DIR}/${timestamp}_fold${fold}_jid${SLURM_ARRAY_JOB_ID}_tid${SLURM_ARRAY_TASK_ID}.log"
    echo "Starting ${DATASET}/${RUNNING_SET}, fold ${fold}"
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        python scripts/ab_testing_fnet_learnable.py \
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
        --models "fnet_learnable" \
        --amp --amp_dtype bf16 \
        --results_root "$RESULTS_ROOT" \
        > "$log_file" 2>&1
    status=$?
    if ((status != 0)); then
        echo "Fold ${fold} failed with status ${status}; see ${log_file}"
        exit "$status"
    fi
    echo "Completed fold ${fold}; log: ${log_file}"
done
