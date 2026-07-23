#!/bin/bash
#SBATCH --job-name=kiba_fnet_%a
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-large
#SBATCH --exclude=h100-m-01
#SBATCH --gpus=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=s226181148@deakin.edu.au
#SBATCH --array=0-19

module purge
module load Anaconda3
source activate
conda activate esm_thuy

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=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

RUNNING_SETS=("warm" "novel-drug" "novel-pair" "novel-prot")
DATASET="kiba"
RUNNING_SET=${RUNNING_SETS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=$((SLURM_ARRAY_TASK_ID % 5))

LEARNING_RATE="5e-4"
BATCH_SIZE=256
EPOCHS=500
MAX_PATIENCE=10
NUM_EXPERTS=6
TOP_K=2

RESULTS_ROOT="fnet_ab_results_20jobs/${DATASET}/${RUNNING_SET}/fold${FOLD}"
LOG_DIR="fnet_ab_logs_20jobs/${DATASET}/${RUNNING_SET}/fold${FOLD}"
mkdir -p "$LOG_DIR"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/${timestamp}_jid${SLURM_ARRAY_JOB_ID}_tid${SLURM_ARRAY_TASK_ID}.log"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${DATASET}/${RUNNING_SET}, fold ${FOLD}"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python scripts/ab_testing_fnet.py \
    --dataset "$DATASET" \
    --running_set "$RUNNING_SET" \
    --fold "$FOLD" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_patience "$MAX_PATIENCE" \
    --learning_rate "$LEARNING_RATE" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --cuda 0 \
    --models "baseline" \
    --amp --amp_dtype bf16 \
    --results_root "$RESULTS_ROOT" \
    > "$log_file" 2>&1
status=$?

if ((status != 0)); then
    echo "Task failed with status ${status}; see ${log_file}"
    exit "$status"
fi
echo "Task completed; log: ${log_file}"
