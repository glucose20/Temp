#!/bin/bash
#SBATCH --job-name=fft_abla
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gpus=h100:1
#SBATCH --exclude=h100-m-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=60:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=s226181148@deakin.edu.au
#SBATCH --array=0-19


# --- Environment setup ---
module purge
module load Anaconda3
source activate
conda activate esm_thuy

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=657df3c06ebe7d9b611a9e81fa9d72eb0e9c76b9

# --- Fixed experiment config ---
# Change this once if you want the same 4-setting sweep on a different dataset.
DATASET="davis"
RUNNING_SETS=("warm" "novel-drug" "novel-pair" "novel-prot")

RUNNING_SET=${RUNNING_SETS[$((SLURM_ARRAY_TASK_ID / 5))]}
EPOCHS=200
PATIENCE=30
BATCH_SIZE=256
LR=1e-4
NUM_EXPERTS=4
TOP_K=2
MOE_NOISE_STD=0.1
LOAD_BALANCE_WEIGHT=0.01
FOLD = $((SLURM_ARRAY_TASK_ID % 5))

LOG_DIR="ablation_fft_log_1e4"
OUTPUT_DIR="ablation_fft_results_1e4/${DATASET}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

JID="${SLURM_JOB_ID:-local$$}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/id${JID}_${TIMESTAMP}_fft_${DATASET}_${RUNNING_SET}.log"

echo "======================================"
echo "Array Task:   $SLURM_ARRAY_TASK_ID"
echo "Dataset:      $DATASET"
echo "Running set:  $RUNNING_SET"
echo "Timestamp:    $TIMESTAMP"
echo "======================================"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    python code/ablation_fft.py \
        --dataset "$DATASET" \
        --running_set "$RUNNING_SET" \
        --fold "$FOLD" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --num_experts "$NUM_EXPERTS" \
        --top_k "$TOP_K" \
        --moe_noise_std "$MOE_NOISE_STD" \
        --load_balance_weight "$LOAD_BALANCE_WEIGHT" \
        --output_dir "$OUTPUT_DIR" \
        --cuda 0 \
        > "$LOG_FILE" 2>&1

echo "Done: $DATASET / $RUNNING_SET"