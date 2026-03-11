#!/bin/bash
#SBATCH --job-name=kbne_1e4
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=s226181148@deakin.edu.au
#SBATCH --array=0-3

# --- Environment setup ---
module purge
module load Anaconda3
source activate
conda activate esm_thuy

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=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

# --- Configuration ---
# Configuration
RUNNING_SETS=("warm" "novel-drug" "novel-pair" "novel-prot")
runset_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))
RUNNING_SET=${RUNNING_SETS[$runset_idx]}

DATASET="kiba"
LEARNING_RATE=1e-4
# BATCH_SIZES=(16 32 64 128)
BATCH_SIZE=64
NUM_FOLDS=5


EPOCHS=500
MAX_PATIENCE=30
LOG_DIR="./sweep_no_encoder_logs"
mkdir -p "$LOG_DIR"
mkdir -p ./savemodel

# --- Parallel Management ---
# How many jobs to run at once on the single H100. 
# 8 might be too many for VRAM; 4 is usually safer.
declare -a PIDS
declare -a LOG_FILES

echo "Starting training for $RUNNING_SET..."

for ((fold=0; fold<$NUM_FOLDS; fold++)); do
    # for bs in "${BATCH_SIZES[@]}"; do
    bs=$BATCH_SIZE
    gpu=0  # All jobs share the single
    
    TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
    LOG_PATH="${LOG_DIR}/${DATASET}/${RUNNING_SET}"
    mkdir -p "${LOG_PATH}"

    LOG_FILE_NAME="${DATASET}_${RUNNING_SET}_fold${fold}_b${batch_size}_lr${LEARNING_RATE}_jid${SLURM_ARRAY_JOB_ID}_tid${SLURM_ARRAY_TASK_ID}"
    FULL_LOG_PATH="${LOG_PATH}/${TIMESTAMP}_${LOG_FILE_NAME}.log"

    echo "Launching: Fold $fold, BS $bs, LR $LEARNING_RATE on GPU $gpu..."
    
    # Start training in background
    python scripts/ab_testing_no_encoder.py \
            --dataset "$DATASET" \
            --running_set "$RUNNING_SET" \
            --epochs "$EPOCHS" \
            --batch_size "$bs" \
            --max_patience "$MAX_PATIENCE" \
            --learning_rate "$LEARNING_RATE" \
            --fold "$fold" \
            --cuda "$gpu" > "$FULL_LOG_PATH" 2>&1 &
    
    # Save PID and log path
    current_pid=$!
    PIDS+=("$current_pid")
    LOG_FILES+=("$FULL_LOG_PATH")
    
    sleep 2
    # done
done

echo "------------------------------------------------------------"
echo "All 8 sub-jobs submitted. Waiting for completion..."
echo "------------------------------------------------------------"

# --- Wait for all PIDs in this array task ---
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    log=${LOG_FILES[$i]}
    
    # Check if process is still running
    if kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[SUCCESS] PID $pid finished."
        else
            echo "[ERROR] PID $pid failed (Exit $exit_code). Check log: $log"
        fi
    else
        echo "Job PID $pid already finished."
    fi
done

echo "============================================================"
echo "Array Task $SLURM_ARRAY_TASK_ID (Fold $FOLD) finished!"
echo "============================================================"