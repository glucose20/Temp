#!/bin/bash
#SBATCH --job-name=kb1281e4      # Name of the job
#SBATCH --nodes=1                  # Use 1 node
#SBATCH --ntasks=1                 # Two separate tasks
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8          # Split CPU cores between jobs
#SBATCH --mem=20G                  # Total memory for both jobs
#SBATCH --time=120:00:00            # Time limit
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=s226181148@deakin.edu.au
#SBATCH --array=0-1


# --- Environment setup ---
module purge
module load Anaconda3
source activate
conda activate esm_thuy

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=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

# --- Define Parameters ---
# DATASETS=("davis" "metz")
RUNNING_SETS=("warm" "novel-drug" "novel-pair" "novel-prot")
# FOLDS=(0 1 2 3 4)

# --- Indexing Logic to map 0-59 to Dataset/RunningSet/Fold ---
# ID = (dataset_idx * 20) + (runset_idx * 5) + fold_idx
# dataset_idx=$(( SLURM_ARRAY_TASK_ID / 40 ))
# runset_idx=$(( (SLURM_ARRAY_TASK_ID % 40) / 10 ))
# fold_idx=$(( (SLURM_ARRAY_TASK_ID % 10) / 2 ))
# lr_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))

# DATASET=${DATASETS[$dataset_idx]}
# RUNNING_SET=${RUNNING_SETS[$runset_idx]}
# FOLD=${FOLDS[$fold_idx]}
# LEARNING_RATE=${LEARNING_RATES[$lr_idx]}


DATASET=("kiba")
EPOCHS=500
MAX_PATIENCE=30
LEARNING_RATE=1e-4
FOLD=0


BATCH_SIZE=128
runset_idx_1=$(( SLURM_ARRAY_TASK_ID * 2))
runset_idx_2=$(( SLURM_ARRAY_TASK_ID * 2 + 1 ))
RUNNING_SET_1=${RUNNING_SETS[$runset_idx_1]}
RUNNING_SET_2=${RUNNING_SETS[$runset_idx_2]}

ROOT_DIR="sweep_double_logs"
mkdir -p "$ROOT_DIR"

echo "======================================"
# echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset:       $DATASET"
echo "Running set:   $RUNNING_SET"
echo "Fold:          $FOLD"
echo "======================================"


declare -a PIDS
declare -a LOG_FILES


LOG_DIR_1="${ROOT_DIR}/${DATASET}/${RUNNING_SET_1}/fold${FOLD}"
mkdir -p "${LOG_DIR_1}"
LOG_DIR_2="${ROOT_DIR}/${DATASET}/${RUNNING_SET_2}/fold${FOLD}"
mkdir -p "${LOG_DIR_2}"


LOG_PREFIX_1="${LOG_DIR_1}/lr${LEARNING_RATE}_b${BATCH_SIZE}_jid${SLURM_ARRAY_JOB_ID}_tid${SLURM_ARRAY_TASK_ID}"
LOG_PREFIX_2="${LOG_DIR_2}/lr${LEARNING_RATE}_b${BATCH_SIZE}_jid${SLURM_ARRAY_JOB_ID}_tid${SLURM_ARRAY_TASK_ID}"

# # 1. Start Job A in the background
python scripts/ab_testing_extra.py \
        --dataset "$DATASET" \
        --running_set "$RUNNING_SET_1" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --max_patience "$MAX_PATIENCE" \
        --learning_rate "$LEARNING_RATE" \
        --fold "$FOLD" \
        --cuda "$gpu" > "$LOG_PREFIX_1" 2>&1 &

# Capture the PID of the process just started
current_pid=$!
PIDS+=("$current_pid")
LOG_FILES+=("$LOG_PREFIX_1")

sleep 2

python scripts/ab_testing_extra.py \
        --dataset "$DATASET" \
        --running_set "$RUNNING_SET_2" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --max_patience "$MAX_PATIENCE" \
        --learning_rate "$LEARNING_RATE" \
        --fold "$FOLD" \
        --cuda "$gpu" > "$LOG_PREFIX_2" 2>&1 &

# Capture the PID of the process just started
current_pid=$!
PIDS+=("$current_pid")
LOG_FILES+=("$LOG_PREFIX_2")


echo "Waiting for ${#PIDS[@]} background jobs to complete..."

# Corrected Waiting Logic
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    log=${LOG_FILES[$i]}
    
    wait "$pid"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[SUCCESS] Job PID $pid finished. (Log: $(basename "$log"))"
    else
        echo "[ERROR] Job PID $pid failed with code $exit_code. Check $log"
    fi
done

echo "All jobs finished. Displaying tail of logs..."
for log in "${LOG_FILES[@]}"; do
    echo "--- Last 5 lines of $(basename "$log") ---"
    tail -n 5 "$log"
done