#!/bin/bash
#SBATCH --job-name=abla_dta
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=s226181148@deakin.edu.au
#SBATCH --array=0-4


# --- Environment setup ---
module purge
module load Anaconda3
source activate
conda activate esm_thuy

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=657df3c06ebe7d9b611a9e81fa9d72eb0e9c76b9

# --- Define Variables ---
LOG_DIR="ablation_log"
mkdir -p "$LOG_DIR"

# ---------------change parameters---------------------
DATASET=("davis")
RUNNING_SETS=("novel-pair" "novel-drug" "novel-prot")
NUM_EXPERTS=(4 1 4)       
TOP_KS=(1 1 1)
# LEARNING_RATES=(5e-5 1e-4 1e-4)
LEARNING_RATES=(5e-5 1e-4 5e-5)
# BATCH_SIZES=(16 16 32)
BATCH_SIZES=(16 16 16)
MOE_NOISE_STDS=(0.1 0.0 0.1)
LOAD_BALANCE_WEIGHTS=(0.01 0.0 0.01)

# idx=$(( SLURM_ARRAY_TASK_ID % 3 ))
# RUNNING_SET=${RUNNING_SETS[$idx]}
# NUM_EXPERT=${NUM_EXPERTS[$idx]}
# TOP_K=${TOP_KS[$idx]}
# LEARNING_RATE=${LEARNING_RATES[$idx]}
# BATCH_SIZE=${BATCH_SIZES[$idx]}
# MOE_NOISE_STD=${MOE_NOISE_STDS[$idx]}   
# LOAD_BALANCE_WEIGHT=${LOAD_BALANCE_WEIGHTS[$idx]}

# NUM_FOLDS=5
NUM_SET=3


#---------------pre-assign parameters--------------------
EPOCHS=150
JID="${SLURM_JOB_ID:-local$$}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
FOLD=$SLURM_ARRAY_TASK_ID

echo "======================================"
echo "Dataset:     $DATASET"
echo "Running set: $RUNNING_SET"
echo "Timestamp:   $TIMESTAMP"
echo "======================================"


# srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
#      python code/ablation_component.py \
#         --dataset "$DATASET" \
#         --running_set "$RUNNING_SET" \
#         --epochs "$EPOCHS" \
#         --fold "$FOLD" \
#         --cuda 0 \
#         > "${LOG_PREFIX}.out" \
#         2> "${LOG_PREFIX}.err"
        

# for ((rs=0; rs<$NUM_SET; rs++)); do

rs=2

RUNNING_SET=${RUNNING_SETS[$rs]}
NUM_EXPERT=${NUM_EXPERTS[$rs]}
TOP_K=${TOP_KS[$rs]}
LEARNING_RATE=${LEARNING_RATES[$rs]}
BATCH_SIZE=${BATCH_SIZES[$rs]}
MOE_NOISE_STD=${MOE_NOISE_STDS[$rs]}   
LOAD_BALANCE_WEIGHT=${LOAD_BALANCE_WEIGHTS[$rs]}


gpu=0  # All jobs share the single
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

LOG_FILE="ablation_${DATASET}_${RUNNING_SET}_fold${FOLD}"
LOG_PREFIX="${LOG_DIR}/id${JID}_${TIMESTAMP}_${LOG_FILE}.log"

echo "Launching: Fold $FOLD on GPU $gpu..."

# Start training in background
python code/ablation_component.py \
        --dataset "$DATASET" \
        --running_set "$RUNNING_SET" \
        --num_experts "$NUM_EXPERT" \
        --top_k "$TOP_K" \
        --moe_noise_std "$MOE_NOISE_STD" \
        --load_balance_weight "$LOAD_BALANCE_WEIGHT" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --epochs "$EPOCHS" \
        --fold "$FOLD" \
        --cuda "$gpu" > "$LOG_PREFIX" 2>&1 

# Save PID and log path
# current_pid=$!
# PIDS+=("$current_pid")
# LOG_FILES+=("$LOG_PREFIX")

        # sleep 2
# done

# echo "------------------------------------------------------------"
# echo "All 8 sub-jobs submitted. Waiting for completion..."
# echo "------------------------------------------------------------"

# # --- Wait for all PIDs in this array task ---
# for i in "${!PIDS[@]}"; do
#     pid=${PIDS[$i]}
#     log=${LOG_FILES[$i]}
    
#     # Check if process is still running
#     if kill -0 "$pid" 2>/dev/null; then
#         wait "$pid"
#         exit_code=$?
#         if [ $exit_code -eq 0 ]; then
#             echo "[SUCCESS] PID $pid finished."
#         else
#             echo "[ERROR] PID $pid failed (Exit $exit_code). Check log: $log"
#         fi
#     else
#         echo "Job PID $pid already finished."
#     fi
# done

echo "============================================================"
echo "Array Task $SLURM_ARRAY_TASK_ID (Fold $FOLD) finished!"
echo "============================================================"
