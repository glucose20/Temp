#!/bin/bash
#SBATCH --job-name=p_mz-npr
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


# --- Environment setup ---
module purge
module load Anaconda3
source activate
conda activate esm_thuy

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=657df3c06ebe7d9b611a9e81fa9d72eb0e9c76b9

# --- Define Variables ---
LOG_DIR="pre_sweep_abs_log"
mkdir -p "$LOG_DIR"

# ---------------Fixed parameters---------------------
DATASET="metz"
RUNNING_SET="novel-pair"
FOLD=0
EPOCHS=200

#----------------Sweep parameters------------------------
LOAD_BALANCE_WEIGHTS=(0.01)
MOE_NOISE_STDs=(0.1)
LEARNING_RATES=(5e-5 1e-4 2e-4 5e-4)
BATCH_SIZES=(256 128 64 16)

#---------------pre-assign parameters--------------------
BATCH_SIZE=256
LR_INDEX=0
LEARNING_RATE="${LEARNING_RATES[$LR_INDEX]}"

JID="${SLURM_JOB_ID:-local$$}"

for MOE_NOISE_STD in "${MOE_NOISE_STDs[@]}"; do
  for LOAD_BALANCE_WEIGHT in "${LOAD_BALANCE_WEIGHTS[@]}"; do

    TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

    LOG_FILE="sweep_ab_${DATASET}_${RUNNING_SET}_b${BATCH_SIZE}_lr${LEARNING_RATE}_moestd${MOE_NOISE_STD}_lbw${LOAD_BALANCE_WEIGHT}"
    LOG_PREFIX="${LOG_DIR}/id${JID}_${TIMESTAMP}_${LOG_FILE}"

    echo "======================================"
    echo "Fold:        $FOLD"
    echo "Dataset:     $DATASET"
    echo "Running set: $RUNNING_SET"
    echo "Timestamp:   $TIMESTAMP"
    echo "Logs:        ${LOG_PREFIX}.out / .err"
    echo "======================================"

    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
      python scripts/ab_testing_colab_presweep.py \
        --dataset "$DATASET" \
        --running_set "$RUNNING_SET" \
        --fold "$FOLD" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --moe_noise_std "$MOE_NOISE_STD" \
        --load_balance_weight "$LOAD_BALANCE_WEIGHT" \
        --cuda 0 \
        > "${LOG_PREFIX}.out" \
        2> "${LOG_PREFIX}.err"

  done
done
