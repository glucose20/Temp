#!/bin/bash
#SBATCH --job-name=dta_%a
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gpus=h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=s226181148@deakin.edu.au
#SBATCH --array=0-39

# --- Environment setup ---
module purge
module load Anaconda3
source activate
conda activate esm_thuy

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=657df3c06ebe7d9b611a9e81fa9d72eb0e9c76b9

# --- Define Parameters ---
# DATASETS=("davis" "metz")
DATASETS=("kiba")
RUNNING_SETS=("warm" "novel-drug" "novel-pair" "novel-prot")
FOLDS=(0 1 2 3 4)
LEARNING_RATES=(1e-4 5e-5)

# --- Indexing Logic to map 0-59 to Dataset/RunningSet/Fold ---
# ID = (dataset_idx * 20) + (runset_idx * 5) + fold_idx
dataset_idx=$(( SLURM_ARRAY_TASK_ID / 40 ))
runset_idx=$(( (SLURM_ARRAY_TASK_ID % 40) / 10 ))
fold_idx=$(( (SLURM_ARRAY_TASK_ID % 10) / 2 ))
lr_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))

DATASET=${DATASETS[$dataset_idx]}
RUNNING_SET=${RUNNING_SETS[$runset_idx]}
FOLD=${FOLDS[$fold_idx]}
LEARNING_RATE=${LEARNING_RATES[$lr_idx]}

# --- Fixed Model Configs ---
# MOE_CONFIG="4exp_top2"
# MOE_NOISE_STD=0.1
# LOAD_BALANCE_WEIGHT=0.01
# EPOCHS=200
# MAX_PATIENCE=20

EPOCHS=500
MAX_PATIENCE=30
# ENCODER_DROPOUT=0.15
# CROSS_ATTENTION_DROPOUT=0.15
# EXPERT_DROPOUT=0.15

ROOT_DIR="sweep_array_logs"
mkdir -p "$ROOT_DIR"

echo "======================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset:       $DATASET"
echo "Running set:   $RUNNING_SET"
echo "Fold:          $FOLD"
echo "======================================"

# --- Hyperparameter Loops ---
# for LEARNING_RATE in 5e-4 1e-3; do
for BATCH_SIZE in 16 32 64 128 256; do 
    
    TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
    LOG_DIR="${ROOT_DIR}/${DATASET}/${RUNNING_SET}/fold${FOLD}"
    mkdir -p "${LOG_DIR}"
    
    LOG_PREFIX="${LOG_DIR}/lr${LEARNING_RATE}_b${BATCH_SIZE}_jid${SLURM_ARRAY_JOB_ID}_tid${SLURM_ARRAY_TASK_ID}"
    
    echo "Running: LR=$LEARNING_RATE, Batch=$BATCH_SIZE"

    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
        python scripts/ab_testing_extra.py \
            --dataset "$DATASET" \
            --running_set "$RUNNING_SET" \
            --fold "$FOLD" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --max_patience "$MAX_PATIENCE" \
            --learning_rate "$LEARNING_RATE" \
            --cuda 0 \
            >> "${LOG_PREFIX}.out" \
            2>> "${LOG_PREFIX}.err"
done
# done


# --expert_config "$MOE_CONFIG" \
# --moe_noise_std "$MOE_NOISE_STD" \
# --load_balance_weight "$LOAD_BALANCE_WEIGHT" \
# --encoder_dropout "$ENCODER_DROPOUT" \
# --cross_attention_dropout "$CROSS_ATTENTION_DROPOUT" \
# --expert_dropout "$EXPERT_DROPOUT" \