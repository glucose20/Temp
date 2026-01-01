#!/bin/bash
#SBATCH --job-name=ab_multi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:5             # Request 5 total GPUs for the job
#SBATCH --cpus-per-task=8        # Matches your 8 cpus per task logic
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --qos=batch-short
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=s226181148@deakin.edu.au

# --- Environment setup ---
module purge
module load Anaconda3
source activate esm_thuy   # Combined source/activate for brevity
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=657df3c06ebe7d9b611a9e81fa9d72eb0e9c76b9

# --- Define Variables ---
LOG_DIR="log_abs"
mkdir -p "$LOG_DIR"

DATASETS=("davis" "kiba" "metz")
RUNNING_SETS=("novel-pair" "novel-drug" "novel-prot" "warm")

DATASET_INDEX=2 # 0 - > 2
RUNNING_SET_INDEX=3 # 0 -> 3

DATASET=${DATASETS[$DATASET_INDEX]}
RUNNING_SET=${RUNNING_SETS[$RUNNING_SET_INDEX]}
EPOCHS=200


EXECUTABLE="python scripts/ab_testing_colab.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="ab_testing_${DATASET}_${RUNNING_SET}"

# --- Execution ---
# We use --gres=gpu:1 here to give each of the 5 tasks 1 GPU
srun --ntasks=5 \
     --exclusive \
     --gres=gpu:1 \
     --output="$LOG_DIR/id%j_${LOG_FILE}_fold%t.out" \
     --error="$LOG_DIR/id%j${LOG_FILE}_fold%t.err" \
     bash -c "$EXECUTABLE \
        --dataset $DATASET \
        --running_set $RUNNING_SET \
        --fold \$SLURM_PROCID \
        --epochs $EPOCHS" \
        --cuda \$SLURM_PROCID


# srun --ntasks=5 \
#      --exclusive \
#      --gres=gpu:1 \
#      --output="$LOG_DIR/task_%j_%t.out" \
#      --error="$LOG_DIR/task_%j_%t.err" \
#      bash -c "python3 scripts/testslurm.py"