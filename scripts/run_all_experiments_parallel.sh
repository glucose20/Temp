#!/bin/bash
################################################################################
# LLMDTA Parallel Experiment Runner (Multi-GPU)
# Runs experiments in parallel across multiple GPUs
# Total: 60 runs (3 datasets × 4 settings × 5 folds)
################################################################################

set -e

# Configuration
EPOCHS=200
BATCH_SIZE=16

# GPU Configuration - Adjust based on your hardware
# Example: If you have 4 GPUs, you can run 4 experiments in parallel
NUM_GPUS=4
GPU_DEVICES=(0 1 2 3)  # Available GPU IDs

DATASETS=("davis" "kiba" "metz")
SETTINGS=("warm" "novel-drug" "novel-prot" "novel-pair")
FOLDS=(0 1 2 3 4)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "============================================================"
echo "LLMDTA - Parallel Experiment Runner (Multi-GPU)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Datasets:    ${DATASETS[@]}"
echo "  Settings:    ${SETTINGS[@]}"
echo "  Folds:       ${FOLDS[@]}"
echo "  Epochs:      $EPOCHS"
echo "  Batch Size:  $BATCH_SIZE"
echo "  GPUs:        ${GPU_DEVICES[@]}"
echo ""
echo "Total Runs:  $((${#DATASETS[@]} * ${#SETTINGS[@]} * ${#FOLDS[@]}))"
echo "Parallelism: $NUM_GPUS concurrent jobs"
echo "============================================================"
echo ""

# Create directories
mkdir -p ./log
mkdir -p ./savemodel
mkdir -p ./results

# Master log
MASTER_LOG="./results/parallel_experiment_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Experiment started at: $(date)" > $MASTER_LOG

# Function to get next available GPU
get_available_gpu() {
    local jobs_per_gpu=()
    
    # Count jobs per GPU
    for gpu in "${GPU_DEVICES[@]}"; do
        count=$(jobs -r | grep -c "GPU_$gpu" || true)
        jobs_per_gpu+=("$count:$gpu")
    done
    
    # Find GPU with minimum jobs
    min_gpu=$(printf '%s\n' "${jobs_per_gpu[@]}" | sort -n | head -1 | cut -d: -f2)
    echo "$min_gpu"
}

# Function to wait for available slot
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
        sleep 10
    done
}

# Function to run experiment in background
run_experiment_async() {
    local dataset=$1
    local setting=$2
    local fold=$3
    local gpu=$4
    
    local log_file="./results/run_${dataset}_${setting}_fold${fold}.log"
    
    echo "[$(date)] Starting: $dataset - $setting - fold $fold on GPU $gpu" | tee -a $MASTER_LOG
    
    (
        # This block runs in background
        export CUDA_VISIBLE_DEVICES=$gpu
        
        if python code/train.py \
            --fold $fold \
            --cuda "0" \
            --dataset $dataset \
            --running_set $setting \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE > $log_file 2>&1; then
            
            echo "[$(date)] SUCCESS: $dataset - $setting - fold $fold (GPU $gpu)" | tee -a $MASTER_LOG
        else
            echo "[$(date)] FAILED: $dataset - $setting - fold $fold (GPU $gpu)" | tee -a $MASTER_LOG
        fi
    ) &
    
    # Tag the job with GPU info for tracking
    local pid=$!
    echo "GPU_$gpu: PID $pid - $dataset $setting fold $fold"
}

# Main loop - Submit all experiments
RUN_COUNT=0
TOTAL_RUNS=$((${#DATASETS[@]} * ${#SETTINGS[@]} * ${#FOLDS[@]}))

for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
        for fold in "${FOLDS[@]}"; do
            # Wait for available GPU slot
            wait_for_slot
            
            # Get next available GPU
            gpu=$(get_available_gpu)
            
            # Submit job
            run_experiment_async "$dataset" "$setting" "$fold" "$gpu"
            
            RUN_COUNT=$((RUN_COUNT + 1))
            echo -e "${CYAN}Submitted: [$RUN_COUNT/$TOTAL_RUNS] $dataset - $setting - fold $fold on GPU $gpu${NC}"
            
            # Small delay to avoid race conditions
            sleep 2
        done
    done
done

echo ""
echo -e "${YELLOW}All jobs submitted! Waiting for completion...${NC}"
echo ""

# Wait for all background jobs to complete
wait

echo ""
echo "============================================================"
echo "All parallel experiments completed!"
echo "============================================================"

# Aggregate results for each dataset-setting combination
echo ""
echo "Aggregating results..."
for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
        echo "Aggregating: $dataset - $setting"
        python code/aggregate_results.py --dataset $dataset --running_set $setting || true
    done
done

echo ""
echo "Done! Check $MASTER_LOG for details."
echo "Results in: ./log/ and ./savemodel/"
