#!/bin/bash
# Bash script to train all folds in parallel
# Usage: bash scripts/train_all_folds.sh

echo "============================================================"
echo "Training LLMDTA - All Folds (Parallel)"
echo "============================================================"

# Configuration
DATASET="davis"
RUNNING_SET="novel-pair"
NUM_FOLDS=5

# GPU devices (adjust based on your hardware)
# If you have multiple GPUs, distribute folds across them
# Example: GPU 0, 1, 2, 3
GPU_DEVICES=(0 0 0 0 0)  # Change to (0 1 2 3 0) if you have 4 GPUs

echo ""
echo "Configuration:"
echo "  Dataset:     $DATASET"
echo "  Running Set: $RUNNING_SET"
echo "  Num Folds:   $NUM_FOLDS"
echo "  GPU Config:  ${GPU_DEVICES[*]}"
echo ""

# Create log directory
mkdir -p ./log
mkdir -p ./savemodel

# Array to store PIDs
declare -a PIDS

# Start training for each fold
for ((fold=0; fold<$NUM_FOLDS; fold++)); do
    gpu=${GPU_DEVICES[$fold]}
    
    echo "Starting Fold $fold on GPU $gpu..."
    
    # Start training in background
    python code/train.py --fold $fold --cuda $gpu > "./log/fold_${fold}_console.log" 2>&1 &
    PIDS[$fold]=$!
    
    # Small delay to avoid race conditions
    sleep 2
done

echo ""
echo "============================================================"
echo "All $NUM_FOLDS training jobs started!"
echo "PIDs: ${PIDS[*]}"
echo "Waiting for completion..."
echo "============================================================"
echo ""

# Wait for all jobs to complete
for ((fold=0; fold<$NUM_FOLDS; fold++)); do
    pid=${PIDS[$fold]}
    echo "Waiting for Fold $fold (PID: $pid)..."
    wait $pid
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[SUCCESS] Fold $fold completed!"
    else
        echo "[ERROR] Fold $fold failed with exit code $exit_code"
    fi
done

echo ""
echo "============================================================"
echo "All training jobs finished!"
echo "============================================================"
echo ""

# Display last 20 lines of each fold's output
for ((fold=0; fold<$NUM_FOLDS; fold++)); do
    echo ""
    echo "--- Fold $fold Output (Last 20 lines) ---"
    tail -n 20 "./log/fold_${fold}_console.log"
done

# Aggregate results
echo ""
echo "============================================================"
echo "Aggregating results..."
echo "============================================================"

python code/aggregate_results.py --dataset $DATASET --running_set $RUNNING_SET

echo ""
echo "Training pipeline completed!"
