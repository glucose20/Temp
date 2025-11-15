#!/bin/bash
################################################################################
# LLMDTA Single Dataset Runner
# Run all experiments for ONE specific dataset
# Usage: bash scripts/run_single_dataset.sh <dataset_name>
# Example: bash scripts/run_single_dataset.sh davis
################################################################################

set -e

# Check if dataset name is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide dataset name"
    echo "Usage: bash scripts/run_single_dataset.sh <dataset_name>"
    echo "Available datasets: davis, kiba, metz"
    exit 1
fi

DATASET=$1
EPOCHS=200
BATCH_SIZE=16
CUDA_DEVICE="0"
SETTINGS=("warm" "novel-drug" "novel-prot" "novel-pair")
FOLDS=(0 1 2 3 4)

echo "============================================================"
echo "LLMDTA - Single Dataset Runner"
echo "============================================================"
echo ""
echo "Dataset:     $DATASET"
echo "Settings:    ${SETTINGS[@]}"
echo "Folds:       ${FOLDS[@]}"
echo "Epochs:      $EPOCHS"
echo "Batch Size:  $BATCH_SIZE"
echo "CUDA Device: $CUDA_DEVICE"
echo ""
echo "Total Runs:  $((${#SETTINGS[@]} * ${#FOLDS[@]}))"
echo "============================================================"
echo ""

# Create directories
mkdir -p ./log
mkdir -p ./savemodel

# Run all settings and folds for this dataset
for setting in "${SETTINGS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running $DATASET - $setting"
    echo "========================================"
    
    for fold in "${FOLDS[@]}"; do
        echo ""
        echo ">>> Fold $fold"
        
        python code/train.py \
            --fold $fold \
            --cuda "$CUDA_DEVICE" \
            --dataset $DATASET \
            --running_set $setting \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE
        
        echo "✓ Completed: $DATASET - $setting - fold $fold"
    done
    
    # Aggregate results for this setting
    echo ""
    echo "Aggregating results for $DATASET - $setting..."
    python code/aggregate_results.py --dataset $DATASET --running_set $setting || true
    echo ""
done

echo ""
echo "============================================================"
echo "All experiments for $DATASET completed!"
echo "============================================================"
