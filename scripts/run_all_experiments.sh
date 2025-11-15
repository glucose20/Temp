#!/bin/bash
################################################################################
# LLMDTA Full Experiment Runner
# Runs all 60 experiments: 3 datasets × 4 settings × 5 folds
# Total: 60 training runs
################################################################################

set -e  # Exit on error

# Configuration
EPOCHS=200
BATCH_SIZE=16
CUDA_DEVICE="0"  # Change to your GPU device
DATASETS=("davis" "kiba" "metz")
SETTINGS=("warm" "novel-drug" "novel-prot" "novel-pair")
FOLDS=(0 1 2 3 4)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "============================================================"
echo "LLMDTA - Full Experiment Runner"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Datasets:    ${DATASETS[@]}"
echo "  Settings:    ${SETTINGS[@]}"
echo "  Folds:       ${FOLDS[@]}"
echo "  Epochs:      $EPOCHS"
echo "  Batch Size:  $BATCH_SIZE"
echo "  CUDA Device: $CUDA_DEVICE"
echo ""
echo "Total Runs:  $((${#DATASETS[@]} * ${#SETTINGS[@]} * ${#FOLDS[@]}))"
echo "============================================================"
echo ""

# Create necessary directories
mkdir -p ./log
mkdir -p ./savemodel
mkdir -p ./results

# Counter for progress tracking
TOTAL_RUNS=$((${#DATASETS[@]} * ${#SETTINGS[@]} * ${#FOLDS[@]}))
CURRENT_RUN=0
START_TIME=$(date +%s)

# Log file for tracking all runs
MASTER_LOG="./results/experiment_master_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Master log file: $MASTER_LOG"
echo "Experiment started at: $(date)" > $MASTER_LOG
echo "============================================================" >> $MASTER_LOG
echo "" >> $MASTER_LOG

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local setting=$2
    local fold=$3
    local run_num=$4
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${YELLOW}Run [$CURRENT_RUN/$TOTAL_RUNS]: $dataset - $setting - fold $fold${NC}"
    echo -e "${CYAN}============================================================${NC}"
    
    # Log to master log
    echo "[$CURRENT_RUN/$TOTAL_RUNS] Starting: $dataset - $setting - fold $fold at $(date)" >> $MASTER_LOG
    
    # Run the training
    RUN_START=$(date +%s)
    
    if python code/train.py \
        --fold $fold \
        --cuda "$CUDA_DEVICE" \
        --dataset $dataset \
        --running_set $setting \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE; then
        
        RUN_END=$(date +%s)
        RUN_DURATION=$((RUN_END - RUN_START))
        
        echo -e "${GREEN}✓ Success: $dataset - $setting - fold $fold (${RUN_DURATION}s)${NC}"
        echo "[$CURRENT_RUN/$TOTAL_RUNS] SUCCESS: $dataset - $setting - fold $fold (Duration: ${RUN_DURATION}s)" >> $MASTER_LOG
    else
        RUN_END=$(date +%s)
        RUN_DURATION=$((RUN_END - RUN_START))
        
        echo -e "${RED}✗ Failed: $dataset - $setting - fold $fold${NC}"
        echo "[$CURRENT_RUN/$TOTAL_RUNS] FAILED: $dataset - $setting - fold $fold (Duration: ${RUN_DURATION}s)" >> $MASTER_LOG
    fi
    
    echo "" >> $MASTER_LOG
    echo ""
}

# Main loop - Run all experiments sequentially
for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
        for fold in "${FOLDS[@]}"; do
            run_experiment "$dataset" "$setting" "$fold" "$CURRENT_RUN"
            
            # Estimate remaining time
            ELAPSED=$(($(date +%s) - START_TIME))
            AVG_TIME=$((ELAPSED / CURRENT_RUN))
            REMAINING_RUNS=$((TOTAL_RUNS - CURRENT_RUN))
            EST_REMAINING=$((AVG_TIME * REMAINING_RUNS))
            
            echo -e "${CYAN}Progress: $CURRENT_RUN/$TOTAL_RUNS completed${NC}"
            echo -e "${CYAN}Estimated remaining time: $((EST_REMAINING / 3600))h $((EST_REMAINING % 3600 / 60))m${NC}"
            echo ""
        done
        
        # After completing all folds for a setting, aggregate results
        echo -e "${YELLOW}Aggregating results for $dataset - $setting...${NC}"
        python code/aggregate_results.py --dataset $dataset --running_set $setting || true
        echo ""
    done
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "============================================================"
echo "All experiments completed!"
echo "============================================================"
echo "Total time: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
echo "Results saved in: ./log/ and ./savemodel/"
echo "Master log: $MASTER_LOG"
echo "============================================================"

# Append final summary to master log
echo "============================================================" >> $MASTER_LOG
echo "Experiment completed at: $(date)" >> $MASTER_LOG
echo "Total duration: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s" >> $MASTER_LOG
echo "============================================================" >> $MASTER_LOG

# Generate final summary report
echo ""
echo "Generating final summary report..."
python code/generate_final_report.py || echo "Note: generate_final_report.py not found, skipping summary generation"

echo ""
echo "Done!"
