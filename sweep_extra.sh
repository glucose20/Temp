    #!/bin/bash
    #SBATCH --job-name=e_mz_wm
    #SBATCH --nodes=1
    #SBATCH --partition=gpu
    #SBATCH --gpus=l40s:1
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
    LOG_DIR="sweep_extra_log"
    mkdir -p "$LOG_DIR"


    # ----------------Best configs for each fold---------------------
    # Dataset,Running_Set,MoE_Config,MoE_Std,LBW,LR
    # BEST_CONFIG=(davis novel-drug 4exp_top2 0.08 0.0 0.0005)
    # BEST_CONFIG=(davis novel-pair 6exp_top2 0.1 0.01 0.0005)
    # BEST_CONFIG=(davis novel-prot 4exp_top1 0.05 0.02 0.0005)
    # BEST_CONFIG=(davis warm 4exp_top2 0.15 0.03 5e-4)
    # BEST_CONFIG=(kiba novel-drug 8exp_top2 0.05 0.03 0.0005)
    # BEST_CONFIG=(kiba novel-pair 6exp_top2 0.08 0.0 0.0005)
    # BEST_CONFIG=(kiba novel-prot 4exp_top1 0.08 0.02 0.0005)
    # BEST_CONFIG=(kiba warm 6exp_top2 0.05 0.03 0.0005)
    # BEST_CONFIG=(metz novel-drug 4exp_top2 0.15 0.02 0.0005)
    # BEST_CONFIG=(metz novel-pair 4exp_top1 0.15 0.03 0.0005)
    # BEST_CONFIG=(metz novel-prot 4exp_top2 0.1 0.02 0.0005)
    # BEST_CONFIG=(metz warm 6exp_top2 0.1 0.03 0.0005)


    # DATASET=${BEST_CONFIG[0]}
    # RUNNING_SET=${BEST_CONFIG[1]}
    # MOE_CONFIG=${BEST_CONFIG[2]}
    # MOE_NOISE_STD=${BEST_CONFIG[3]}
    # LOAD_BALANCE_WEIGHT=${BEST_CONFIG[4]}
    # LEARNING_RATE=${BEST_CONFIG[5]} 

    #----------------Sweep parameters------------------------
    FOLD=4

    # ---------------Fixed parameters---------------------

    # Refer from MixingDTA 
    EPOCHS=500
    BATCH_SIZE=64
    MAX_PATIENCE=30


    ENCODER_DROPOUT=0.15
    CROSS_ATTENTION_DROPOUT=0.15
    EXPERT_DROPOUT=0.15


    JID="${SLURM_JOB_ID:-local$$}"

    echo "======================================"
    echo "Dataset:     $DATASET"
    echo "Running set: $RUNNING_SET"
    echo "Timestamp:   $TIMESTAMP"
    echo "======================================"


    # for loop in range 0-4 to run 5 folds
    for fold_i in {0..4}; do
        TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
        LOG_FILE="${DATASET}_${RUNNING_SET}_b${BATCH_SIZE}_fold${fold_i}_moe${MOE_CONFIG}_lr${LEARNING_RATE}_moestd${MOE_NOISE_STD}_lbw${LOAD_BALANCE_WEIGHT}_dropoutE${ENCODER_DROPOUT}_C${CROSS_ATTENTION_DROPOUT}_X${EXPERT_DROPOUT}"
        LOG_PREFIX="${LOG_DIR}/id${JID}_${TIMESTAMP}_${LOG_FILE}"
        echo "Starting fold $fold_i"
        srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
            python scripts/ab_testing_extra.py \
                --dataset "$DATASET" \
                --running_set "$RUNNING_SET" \
                --fold "$fold_i" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --max_patience "$MAX_PATIENCE" \
                --learning_rate "$LEARNING_RATE" \
                --expert_config "$MOE_CONFIG" \
                --moe_noise_std "$MOE_NOISE_STD" \
                --load_balance_weight "$LOAD_BALANCE_WEIGHT" \
                --encoder_dropout "$ENCODER_DROPOUT" \
                --cross_attention_dropout "$CROSS_ATTENTION_DROPOUT" \
                --expert_dropout "$EXPERT_DROPOUT" \
                --cuda 0 \
                >> "${LOG_PREFIX}.out" \
                2>> "${LOG_PREFIX}.err"
        echo "Completed fold $fold_i"
    done


    # srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    #      python scripts/ab_testing_colab.py \
    #         --dataset "$DATASET" \
    #         --running_set "$RUNNING_SET" \
    #         --fold "$FOLD" \
    #         --epochs "$EPOCHS" \
    #         --batch_size "$BATCH_SIZE" \
    #         --max_patience "$MAX_PATIENCE" \
    #         --learning_rate "$LEARNING_RATE" \
    #         --expert_config "$MOE_CONFIG" \
    #         --moe_noise_std "$MOE_NOISE_STD" \
    #         --load_balance_weight "$LOAD_BALANCE_WEIGHT" \
    #         --cuda 0 \
    #         > "${LOG_PREFIX}.out" \
    #         2> "${LOG_PREFIX}.err"


    # srun --ntasks=4 \
    #      --exclusive \
    #      --gres=gpu:1 \
    #      --output="${LOG_PREFIX}_lbw${LOAD_BALANCE_WEIGHTS[$SLURM_PROCID]}.out"  \
    #      --error="${LOG_PREFIX}_lbw${LOAD_BALANCE_WEIGHTS[$SLURM_PROCID]}.err" \
    #      python scripts/ab_testing_colab.py \
    #         --dataset "$DATASET" \
    #         --running_set "$RUNNING_SET" \
    #         --fold "$FOLD" \
    #         --epochs "$EPOCHS" \
    #         --batch_size "$BATCH_SIZE" \
    #         --learning_rate "$LEARNING_RATE" \
    #         --expert_config "$MOE_CONFIG" \
    #         --moe_noise_std "$MOE_NOISE_STD" \
    #         --load_balance_weight "${LOAD_BALANCE_WEIGHTS[$SLURM_PROCID]}" \
    #         --cuda \$SLURM_PROCID \

