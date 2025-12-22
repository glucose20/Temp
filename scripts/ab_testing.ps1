# A/B Testing Script: Compare Baseline vs MoE
# Usage: .\scripts\ab_testing.ps1 -dataset davis -running_set novel-pair -fold 0

param(
    [string]$dataset = "davis",
    [string]$running_set = "novel-pair",
    [int]$fold = 0,
    [int]$epochs = 100,
    [string]$cuda = "0"
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "A/B Testing: Baseline vs MoE" -ForegroundColor Cyan
Write-Host "Dataset: $dataset-$running_set, Fold: $fold, Epochs: $epochs" -ForegroundColor Cyan
Write-Host "Timestamp: $timestamp" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# Create results directory
$resultsDir = "./ab_results/$timestamp"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

# ============================================================
# Test A: Baseline (No MoE - single expert)
# ============================================================
Write-Host "`n[TEST A] Running Baseline (num_experts=1, top_k=1)..." -ForegroundColor Yellow

$baselineLog = "$resultsDir/baseline_fold${fold}.log"
python code/train.py `
    --fold $fold `
    --dataset $dataset `
    --running_set $running_set `
    --epochs $epochs `
    --cuda $cuda `
    --num_experts 1 `
    --top_k 1 `
    --load_balance_weight 0 `
    --wandb_project "LLMDTA-AB-Test" `
    --no_wandb `
    2>&1 | Tee-Object -FilePath $baselineLog

Write-Host "[TEST A] Baseline completed!" -ForegroundColor Green

# ============================================================
# Test B: MoE with 4 experts, top-2
# ============================================================
Write-Host "`n[TEST B] Running MoE (num_experts=4, top_k=2)..." -ForegroundColor Yellow

$moeLog = "$resultsDir/moe_4exp_top2_fold${fold}.log"
python code/train.py `
    --fold $fold `
    --dataset $dataset `
    --running_set $running_set `
    --epochs $epochs `
    --cuda $cuda `
    --num_experts 4 `
    --top_k 2 `
    --load_balance_weight 0.01 `
    --moe_noise_std 0.1 `
    --wandb_project "LLMDTA-AB-Test" `
    --no_wandb `
    2>&1 | Tee-Object -FilePath $moeLog

Write-Host "[TEST B] MoE (4 experts, top-2) completed!" -ForegroundColor Green

# ============================================================
# Test C: MoE with 4 experts, top-1 (sparse)
# ============================================================
Write-Host "`n[TEST C] Running Sparse MoE (num_experts=4, top_k=1)..." -ForegroundColor Yellow

$sparseMoeLog = "$resultsDir/moe_4exp_top1_fold${fold}.log"
python code/train.py `
    --fold $fold `
    --dataset $dataset `
    --running_set $running_set `
    --epochs $epochs `
    --cuda $cuda `
    --num_experts 4 `
    --top_k 1 `
    --load_balance_weight 0.02 `
    --moe_noise_std 0.15 `
    --wandb_project "LLMDTA-AB-Test" `
    --no_wandb `
    2>&1 | Tee-Object -FilePath $sparseMoeLog

Write-Host "[TEST C] Sparse MoE (4 experts, top-1) completed!" -ForegroundColor Green

# ============================================================
# Test D: MoE with 8 experts, top-2
# ============================================================
Write-Host "`n[TEST D] Running MoE (num_experts=8, top_k=2)..." -ForegroundColor Yellow

$moe8Log = "$resultsDir/moe_8exp_top2_fold${fold}.log"
python code/train.py `
    --fold $fold `
    --dataset $dataset `
    --running_set $running_set `
    --epochs $epochs `
    --cuda $cuda `
    --num_experts 8 `
    --top_k 2 `
    --load_balance_weight 0.01 `
    --moe_noise_std 0.1 `
    --wandb_project "LLMDTA-AB-Test" `
    --no_wandb `
    2>&1 | Tee-Object -FilePath $moe8Log

Write-Host "[TEST D] MoE (8 experts, top-2) completed!" -ForegroundColor Green

# ============================================================
# Summary
# ============================================================
Write-Host "`n" 
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "A/B Testing Complete! Results saved to: $resultsDir" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

Write-Host "`nExtract test results:" -ForegroundColor Yellow
Write-Host "Baseline:     " -NoNewline; Select-String -Path $baselineLog -Pattern "Test at fold" | Select-Object -Last 1
Write-Host "MoE 4exp-top2:" -NoNewline; Select-String -Path $moeLog -Pattern "Test at fold" | Select-Object -Last 1
Write-Host "MoE 4exp-top1:" -NoNewline; Select-String -Path $sparseMoeLog -Pattern "Test at fold" | Select-Object -Last 1
Write-Host "MoE 8exp-top2:" -NoNewline; Select-String -Path $moe8Log -Pattern "Test at fold" | Select-Object -Last 1

Write-Host "`nTo compare results in detail, check log files in: $resultsDir" -ForegroundColor Gray
