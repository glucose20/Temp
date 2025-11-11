# PowerShell script to train all folds in parallel
# Usage: .\scripts\train_all_folds.ps1

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Training LLMDTA - All Folds (Parallel)" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

# Configuration
$DATASET = "davis"
$RUNNING_SET = "novel-pair"
$NUM_FOLDS = 5

# GPU devices (adjust based on your hardware)
# If you have multiple GPUs, distribute folds across them
# Example: GPU 0, 1, 2, 3
$GPU_DEVICES = @("0", "0", "0", "0", "0")  # Change to @("0", "1", "2", "3", "0") if you have 4 GPUs

Write-Host "`nConfiguration:" -ForegroundColor Green
Write-Host "  Dataset:     $DATASET"
Write-Host "  Running Set: $RUNNING_SET"
Write-Host "  Num Folds:   $NUM_FOLDS"
Write-Host "  GPU Config:  $($GPU_DEVICES -join ', ')"
Write-Host ""

# Create log directory
New-Item -ItemType Directory -Force -Path "./log" | Out-Null
New-Item -ItemType Directory -Force -Path "./savemodel" | Out-Null

# Array to store job objects
$jobs = @()

# Start training for each fold
for ($fold = 0; $fold -lt $NUM_FOLDS; $fold++) {
    $gpu = $GPU_DEVICES[$fold]
    
    Write-Host "Starting Fold $fold on GPU $gpu..." -ForegroundColor Cyan
    
    # Start training as background job
    $job = Start-Job -ScriptBlock {
        param($fold, $gpu, $workDir)
        Set-Location $workDir
        python code/train.py --fold $fold --cuda $gpu 2>&1
    } -ArgumentList $fold, $gpu, $PWD
    
    $jobs += $job
    
    # Small delay to avoid race conditions
    Start-Sleep -Seconds 2
}

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "All $NUM_FOLDS training jobs started!" -ForegroundColor Green
Write-Host "Waiting for completion..." -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Monitor jobs
$completedCount = 0
while ($completedCount -lt $NUM_FOLDS) {
    $completedCount = 0
    for ($i = 0; $i -lt $NUM_FOLDS; $i++) {
        $state = $jobs[$i].State
        if ($state -eq "Completed" -or $state -eq "Failed") {
            $completedCount++
        }
    }
    
    Write-Host "`rProgress: $completedCount / $NUM_FOLDS folds completed" -NoNewline
    Start-Sleep -Seconds 5
}

Write-Host "`n"
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "All training jobs finished!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Display job results
for ($i = 0; $i -lt $NUM_FOLDS; $i++) {
    $job = $jobs[$i]
    $output = Receive-Job -Job $job
    
    Write-Host "`n--- Fold $i Output (Last 20 lines) ---" -ForegroundColor Cyan
    $output | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
    
    if ($job.State -eq "Failed") {
        Write-Host "[ERROR] Fold $i failed!" -ForegroundColor Red
    } else {
        Write-Host "[SUCCESS] Fold $i completed!" -ForegroundColor Green
    }
}

# Clean up jobs
$jobs | Remove-Job

# Aggregate results
Write-Host "`n"
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Aggregating results..." -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

python code/aggregate_results.py --dataset $DATASET --running_set $RUNNING_SET

Write-Host "`nTraining pipeline completed!" -ForegroundColor Green
