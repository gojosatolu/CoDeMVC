# No-CRM Causal Ablation Study
# Purpose: Verify the independence and power of Causal/Counterfactual modules without CRM

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting No-CRM Causal Grid Search" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$resultsDir = "nocrm_causal_results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

# The datasets specified by the user
$datasets = @("Caltech101-20","STL10")

# Optimized weights for exploration
$weights = @(0.1, 0.2, 0.3, 0.5, 0.8, 1.0)

foreach ($dataset in $datasets) {
    Write-Host "`n>>> Dataset: $dataset (NO-CRM MODE)" -ForegroundColor Yellow

    foreach ($w in $weights) {
        Write-Host "  >>> Testing causal_weight = $w" -ForegroundColor Cyan
        
        # 1. Disentangle Only (No CRM, No CF)
        Write-Host "    Running Disentangle Only (-CRM)..." -ForegroundColor Gray
        $log = "$resultsDir/${dataset}_DisOnly_cw${w}.log"
        py train.py --dataset $dataset --causal_weight $w --no_counterfactual --no_crm > $log 2>&1
        
        # 2. Full Causal (No CRM, +CF)
        Write-Host "    Running Full Causal (+CF, -CRM)..." -ForegroundColor Gray
        $log = "$resultsDir/${dataset}_FullCausal_cw${w}.log"
        py train.py --dataset $dataset --causal_weight $w --no_crm > $log 2>&1
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "No-CRM Causal Analysis Completed!" -ForegroundColor Cyan
Write-Host "Check results in: $resultsDir/" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
