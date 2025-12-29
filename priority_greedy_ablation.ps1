# Priority Greedy Ablation Study for CoDe-MVC
# Targeted at High-Confidence Datasets with specific weight ranges

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Priority Greedy Ablation Study" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$resultsDir = "priority_results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

# --- TIER 1 & 2: High Confidence & Robustness (Standard Weights) ---
$tier12 = @(
    "STL10"
    )
$weights12 = @(0.1, 0.2, 0.3, 0.5, 0.8, 1.0)

# --- TIER 3: High-Dimensional Challenge (Lower Weights) ---
$tier3 = @( 
    "Caltech101-20")
$weights3 = @(0.05, 0.1, 0.2, 0.3, 0.5)

function Run-Ablation($dataset, $weights) {
    Write-Host "`n>>> Processing Dataset: $dataset" -ForegroundColor Yellow
    
    # 1. Baseline (Level 0)
    Write-Host "  [L0] Running Baseline (No CRM, No Causal)..." -ForegroundColor Gray
    $log = "$resultsDir/${dataset}_L0_Baseline.log"
    if (-not (Test-Path $log)) {
        py train.py --dataset $dataset --causal_weight 0 --no_counterfactual --no_crm > $log 2>&1
    }

    # 2. +CRM (Level 1)
    Write-Host "  [L1] Running +CRM (No Causal)..." -ForegroundColor Gray
    $log = "$resultsDir/${dataset}_L1_CRM.log"
    if (-not (Test-Path $log)) {
        py train.py --dataset $dataset --causal_weight 0 --no_counterfactual > $log 2>&1
    }

    foreach ($w in $weights) {
        Write-Host "  >>> Testing causal_weight = $w" -ForegroundColor Cyan
        
        # 3. +Disentangle (Level 2)
        Write-Host "    [L2] Running +Disentangle..." -ForegroundColor Gray
        $log = "$resultsDir/${dataset}_L2_Disentangle_cw${w}.log"
        py train.py --dataset $dataset --causal_weight $w --no_counterfactual > $log 2>&1
        
        # 4. Full Model (Level 3 - with Causal Warm-up)
        Write-Host "    [L3] Running Full Model (Warm-up enabled)..." -ForegroundColor Gray
        $log = "$resultsDir/${dataset}_L3_Full_cw${w}.log"
        py train.py --dataset $dataset --causal_weight $w > $log 2>&1
    }
}

# Run Tier 1 & 2
foreach ($d in $tier12) {
    Run-Ablation $d $weights12
}

# Run Tier 3
foreach ($d in $tier3) {
    Run-Ablation $d $weights3
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Priority Ablation Study Completed!" -ForegroundColor Cyan
Write-Host "Check results in: $resultsDir/" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
