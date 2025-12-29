# Grid Search and Ablation Study for Multi-View Clustering
# PowerShell Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Grid Search and Ablation Study" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create results directory
$resultsDir = "grid_search_results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

# Dataset list - All configured datasets
$datasets = @(
    "Prokaryotic", "Synthetic3d", "Caltech-2V", "Caltech-3V", "Caltech-4V", "Caltech-5V",
    "Cifar10", "Cifar100", "COIL20", "ALOI", "OutdoorScene", "Caltech101",
    "MSRC-v1", "NUS-WIDE", "Caltech101-20", "Animal", "NoisyBDGP", "NoisyHW", "STL10", "NoisyMNIST"
)

# Causal weight values to test
$weights = @(0.1, 0.2, 0.3, 0.5, 0.8, 1.0)

# Loop through all datasets
foreach ($dataset in $datasets) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "Processing Dataset: $dataset" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    
    # Ablation 1: Baseline (SCMVC only, no CRM, no causal)
    Write-Host "[$dataset] Running Baseline (No CRM, No Causal)..." -ForegroundColor Green
    $logFile = "$resultsDir/${dataset}_baseline.log"
    py train.py --dataset $dataset --causal_weight 0 --no_counterfactual --no_crm > $logFile 2>&1
    Write-Host "[$dataset] Baseline completed." -ForegroundColor Green
    
    # Ablation 2: +CRM (with CRM, but no causal loss)
    Write-Host "[$dataset] Running +CRM (CRM only, No Causal)..." -ForegroundColor Green
    $logFile = "$resultsDir/${dataset}_crm.log"
    py train.py --dataset $dataset --causal_weight 0 --no_counterfactual > $logFile 2>&1
    Write-Host "[$dataset] +CRM completed." -ForegroundColor Green
    
    # Ablation 3 & 4: Grid Search over causal_weight
    foreach ($weight in $weights) {
        Write-Host "[$dataset] Testing causal_weight=$weight" -ForegroundColor Cyan
        
        # +Disentangle (CRM + Causal Disentanglement, no counterfactual)
        Write-Host "  Running +Disentangle..." -ForegroundColor Gray
        $logFile = "$resultsDir/${dataset}_disentangle_cw${weight}.log"
        py train.py --dataset $dataset --causal_weight $weight --no_counterfactual > $logFile 2>&1
        Write-Host "  +Disentangle completed." -ForegroundColor Gray
        
        # Full Model (CRM + Disentanglement + Counterfactual)
        Write-Host "  Running Full Model..." -ForegroundColor Gray
        $logFile = "$resultsDir/${dataset}_full_cw${weight}.log"
        py train.py --dataset $dataset --causal_weight $weight > $logFile 2>&1
        Write-Host "  Full Model completed." -ForegroundColor Gray
    }
    
    Write-Host "[$dataset] All experiments completed" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All Grid Search and Ablation Completed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved in: $resultsDir/" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run: python parse_grid_search_results.py" -ForegroundColor White
Write-Host "  2. Check: grid_search_summary.csv" -ForegroundColor White
Write-Host ""
