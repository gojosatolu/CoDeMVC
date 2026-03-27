# batch_train_causal.ps1
# Sequential training for key causal datasets: ProteinFold, BDGP, Prokaryotic

$datasets = @("Caltech-5V","NoisyBDGP","NoisyHW","NoisyMNIST","NUS-WIDE","OutdoorScene","Scene-15")

foreach ($ds in $datasets) {
    Write-Host "[*] starting Experiment: $ds" -ForegroundColor Green
    Write-Host "[*] Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    
    # Run training and save checkpoints
    py train.py --dataset $ds
    
    Write-Host "[+] Finished Experiment: $ds" -ForegroundColor Cyan
    Write-Host "-------------------------------------------"
}

Write-Host "[***] BATCH TRAINING COMPLETE!" -ForegroundColor Yellow
Write-Host "All checkpoints are saved in ./models/ for causal analysis." -ForegroundColor Gray
