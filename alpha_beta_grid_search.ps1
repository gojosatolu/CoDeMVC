# Pure ASCII script to avoid encoding issues in PowerShell 5.1
$currentScriptDir = Get-Location
$scriptPath = Join-Path $currentScriptDir "train.py"
$resultsDir = "alpha_beta_priority_results"

Write-Host "--- Grid Search Initialization ---"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
    Write-Host "Created results directory: $resultsDir"
}

# Values for Alpha and Beta from 0.0 to 1.0
$alphas = @(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
$betas  = @(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

$datasetBaselines = [ordered]@{
    "ORL"          = @{ ACC = 0.4250; NMI = 0.6577; PUR = 0.4625 }
    "EYaleB"       = @{ ACC = 0.2828; NMI = 0.2212; PUR = 0.3000 }
    "Yale"         = @{ ACC = 0.4970; NMI = 0.5565; PUR = 0.5030 }
    "Reuters-1200" = @{ ACC = 0.5683; NMI = 0.3215; PUR = 0.5683 }
    "COIL20"       = @{ ACC = 0.6847; NMI = 0.7573; PUR = 0.6951 }
    "OutdoorScene" = @{ ACC = 0.7154; NMI = 0.6019; PUR = 0.7154 }
    "Animal"       = @{ ACC = 0.1993; NMI = 0.1774; PUR = 0.2308 }
    "ALOI"         = @{ ACC = 0.8523; NMI = 0.9227; PUR = 0.8598 }
}

function Run-Experiment {
    param($dataset, $alpha, $beta, $ablationType, $logFile)

    $a_str = $alpha.ToString("F1")
    $b_str = $beta.ToString("F1")
    
    $cmd = "py -u `"$scriptPath`" --dataset $dataset --alpha $a_str --beta $b_str --causal_weight 0"
    if ($ablationType -eq "L2") { 
        $cmd += " --no_counterfactual --no_crm" 
    }
    
    # Run command and redirect
    Write-Host "  Executing: $dataset (a=$a_str, b=$b_str, type=$ablationType)" -ForegroundColor Gray
    
    $cmdFull = "$cmd > `"$logFile`" 2>&1"
    cmd /c $cmdFull

    if (Test-Path $logFile) {
        $content = Get-Content $logFile
        $lastLine = $content | Select-Object -Last 1
        if ($lastLine -match "The best clustering performace: ACC = ([\d\.]+) NMI = ([\d\.]+) PUR=([\d\.]+)") {
            return @{
                ACC = [double]$matches[1]
                NMI = [double]$matches[2]
                PUR = [double]$matches[3]
            }
        }
    }
    return $null
}

$csvPath = Join-Path $resultsDir "alpha_beta_monotony_summary.csv"
"Dataset,Alpha,Beta,Ablation,ACC,NMI,PUR,ImprovementType" | Out-File -FilePath $csvPath -Encoding ascii

foreach ($dsName in $datasetBaselines.Keys) {
    $base = $datasetBaselines[$dsName]
    Write-Host "`n>>> Processing Dataset: $dsName (Base ACC: $($base.ACC))" -ForegroundColor Cyan

    foreach ($alpha in $alphas) {
        foreach ($beta in $betas) {
            $comb = "a${alpha}_b${beta}"
            
            # --- 1. Run L2 ---
            $l2Log = Join-Path $resultsDir "${dsName}_L2_${comb}.log"
            $l2 = Run-Experiment -dataset $dsName -alpha $alpha -beta $beta -ablationType "L2" -logFile $l2Log

            if ($null -eq $l2) { 
                Write-Host "      [X] L2 Failure (parsing error)" -ForegroundColor DarkRed
                continue 
            }

            if ($l2.ACC -le $base.ACC) {
                # Write-Host "      [-] Skip: L2 ACC ($($l2.ACC)) <= Base" -ForegroundColor DarkGray
                continue
            }

            # --- 2. Run L3 ---
            $l3Log = Join-Path $resultsDir "${dsName}_L3_${comb}.log"
            $l3 = Run-Experiment -dataset $dsName -alpha $alpha -beta $beta -ablationType "L3" -logFile $l3Log

            if ($null -eq $l3) { continue }

            if ($l3.ACC -gt $l2.ACC) {
                $allInc = ($l3.NMI -gt $l2.NMI -and $l2.NMI -gt $base.NMI) -and ($l3.PUR -gt $l2.PUR -and $l2.PUR -gt $base.PUR)
                
                $type = "Only_ACC_Increase"
                $color = "Yellow"
                if ($allInc) {
                    $type = "All_Metrics_Increase"
                    $color = "Green"
                }
                
                Write-Host "      [OK] SUCCESS: ACC ($($base.ACC) -> $($l2.ACC) -> $($l3.ACC)) [$type]" -ForegroundColor $color
                
                "$dsName,$alpha,$beta,L2_Disentangle,$($l2.ACC),$($l2.NMI),$($l2.PUR),$type" | Out-File -FilePath $csvPath -Append -Encoding ascii
                "$dsName,$alpha,$beta,L3_FullModel,$($l3.ACC),$($l3.NMI),$($l3.PUR),$type" | Out-File -FilePath $csvPath -Append -Encoding ascii
            } else {
                # Write-Host "      [-] Reject: L3 ACC ($($l3.ACC)) <= L2" -ForegroundColor DarkGray
            }
        }
    }
}
Write-Host "`nGrid search complete. Results saved in: $csvPath" -ForegroundColor Magenta
