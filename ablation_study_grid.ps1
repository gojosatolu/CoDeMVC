# ===== 环境初始化 =====
$script_base = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }

$log_root = Join-Path $script_base "ablation_results"
$train_script = Join-Path $script_base "train.py"

# ===== 数据集配置（修复版）=====
$dataset_configs = @(
    @("EYaleB", 3),
    @("Yale", 3),
    @("ORL", 3),
    @("BDGP", 2),
    @("OutdoorScene", 4),
    @("Prokaryotic", 3),
    @("Caltech-5V", 5),
    @("NoisyBDGP", 2),
    @("NoisyHW", 6),
    @("NoisyMNIST", 2),
    @("NUS-WIDE", 5),
    @("Scene-15", 3),
    @("ProteinFold", 12),
    @("WebKB", 2)   
)

# ===== 主循环 =====
foreach ($config in $dataset_configs) {

    $ds = $config[0]
    $ds_log_dir = Join-Path $log_root $ds

    if (!(Test-Path $ds_log_dir)) {
        New-Item -ItemType Directory -Path $ds_log_dir -Force | Out-Null
    }

    for ($a_idx = 0; $a_idx -le 10; $a_idx++) {
        $alpha = $a_idx / 10.0

        for ($b_idx = 0; $b_idx -le 10; $b_idx++) {
            $beta = $b_idx / 10.0

            $log_path = Join-Path $ds_log_dir "Log_${ds}_A${alpha}_B${beta}.txt"

            # ===== 稳健版断点续传逻辑 =====
            $is_completed = $false
            if (Test-Path $log_path) {

                $content = Get-Content $log_path -Raw
                if ($content -match "The best clustering performace") {
                    $is_completed = $true
                }
            }

            if ($is_completed) {
                Write-Host "[SKIP] Already completed: $ds | A=$alpha | B=$beta" -ForegroundColor Gray
                continue
            }

            Write-Host ">>> Running: $ds | A=$alpha | B=$beta" -ForegroundColor Green

            & py $train_script --dataset $ds --alpha $alpha --beta $beta |
                Tee-Object -FilePath $log_path


            if ($LASTEXITCODE -ne 0) {
                Write-Error "Crashed at A=$alpha, B=$beta"
                exit
            }
        }
    }
}