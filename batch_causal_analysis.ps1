# batch_causal_analysis.ps1
# 批处理因果分析 (深度对齐版)

$dataset_configs = @(
    @("ProteinFold", 12),
    @("BDGP", 2),
    @("Prokaryotic", 3),
    @("Caltech-5V", 5),
    @("NoisyBDGP", 2),
    @("NoisyHW", 6),
    @("NoisyMNIST", 2),
    @("NUS-WIDE", 5),
    @("OutdoorScene", 4),
    @("Scene-15", 3)
)

$max_order = 2

foreach ($config in $dataset_configs) {
    $ds = $config[0]
    $view = $config[1]
    
    Write-Host "`n[*] >>> Starting Analysis: $ds ($view Views) <<<" -ForegroundColor Cyan
    
    # 核心修正：因果注意力参数存储在 XXX.pth 中 (Network 的主权重)
    $model_path = "./models/${ds}.pth"

    if (Test-Path $model_path) {
        Write-Host "[*] Found Feature-Causal Weights: $model_path" -ForegroundColor Gray
        # 调用分析引擎，确保加载的是包含 FeatureCausalAttention 的 Network 权重
        & py ./analyze_causal_graph.py --dataset $ds --view $view --max_order $max_order --model_path $model_path
        Write-Host "[+] End of Session for $ds" -ForegroundColor Green
    } else {
        Write-Warning "[-] Skip ${ds}: Metrics not found at $model_path. Ensure you have retrained the model with the new architecture."
    }

    Write-Host "---------------------------------------------------------"
}