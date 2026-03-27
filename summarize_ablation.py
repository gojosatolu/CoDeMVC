import os
import re
import pandas as pd
from tqdm import tqdm

def parse_log_file(file_path):
    """
    解析消融实验日志，提取 Alpha, Beta, ACC, NMI, PUR。
    支持 UTF-16LE 和 UTF-8 编码。
    """
    # 尝试多种编码加载 (PowerShell 默认可能是 utf-16)
    content = ""
    for enc in ['utf-16', 'utf-8', 'utf-16-le', 'gbk']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
                if "The best clustering performace" in content:
                    break
        except:
            continue
    
    if not content: return None

    # 正则提取指标
    # 样板: The best clustering performace: ACC = 0.8500 NMI = 0.8123 PUR=0.8650
    metric_pattern = r"The best clustering performace: ACC = ([\d\.]+) NMI = ([\d\.]+) PUR=([\d\.]+)"
    match = re.search(metric_pattern, content)
    
    if match:
        # 文件名提取 A 和 B: Log_ORL_A0_B0.1.txt
        fname = os.path.basename(file_path)
        # 使用更加严谨的正则，排除末尾的 .txt 干扰
        a_match = re.search(r"_A([\d\.]+?)(?=_B|_|\.txt)", fname)
        b_match = re.search(r"_B([\d\.]+?)(?=_|\.txt)", fname)
        
        alpha = float(a_match.group(1)) if a_match else 0.0
        beta = float(b_match.group(1)) if b_match else 0.0

        
        return {
            'Alpha': alpha,
            'Beta': beta,
            'ACC': float(match.group(1)),
            'NMI': float(match.group(2)),
            'PUR': float(match.group(3))
        }
    return None

def main():
    results_root = r"E:\CoDeMVC\ablation_results"
    all_data = []

    if not os.path.exists(results_root):
        print(f"[!] Error: Path {results_root} not found.")
        return

    # 遍历数据集文件夹
    datasets = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
    
    print(f"[*] Analyzing {len(datasets)} datasets in {results_root}...")

    for ds in datasets:
        ds_path = os.path.join(results_root, ds)
        logs = [f for f in os.listdir(ds_path) if f.endswith('.txt')]
        
        for log in logs:
            res = parse_log_file(os.path.join(ds_path, log))
            if res:
                res['Dataset'] = ds
                all_data.append(res)

    if not all_data:
        print("[!] No valid logs found. Check if training is completed.")
        return

    df = pd.DataFrame(all_data)
    
    # 汇总最佳结果
    print("\n" + "="*80)
    print(f"{'Dataset':<15} | {'Alpha':<6} | {'Beta':<6} | {'ACC':<7} | {'NMI':<7} | {'PUR':<7}")
    print("-" * 80)

    summary_rows = []
    for ds in df['Dataset'].unique():
        sub_df = df[df['Dataset'] == ds]
        # 优先按 ACC 排序，其次 NMI，再次 PUR
        best_idx = sub_df.sort_values(by=['ACC', 'NMI', 'PUR'], ascending=False).index[0]
        best_row = sub_df.loc[best_idx]
        
        print(f"{ds:<15} | {best_row['Alpha']:<6.1f} | {best_row['Beta']:<6.1f} | "
              f"{best_row['ACC']:<7.4f} | {best_row['NMI']:<7.4f} | {best_row['PUR']:<7.4f}")
        
        best_row_dict = best_row.to_dict()
        summary_rows.append(best_row_dict)

    print("="*80)

    # 导出 CSV 方便复制到 Excel/论文容器
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("ablation_summary_best.csv", index=False)
    print(f"\n[+] Detailed summary saved to ablation_summary_best.csv")

if __name__ == "__main__":
    main()
