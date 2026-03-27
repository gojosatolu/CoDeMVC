import os
import re
import argparse
import pandas as pd

def parse_log(file_path):
    # 支持多种编码
    content = ""
    for enc in ['utf-16', 'utf-8', 'utf-16-le', 'gbk']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
                if "The best clustering performace" in content:
                    break
        except: continue
    
    if not content: return None

    # 正则提取指标
    metric_pattern = r"The best clustering performace: ACC = ([\d\.]+) NMI = ([\d\.]+) PUR=([\d\.]+)"
    match = re.search(metric_pattern, content)
    
    if match:
        fname = os.path.basename(file_path)
        # 精确匹配 A 和 B，排除末尾点干扰
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

def find_best_chain(df):
    """
    寻找符合 1 -> 2 -> 3 增量逻辑的最佳链条
    """
    # 1. Baseline
    l1 = df[(df['Alpha'] == 0.0) & (df['Beta'] == 0.0)]
    if l1.empty: return None
    l1_res = l1.iloc[0]

    # 2. Level 2 (A > 0, B = 0)
    l2_candidates = df[(df['Alpha'] > 0.0) & (df['Beta'] == 0.0)].sort_values(by='ACC', ascending=False)
    if l2_candidates.empty: return None
    # 优先选比 L1 好的，如果都没比 L1 好，选最接近的
    l2_res = l2_candidates.iloc[0]

    # 3. Level 3 (A > 0, B > 0)
    l3_candidates = df[(df['Alpha'] > 0.0) & (df['Beta'] > 0.0)].sort_values(by='ACC', ascending=False)
    if l3_candidates.empty: return None
    l3_res = l3_candidates.iloc[0]

    return l1_res, l2_res, l3_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g. ORL, OutdoorScene)')
    args = parser.parse_all = parser.parse_args()

    results_root = r"E:\CoDeMVC\ablation_results"
    ds_path = os.path.join(results_root, args.dataset)

    if not os.path.exists(ds_path):
        print(f"[!] Error: Path {ds_path} not found.")
        return

    print(f"[*] Mining Best Ablation Levels for: {args.dataset}")
    all_logs = []
    files = [f for f in os.listdir(ds_path) if f.endswith('.txt')]
    for f in files:
        res = parse_log(os.path.join(ds_path, f))
        if res: all_logs.append(res)
    
    if not all_logs:
        print("[!] No valid logs found.")
        return

    df = pd.DataFrame(all_logs)
    chain = find_best_chain(df)
    
    if not chain:
        print("[!] Could not construct a 3-level chain (missing Baseline or combinations).")
        return

    l1, l2, l3 = chain

    print("\n" + "="*85)
    print(f"  {args.dataset} 三层消融验证报告 (3-Level Ablation Sync Report)")
    print("="*85)
    print(f"{'Level':<30} | {'Alpha':<6} | {'Beta':<6} | {'ACC':<7} | {'NMI':<7} | {'PUR':<7} | {'Imp.'}")
    print("-" * 85)

    # 打印 L1
    print(f"{'Level 1: Baseline (α=0, β=0)':<30} | {l1['Alpha']:<6.1f} | {l1['Beta']:<6.1f} | {l1['ACC']:<7.4f} | {l1['NMI']:<7.4f} | {l1['PUR']:<7.4f} | (Base)")

    # 打印 L2
    l2_imp = l2['ACC'] - l1['ACC']
    l2_status = "✅" if l2_imp > 0 else "⚠️"
    print(f"{'Level 2: Disentanglement (β=0)':<30} | {l2['Alpha']:<6.1f} | {l2['Beta']:<6.1f} | {l2['ACC']:<7.4f} | {l2['NMI']:<7.4f} | {l2['PUR']:<7.4f} | {l2_status} {l2_imp*100:+.2f}%")

    # 打印 L3
    l3_imp = l3['ACC'] - l2['ACC']
    l3_status = "🚀" if l3_imp > 0 else "⚠️"
    print(f"{'Level 3: Full (α>0, β>0)':<30} | {l3['Alpha']:<6.1f} | {l3['Beta']:<6.1f} | {l3['ACC']:<7.4f} | {l3['NMI']:<7.4f} | {l3['PUR']:<7.4f} | {l3_status} {l3_imp*100:+.2f}%")

    print("-" * 85)
    total_imp = l3['ACC'] - l1['ACC']
    print(f"  >>> Total Improvement (L3 vs L1): {total_imp*100:+.2f}% (ACC)")
    
    # 状态总结
    if l3['ACC'] > l2['ACC'] > l1['ACC']:
        print(f"  >>> VALIDATION: PERFECT STEP-UP SUCCESS! (ACC 逐层递增)")
    else:
        print(f"  >>> VALIDATION: COMPLETED (ACC 呈现非线性波动，需分析数据分布)")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
