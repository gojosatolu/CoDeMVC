import os
import re
import pandas as pd

def parse_log_file(file_path):
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

    metric_pattern = r"The best clustering performace: ACC = ([\d\.]+) NMI = ([\d\.]+) PUR=([\d\.]+)"
    match = re.search(metric_pattern, content)
    
    if match:
        fname = os.path.basename(file_path)
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

def get_best_for_dataset(root_dir, dataset):
    ds_path = os.path.join(root_dir, dataset)
    if not os.path.exists(ds_path):
        return None
    
    logs = [f for f in os.listdir(ds_path) if f.endswith('.txt')]
    all_res = []
    for log in logs:
        res = parse_log_file(os.path.join(ds_path, log))
        if res: all_res.append(res)
        
    if not all_res: return None
    
    df = pd.DataFrame(all_res)
    # 按照 ACC > NMI > PUR 的优先级找出最好的一轮
    best_row = df.sort_values(by=['ACC', 'NMI', 'PUR'], ascending=False).iloc[0]
    return best_row

def main():
    fixed_root = r"E:\CoDeMVC\ablation_results"
    learn_root = r"E:\CoDeMVC\ablation_results_learnable"
    
    datasets = ["ORL", "OutdoorScene", "EYaleB"]
    
    print("\n" + "="*95)
    print(" 🎯 动态因果注入因子 (Learnable Epsilon) 有效性全面验证报告")
    print("="*95)
    print(f"{'Dataset':<15} | {'Version':<18} | {'Opt. α':<7} | {'Opt. β':<7} | {'ACC':<8} | {'NMI':<8} | {'PUR':<8}")
    print("-" * 95)

    for ds in datasets:
        fixed_best = get_best_for_dataset(fixed_root, ds)
        learn_best = get_best_for_dataset(learn_root, ds)
        
        # 打印 Fixed 版本
        if fixed_best is not None:
            print(f"{ds:<15} | {'Fixed (ε=0.1)':<18} | {fixed_best['Alpha']:<7.1f} | {fixed_best['Beta']:<7.1f} | {fixed_best['ACC']:<8.4f} | {fixed_best['NMI']:<8.4f} | {fixed_best['PUR']:<8.4f}")
        else:
            print(f"{ds:<15} | {'Fixed (ε=0.1)':<18} | {'N/A':<7} | {'N/A':<7} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
            
        # 打印 Learnable 版本
        if learn_best is not None:
            imp_acc = ""
            if fixed_best is not None:
                diff = learn_best['ACC'] - fixed_best['ACC']
                if diff > 0: imp_acc = f"(🚀 +{diff*100:.2f}%)"
                elif diff < 0: imp_acc = f"(⚠️ {diff*100:.2f}%)"
                else: imp_acc = "(=)"
                
            print(f"{'':<15} | {'Learnable (ε=var)':<18} | {learn_best['Alpha']:<7.1f} | {learn_best['Beta']:<7.1f} | {learn_best['ACC']:<8.4f} {imp_acc} | {learn_best['NMI']:<8.4f} | {learn_best['PUR']:<8.4f}")
        else:
            print(f"{'':<15} | {'Learnable (ε=var)':<18} | {'N/A':<7} | {'N/A':<7} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
            
        print("-" * 95)
        
    print("="*95)
    print(" 💡 结论指引：如果 Learnable 版本出现 🚀，则证明自适应 Epsilon 成功突破了固定权重的瓶颈。")
    print("="*95)

if __name__ == "__main__":
    main()
