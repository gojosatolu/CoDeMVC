import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.nn.functional import cosine_similarity, normalize

# Import strict project modules
from network import Network
from causal_module import CausalDebiasedMultiViewClustering
from dataloader import load_data

# Set plotting style suitable for academic papers
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis: Stability & Visualization")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pth file')
    parser.add_argument('--baseline_path', type=str, help='Path to baseline .pth (optional for comparison)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--high_feature_dim', type=int, default=20)
    args = parser.parse_args()
    return args

def load_trained_model(args, path, dims, view):
    """Load model and causal module from checkpoint"""
    print(f"Loading backbone from {path}...")
    
    # Initialize networks
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, args.device)
    
    # Initialize Causal Module
    causal_module = CausalDebiasedMultiViewClustering(
        num_views=view,
        feature_dim=args.feature_dim,
        device=args.device
    ).to(args.device)

    try:
        # 1. Load Backbone
        checkpoint = torch.load(path, map_location=args.device)
        model.load_state_dict(checkpoint, strict=False)
        model.to(args.device)
        model.eval()
        
        # 2. Load Causal Module (Try separate file first, then main file)
        # Infer causal path: models/Dataset.pth -> models/Dataset_causal.pth
        causal_path = path.replace('.pth', '_causal.pth')
        
        if os.path.exists(causal_path):
            print(f"Loading causal module from {causal_path}...")
            causal_ckpt = torch.load(causal_path, map_location=args.device)
            causal_module.load_state_dict(causal_ckpt, strict=False)
            print("  [INFO] Causal weights loaded from separate file.")
        else:
            # Fallback: Try to find causal weights in the main checkpoint
            print(f"  [INFO] Separate causal file not found. Checking main checkpoint...")
            causal_keys = {k: v for k, v in checkpoint.items() if 'disentanglers' in k}
            if causal_keys:
                causal_module.load_state_dict(checkpoint, strict=False)
                print("  [INFO] Causal weights loaded from main checkpoint.")
            else:
                print("  [WARN] No causal weights found! Using random initialization (Results may be invalid).")

        return model, causal_module

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def stability_test(args, model, causal_module, data_loader, view, is_baseline=False):
    """
    Experiment A: Style Intervention Stability Test
    Measure Cosine Similarity between S (original) and S_cf (counterfactual)
    """
    label = "Baseline" if is_baseline else "CoDe-MVC"
    print(f"\n>>> Running Experiment A: Stability Test ({label})...")
    
    similarities = []
    
    # Standard lambda for boxplot
    lambda_val = 0.4
    
    with torch.no_grad():
        for xs, _, _ in data_loader:
            for v in range(view):
                xs[v] = xs[v].to(args.device)
            
            # 1. Encode
            zs = []
            for v in range(view):
                zs.append(model.encoders[v](xs[v]))
            
            if is_baseline:
                # FAIR BASELINE COMPARISON:
                # Baseline has no Disentangler. We simulate "Blind Mixing" perturbation on Z directly.
                # Stability = Cosine(Z, Z_mixed)
                # This tests: "If we perturb the features blindly, does the model hold?"
                for v in range(view):
                    z = zs[v]
                    idx = torch.randperm(z.size(0))
                    # Linear Mixup on Feature Space directly
                    z_mixed = (1 - lambda_val) * z + lambda_val * z[idx]
                    
                    sim = cosine_similarity(z, z_mixed, dim=1)
                    similarities.append(sim.cpu().numpy())
            else:
                # CoDe-MVC:
                # Uses educated Causal Disentanglement + Style Mixup
                # Stability = Cosine(S, S_cf)
                # Tests: "If we perturb ONLY style, does S hold?"
                
                # NOTE: We must manually trigger mixup with specific lambda here if possible
                # For now, we rely on the Module's internal logic which uses 0.4 by default
                s_list, s_cf_list, _, _ = causal_module(zs, return_counterfactuals=True)
                
                for v in range(view):
                    sim = cosine_similarity(s_list[v], s_cf_list[v], dim=1)
                    similarities.append(sim.cpu().numpy())
    
    # Flatten
    similarities = np.concatenate(similarities)
    
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    print(f"  Stability Score (Cosine Sim): {mean_sim:.4f} ± {std_sim:.4f}")
    return similarities

def visualize_tsne(args, model, causal_module, data_loader, view, save_name="tsne_viz.png"):
    """
    Experiment B: t-SNE Visualization with Counterfactual Perturbation
    """
    print(f"\n>>> Running Experiment B: t-SNE Visualization ({save_name})...")
    
    all_s = []
    all_s_cf = []
    labels = []
    
    # Collect small batch of data (e.g., first 500 samples)
    count = 0
    limit = 1000
    
    with torch.no_grad():
        for xs, y, _ in data_loader:
            if count >= limit: break
            
            for v in range(view):
                xs[v] = xs[v].to(args.device)
            
            # Encode
            zs = []
            for v in range(view):
                zs.append(model.encoders[v](xs[v]))
            
            # Causal Process
            s_list, s_cf_list, _, _ = causal_module(zs, return_counterfactuals=True)
            
            # We use View 0 for visualization typicality
            all_s.append(s_list[0].cpu().numpy())
            all_s_cf.append(s_cf_list[0].cpu().numpy())
            labels.append(y.numpy())
            
            count += xs[0].size(0)
    
    S = np.concatenate(all_s)[:limit]
    S_cf = np.concatenate(all_s_cf)[:limit]
    Y = np.concatenate(labels)[:limit].squeeze()
    
    # Run t-SNE
    print("  Comparing t-SNE embeddings...")
    # Optimized for better cluster separation visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, init='pca', learning_rate=200)
    
    # Combine to ensure same space mapping
    combined = np.vstack([S, S_cf])
    embedded = tsne.fit_transform(combined)
    
    S_emb = embedded[:len(S)]
    S_cf_emb = embedded[len(S):]
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot Original S
    scatter1 = axes[0].scatter(S_emb[:, 0], S_emb[:, 1], c=Y, cmap='tab10', s=10, alpha=0.6)
    axes[0].set_title("Original Semantic Features (S)")
    axes[0].axis('off')
    
    # Plot Counterfactual S_cf
    scatter2 = axes[1].scatter(S_cf_emb[:, 0], S_cf_emb[:, 1], c=Y, cmap='tab10', s=10, alpha=0.6)
    axes[1].set_title("Counterfactual Features (S_cf) [Style Perturbed]")
    axes[1].axis('off')
    
    # Add colorbar
    plt.colorbar(scatter1, ax=axes.ravel().tolist())
    
    plt.suptitle(f"Causal Invariance Visualization: {args.dataset}", fontsize=16)
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    print(f"  Saved visualization to {save_name}")
    plt.close()

def visualize_disentanglement(args, model, causal_module, data_loader, view, save_name="disentangle_viz.png"):
    """
    Experiment D: Disentanglement Verification (Content S vs Style N)
    This proves that S contains semantics (clusters) while N contains only noise/style (random).
    """
    print(f"\n>>> Running Experiment D: Disentanglement Verification ({save_name})...")
    
    all_s = []
    all_n = []
    labels = []
    
    # Collect small batch
    count = 0
    limit = 1000
    
    with torch.no_grad():
        for xs, y, _ in data_loader:
            if count >= limit: break
            
            for v in range(view):
                xs[v] = xs[v].to(args.device)
            
            # Encode
            zs = []
            for v in range(view):
                zs.append(model.encoders[v](xs[v]))
            
            # Extract S and N
            # We use View 0 for visualization
            s, n = causal_module.disentanglers[0](zs[0])
            
            all_s.append(s.cpu().numpy())
            all_n.append(n.cpu().numpy())
            labels.append(y.numpy())
            
            count += xs[0].size(0)
    
    S = np.concatenate(all_s)[:limit]
    N = np.concatenate(all_n)[:limit]
    Y = np.concatenate(labels)[:limit].squeeze()
    
    # Run t-SNE
    print("  Computing t-SNE for S and N...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate=200)
    
    # Combine to ensure scale comparability (opt) or just fit separately
    # Fitting separately is usually safer for distinct manifolds
    S_emb = tsne.fit_transform(S)
    N_emb = tsne.fit_transform(N)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot Content S
    scatter1 = axes[0].scatter(S_emb[:, 0], S_emb[:, 1], c=Y, cmap='tab10', s=10, alpha=0.7)
    axes[0].set_title("Causal Content (S)\n[Expect: Clear Clusters]")
    axes[0].axis('off')
    
    # Plot Style N
    scatter2 = axes[1].scatter(N_emb[:, 0], N_emb[:, 1], c=Y, cmap='tab10', s=10, alpha=0.7)
    axes[1].set_title("Spurious Style (N)\n[Expect: Random/Meshed Distribution]")
    axes[1].axis('off')
    
    plt.colorbar(scatter1, ax=axes.ravel().tolist())
    plt.suptitle(f"Disentanglement Verification: {args.dataset}", fontsize=16)
    
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    print(f"  Disentanglement Viz saved to {save_name}")
    plt.close()

def plot_stability_comparison(sim_base, sim_full, dataset_name):
    """Draw Boxplot comparing Baseline vs Full Model stability"""
    plt.figure(figsize=(6, 8))
    
    data = [sim_base, sim_full]
    labels = ['Baseline\n(Unregularized)', 'CoDe-MVC\n(Ours)']
    
    sns.boxplot(data=data, palette=['#e74c3c', '#2ecc71'], width=0.5)
    plt.xticks(range(2), labels)
    plt.ylabel('Cosine Similarity (S vs S_cf)')
    plt.title(f'Style Intervention Stability\n({dataset_name})')
    plt.ylim(0, 1.1)
    
    save_path = f"{dataset_name}_stability_boxplot.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"  Boxplot saved to {save_path}")

def sensitivity_test(args, model, causal_module, data_loader, view, is_baseline=False):
    """
    Experiment C: Intervention Strength Sensitivity
    Vary lambda_mix from 0.0 to 1.0 and observe stability drop
    """
    label = "Baseline" if is_baseline else "CoDe-MVC"
    print(f"\n>>> Running Experiment C: Sensitivity Test ({label})...")
    
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []
    
    with torch.no_grad():
        for lam in lambdas:
            batch_sims = []
            
            for xs, _, _ in data_loader:
                 # Only use first batch for speed
                for v in range(view): xs[v] = xs[v].to(args.device)
                
                zs = [model.encoders[v](xs[v]) for v in range(view)]
                
                if is_baseline:
                    # Blind Mixup on Z
                    for v in range(view):
                        z = zs[v]
                        idx = torch.randperm(z.size(0))
                        z_mixed = (1 - lam) * z + lam * z[idx]
                        sim = cosine_similarity(z, z_mixed, dim=1).mean().item()
                        batch_sims.append(sim)
                else:
                    # CoDe-MVC Style Mixup
                    # We need to manually perform the mixup logic here to control lambda
                    s_list, _, _, _ = causal_module(zs, return_counterfactuals=False) # Get S/N
                    
                    # Manually Mix N
                    s_cf_list_manual = []
                    for v in range(view):
                        dis = causal_module.disentanglers[v]
                        s, n = s_list[v], causal_module.disentanglers[v](zs[v])[1] # Extract N
                        
                        idx = torch.randperm(n.size(0))
                        n_mixed = (1 - lam) * n + lam * n[idx]
                        
                        z_cf = dis.reconstruct(s, n_mixed)
                        s_cf_extracted, _ = dis(z_cf)
                        
                        sim = cosine_similarity(s, s_cf_extracted, dim=1).mean().item()
                        batch_sims.append(sim)
                
                break # Just one batch is enough for trend

            avg_sim = np.mean(batch_sims)
            results.append(avg_sim)
            print(f"  Lambda={lam:.1f} -> Stability={avg_sim:.4f}")
            
    return lambdas, results

def plot_sensitivity(lambdas, res_base, res_full, dataset_name):
    """Draw Sensitivity Line Chart"""
    plt.figure(figsize=(8, 6))
    
    plt.plot(lambdas, res_full, marker='o', linewidth=2, label='CoDe-MVC (Ours)', color='#2ecc71')
    if res_base:
        plt.plot(lambdas, res_base, marker='s', linewidth=2, linestyle='--', label='Baseline', color='#e74c3c')
    
    plt.xlabel('Intervention Strength ($\lambda_{mix}$)')
    plt.ylabel('Representation Stability (Cosine Sim)')
    plt.title(f'Robustness to Perturbation Strength\n({dataset_name})')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(0, 1.1)
    
    save_path = f"{dataset_name}_sensitivity_curve.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"  Sensitivity Curve saved to {save_path}")

def main():
    args = parse_args()
    
    # Load Data
    print(f"Loading Dataset: {args.dataset}")
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, drop_last=True # Needs drop_last for mixup safety
    )
    
    # 1. Evaluate Target Model (Full Model) -----------------------
    print(f"\n--- Evaluating Target Model: {args.model_path} ---")
    model, causal = load_trained_model(args, args.model_path, dims, view)
    if model is None: return

    # Experiment A
    sim_full = stability_test(args, model, causal, data_loader, view, is_baseline=False)
    visualize_tsne(args, model, causal, data_loader, view, save_name=f"{args.dataset}_tsne_full.png")
    
    # Experiment D (Disentanglement)
    visualize_disentanglement(args, model, causal, data_loader, view, save_name=f"{args.dataset}_disentangle_viz.png")
    
    # Experiment C (Sensitivity)
    lambdas, sense_full = sensitivity_test(args, model, causal, data_loader, view, is_baseline=False)

    # 2. Evaluate Baseline (Optional) -----------------------------
    sense_base = None
    if args.baseline_path and os.path.exists(args.baseline_path):
        print(f"\n--- Evaluating Baseline Model: {args.baseline_path} ---")
        model_base, causal_base = load_trained_model(args, args.baseline_path, dims, view)
        
        # Pass is_baseline=True to trigger Fair Comparison Mode
        sim_base = stability_test(args, model_base, causal_base, data_loader, view, is_baseline=True)
        
        # Note: Baseline t-SNE with causal module logic is tricky if no weights exist.
        # We skip t-SNE for baseline here to perform Sensitivity Test instead.
        
        # Experiment C (Sensitivity)
        _, sense_base = sensitivity_test(args, model_base, causal_base, data_loader, view, is_baseline=True)
        
        # Plot Comparison
        plot_stability_comparison(sim_base, sim_full, args.dataset)
        plot_sensitivity(lambdas, sense_base, sense_full, args.dataset)
    else:
        print("\n[INFO] No baseline path provided. Skipping comparison plots.")
        # Plot single curve if needed
        plot_sensitivity(lambdas, None, sense_full, args.dataset)

if __name__ == "__main__":
    main()
