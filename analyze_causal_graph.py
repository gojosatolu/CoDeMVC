import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib
from itertools import combinations
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 核心路径集成
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
causality_lab_path = os.path.join(project_root, 'causality-lab-main')
sys.path.insert(0, causality_lab_path)

from network import Network
from causal_reasoning.cleann_explainer import CLEANN
from plot_utils.draw_graph import draw_node, draw_edge
from plot_utils.graph_layout import ForceDirectedLayout
from causal_discovery_utils.performance_measures import calc_skeleton_accuracy
from graphical_models import PAG, DAG
from dataloader import load_data

class CITestTqdmWrapper:
    def __init__(self, original_test, pbar):
        self.original_test = original_test
        self.pbar = pbar
    def __getattr__(self, name):
        return getattr(self.original_test, name)
    def calc_statistic(self, x, y, zz):
        self.pbar.update(1)
        return self.original_test.calc_statistic(x, y, zz)

def dummy_dims_mapping(dataset, view):
    mapping = {
        'ORL': [4096, 3304, 6750], 'ProteinFold': [27]*12, 
        'Prokaryotic': [438, 3, 393], 'BDGP': [1750, 79],
        'NoisyBDGP': [1750, 79], 'Caltech-5V': [40, 254, 928, 512, 1984],
        'NoisyHW': [216, 76, 64, 6, 240, 47], 'NoisyMNIST': [784, 784],
        'NUS-WIDE': [64, 144, 73, 128, 225], 'OutdoorScene': [512, 432, 256, 48],
        'Scene-15': [20, 59, 40]
    }
    return mapping.get(dataset, [1000] * view)

def get_feature_attribution(model, view_idx, latent_node_idx, top_k=3):
    with torch.no_grad():
        encoder_module = model.encoders[view_idx].encoder
        weights = [l.weight.data for l in encoder_module if isinstance(l, torch.nn.Linear)]
        if not weights: return []
        combined_w = weights[0]
        for v_w in weights[1:]: combined_w = torch.matmul(v_w, combined_w)
        importance = torch.abs(combined_w[latent_node_idx])
        return torch.topk(importance, min(top_k, importance.shape[0])).indices.cpu().numpy().tolist()

def enhanced_padding_draw_graph(graph, node_size_factor=0.6, factor=1000, iterations=500):
    node_radius = 0.04 * node_size_factor
    layout = ForceDirectedLayout(graph, (-factor, factor), (-factor, factor), num_iterations=iterations)
    pos_dict = layout.calc_layout()
    nodes = list(graph.nodes_set)
    for node in nodes:
        norm_pos = (pos_dict[node] / factor + 1) / 2
        pos_dict[node] = 0.1 + 0.8 * norm_pos
    min_dist = 2.2 * node_radius
    for _ in range(150):
        collision = False
        for i, j in combinations(nodes, 2):
            p1, p2 = pos_dict[i], pos_dict[j]
            d = np.linalg.norm(p1 - p2)
            if d < min_dist:
                push = (min_dist - d) / 2
                vec = (p1 - p2) / (d + 1e-8)
                pos_dict[i] += vec * push
                pos_dict[j] -= vec * push
                collision = True
        if not collision: break
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if isinstance(graph, (PAG, DAG)):
        for node_i, node_j in combinations(graph.nodes_set, 2):
            if graph.is_connected(node_i, node_j):
                try:
                    mark_i = graph.get_edge_mark(node_j, node_i)
                    mark_j = graph.get_edge_mark(node_i, node_j)
                except: mark_i, mark_j = None, None
                draw_edge(ax, pos_dict[node_i], pos_dict[node_j], node_radius, mark_i, mark_j, line_color='blue')
    font_style = {'fontsize': 14, 'fontweight': 'bold'}
    for node in graph.nodes_set:
        draw_node(ax, pos_dict[node], node_radius, node_name=str(node), 
                  line_color='black', fill_color='white', text_color='black', font=font_style)
    return fig

def main():
    parser = argparse.ArgumentParser(description="AC2-MVC Causal Analysis & Intervention Engine")
    parser.add_argument('--dataset', type=str, default='ORL')
    parser.add_argument('--view', type=int, default=3)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--p_val', type=float, default=0.05)
    parser.add_argument('--max_order', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='')
    args = parser.parse_args()

    dims = dummy_dims_mapping(args.dataset, args.view)
    model = Network(view=args.view, input_size=dims, feature_dim=args.feature_dim, high_feature_dim=20, device='cpu', use_crm=True)
    
    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        print(f"[*] Loaded model: {args.model_path}")

    model.eval()
    
    # 0. 数据加载 (提前加载以获取类别数和样本量)
    dataset, _, _, num_samples, class_num = load_data(args.dataset)
    print(f"[*] Dataset: {args.dataset}, Samples: {num_samples}, Clusters: {class_num}")

    # 1. 提取注意力矩阵
    attention_matrices = []
    with torch.no_grad():
        for v in range(args.view):
            causal_attn = model.encoders[v].causal_attention
            Q, K, scale = causal_attn.Q, causal_attn.K, causal_attn.scale
            A = torch.nn.functional.softmax(torch.matmul(Q, K.t()) / scale, dim=-1)
            attention_matrices.append(A.cpu().numpy())

    if not os.path.exists('analysis_reports'): os.makedirs('analysis_reports')
    
    learned_graphs = []
    hub_nodes = []
    for v in range(args.view):

        print(f"\n  -> Stage 1: Structural Causal Discovery (View {v+1}/{args.view})")
        explainer = CLEANN(attention_matrix=attention_matrices[v], num_samples=num_samples, p_val_th=args.p_val, explanation_tester=None)
        
        with tqdm(desc=f"     [Causal Engine]", unit=" test") as pbar:
            wrapped_ci = CITestTqdmWrapper(explainer.ci_test, pbar)
            icd_alg = explainer.StructureLearning(nodes_set=explainer.nodes_set, ci_test=wrapped_ci)
            done = False
            while not done:
                current_order = icd_alg._state.get('cond_set_size', 0)
                if current_order > args.max_order: break
                pbar.set_postfix({"order": current_order})
                done, _ = icd_alg.learn_structure_iteration()
        
        pag_graph = icd_alg.graph
        # 骨干提取
        corr_mat = explainer.ci_test.correlation_matrix
        edge_candidates = []
        for i, j in combinations(pag_graph.nodes_set, 2):
            if pag_graph.is_connected(i, j):
                edge_candidates.append((i, j, abs(corr_mat[i, j])))
        edge_candidates.sort(key=lambda x: x[2], reverse=True)
        top_k = min(40, len(edge_candidates))
        strong_edges = set([tuple(sorted((e[0], e[1]))) for e in edge_candidates[:top_k]])
        for i, j in combinations(pag_graph.nodes_set, 2):
            if pag_graph.is_connected(i, j) and tuple(sorted((i, j))) not in strong_edges:
                pag_graph.delete_edge(i, j)
        
        learned_graphs.append(pag_graph)
        fig = enhanced_padding_draw_graph(pag_graph, node_size_factor=0.6, iterations=500)
        fig.savefig(f"analysis_reports/{args.dataset}_View{v+1}_NativePAG.png", dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)

        adj_m = pag_graph.get_skeleton_mat()
        hub_node = int(np.argmax(adj_m.sum(axis=0) + adj_m.sum(axis=1)))
        hub_nodes.append(hub_node)
        print(f"     [+] Structural Hub Detected: Node {hub_node}")

    # 第二阶段：Virtual Intervention (do-calculus) 实验
    print(f"\n  -> Stage 2: Causal Intervention Stress Test (do-calculus)...")
    # 此处无需再次 load_data
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    xs, _, _ = next(iter(data_loader))
    for v in range(args.view): xs[v] = xs[v].to('cpu')
    
    with torch.no_grad():
        _, zs_orig, _, H_orig, _ = model(xs)
        # 获取原始聚类标签（作为基准）
        kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=42)
        y_orig = kmeans.fit_predict(H_orig.numpy())

    # 遍历 View 并进行干预测试
    for v in range(args.view):
        impact_scores = []
        for node_idx in tqdm(range(args.feature_dim), desc=f"     [do(Node_i=0) View {v+1}]"):
            with torch.no_grad():
                # 实施干预：克隆特征并置零一个维度
                intervened_zs = [z.clone() for z in zs_orig]
                intervened_zs[v][:, node_idx] = 0 # do-算子
                # 重新融合
                H_interv = model.feature_fusion(intervened_zs, zs_gradient=False)
                y_interv = kmeans.predict(H_interv.numpy()) # 使用相同的 KMeans 模型
                
                # 计算翻转率 (Label Flip Rate)
                lfr = np.mean(y_orig != y_interv)
                impact_scores.append(lfr)
        
        top_impact_node = int(np.argmax(impact_scores))
        max_lfr = impact_scores[top_impact_node]
        
        print(f"\n[============= Causal Unification Report: View {v+1} =============]")
        print(f"  * Structural Hub (PAG): Node {hub_nodes[v]}")
        print(f"  * Top-Impact Node (do): Node {top_impact_node} (LFR={max_lfr:.4f})")
        
        # 核心论证逻辑
        if hub_nodes[v] == top_impact_node or max_lfr > 0.1:
            if hub_nodes[v] == top_impact_node:
                status = "PERFECT UNIFICATION! (Theory matched Reality)"
            else:
                status = "STRONG ALIGNMENT (High Sensitivity Detected)"
            print(f"  * Status: {status}")
            print(f"  * Scientific Conclusion: Node {top_impact_node} is a CRITICAL causal driver for clustering.")
            print(f"  * Physical Root: {get_feature_attribution(model, v, top_impact_node)}")
        else:
            print(f"  * Status: DIVERGENT (Potential non-linear complexity)")
        print(f"[=============================================================]")

    # 一致性指标汇总
    if len(learned_graphs) >= 2:
        print(f"\n[==================== Alignment Metrics =====================]")
        ref_g = learned_graphs[0]
        for j in range(1, len(learned_graphs)):
            try:
                acc = calc_skeleton_accuracy(learned_graphs[j], ref_g)
                print(f"  * View 1 vs View {j+1}: F1={acc['edge_F1']:.4f}")
            except ZeroDivisionError: print(f"  * View 1 vs View {j+1}: F1=0.0000")
        print(f"[=============================================================]")

if __name__ == "__main__":
    main()
