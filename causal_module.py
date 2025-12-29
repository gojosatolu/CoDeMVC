import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CausalDisentangler(nn.Module):
    """
    Disentangles features into Causal Content (S) and Spurious Style (N).
    Includes a Combiner to reconstruct the original features or generate counterfactuals.
    """
    def __init__(self, feature_dim, content_dim=None):
        super(CausalDisentangler, self).__init__()
        self.feature_dim = feature_dim
        # Default content_dim is half of feature_dim
        self.content_dim = content_dim if content_dim is not None else feature_dim // 2
        self.style_dim = feature_dim - self.content_dim
        
        # Content Encoder: Extracts causal/semantic factors
        self.content_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, self.content_dim),
            nn.BatchNorm1d(self.content_dim)
        )
        
        # Style Encoder: Extracts view-specific/noise factors
        self.style_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, self.style_dim),
            nn.BatchNorm1d(self.style_dim)
        )
        
        # Combiner: Reconstructs feature from (S, N)
        self.combiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        s = self.content_encoder(x)
        n = self.style_encoder(x)
        return s, n
    
    def reconstruct(self, s, n):
        combined = torch.cat([s, n], dim=1)
        return self.combiner(combined)

class CausalDebiasedMultiViewClustering(nn.Module):
    """
    Revised Causal Module using Disentanglement and Counterfactual Generation.
    Strategy:
    1. Disentangle z -> (s, n)
    2. Swap 'n' to generate Counterfactual z_cf
    3. Ensure s is invariant to this swap.
    """
    def __init__(self, num_views, feature_dim, device):
        super(CausalDebiasedMultiViewClustering, self).__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.device = device
        
        # Shared Disentangler across views (or separate? enforcing shared semantic space implies shared disentangler might be better, 
        # but views have different properties. Let's use separate disentanglers but project to shared content relationship).
        # Actually, for SCMVC, the encoders are view-specific. The 'content' S should be aligned.
        # We use a ModuleList of disentanglers.
        self.disentanglers = nn.ModuleList([
            CausalDisentangler(feature_dim) for _ in range(num_views)
        ])
        
    def forward(self, view_features_list, return_counterfactuals=True):
        """
        Args:
            view_features_list: List of [B, D]
        Returns:
            s_list: Content features [B, S_dim]
            s_cf_list: Counterfactual Content features
            z_rec_list: Reconstructed original features
            z_cf_list: Counterfactual features (useful for other tasks, or debugging)
        """
        s_list = []
        n_list = []
        z_rec_list = []
        
        # 1. Disentangle and Reconstruct
        for v in range(self.num_views):
            z = view_features_list[v]
            dis = self.disentanglers[v]
            
            s, n = dis(z)
            z_rec = dis.reconstruct(s, n)
            
            s_list.append(s)
            n_list.append(n)
            z_rec_list.append(z_rec)
            
        if not return_counterfactuals:
            return s_list, None, z_rec_list, None

        # 2. Generate Counterfactuals (Style Mixup instead of Hard Shuffle)
        s_cf_list = []
        z_cf_list = []
        
        for v in range(self.num_views):
            dis = self.disentanglers[v]
            s = s_list[v]
            n = n_list[v]
            
            # Soft Mixup: Instead of replacing 100% of Style, we mix it.
            # This ensures counterfactuals stay closer to the data manifold.
            idx = torch.randperm(n.size(0))
            lambda_mix = 0.4  # 40% of foreign style, 60% original
            n_mixed = (1 - lambda_mix) * n + lambda_mix * n[idx]
            
            # Generate z_cf
            z_cf = dis.reconstruct(s, n_mixed)
            z_cf_list.append(z_cf)
            
            # Extract s from z_cf (Should be close to s)
            s_extracted, _ = dis(z_cf)
            s_cf_list.append(s_extracted)
            
        return s_list, s_cf_list, z_rec_list, z_cf_list

class CausalContrastiveLoss(nn.Module):
    """
    Loss function for Causal Disentanglement.
    Includes:
    1. MVC Loss (on Content S)
    2. Invariance Loss (S vs S_cf)
    3. Reconstruction Loss (Z vs Z_rec)
    """
    def __init__(self, temperature=0.5):
        super(CausalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.mse = nn.MSELoss()
        
    def forward(self, view_features, s_list, s_cf_list, z_rec_list):
        if s_list is None:
            return torch.tensor(0.0).to(view_features[0].device)
            
        device = view_features[0].device
        num_views = len(view_features)
        
        # 1. Reconstruction Loss: Ensure Disentangler preserves information
        loss_rec = 0.0
        for v in range(num_views):
            loss_rec += self.mse(view_features[v], z_rec_list[v])
        loss_rec /= num_views
        
        # 2. Invariance Loss: Content should be stable against Style Shuffle
        # L_inv = || s - s_cf ||^2
        loss_inv = 0.0
        if s_cf_list is not None:
            for v in range(num_views):
                loss_inv += self.mse(s_list[v], s_cf_list[v])
            loss_inv /= num_views
        
        # 3. Content Contrastive Loss (Cross-View Consistency)
        # We want S_v1 to be similar to S_v2 (Alignment)
        loss_mvc = 0.0
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # Standard Contrastive on S
                # Normalize
                z_i = F.normalize(s_list[i], dim=1)
                z_j = F.normalize(s_list[j], dim=1)
                
                sim = torch.matmul(z_i, z_j.T) / self.temperature
                
                # NT-Xent simplified or just use diagonal alignment
                # SCMVC uses a specific weighted contrastive, here we use standard infoNCE logic
                # or match the implementation of 'ContrastiveLoss' in loss.py
                # Let's use a simple alignment for the 'causal' part
                
                # Positive pairs: Diagonals
                labels = torch.arange(z_i.size(0)).to(device)
                l1 = F.cross_entropy(sim, labels)
                l2 = F.cross_entropy(sim.T, labels)
                loss_mvc += (l1 + l2) / 2
                
        loss_mvc /= (num_views * (num_views - 1) / 2)

        # Total Causal Loss strategy - Dataset Adaptive Weights
        # Three components balance different objectives:
        # - loss_rec: Reconstruction quality (info preservation)
        # - loss_inv: Invariance strength (robustness vs discriminability trade-off)
        # - loss_mvc: Cross-view consistency (multi-view fusion)
        
        # Recommended configurations by dataset type:
        # Type 1 - Structured Multi-View (MSRC-v1, Animal, NUS-WIDE):
        #   Focus on cross-view consistency and moderate invariance
        #   0.3 * rec + 0.5 * inv + 0.6 * mvc
        
        # Type 2 - Scene/Background-Heavy (OutdoorScene, Caltech):  
        #   Need strong invariance to filter backgrounds
        #   0.2 * rec + 0.8 * inv + 0.3 * mvc
        
        # Type 3 - Small Sample (MSRC-v1 when N<500):
        #   Emphasize reconstruction to avoid overfitting
        #   0.6 * rec + 0.2 * inv + 0.4 * mvc
        
        # Default: Balanced configuration for general multi-view datasets
        # We restore these to stable values because extreme weights (like 1.0 for inv)
        # cause catastrophic forgetting of clusters during the CF phase.
        alpha_rec = 0.3  # Reconstruction weight
        beta_inv = 0.5   # Invariance weight  
        gamma_mvc = 0.5  # Cross-view consistency weight
        
        return alpha_rec * loss_rec + beta_inv * loss_inv + gamma_mvc * loss_mvc

# Legacy class for compatibility if needed, but we don't use it
class ViewInvarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.tensor(0.0)

