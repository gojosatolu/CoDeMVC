import torch.nn as nn
from torch.nn.functional import normalize
import torch

# Causal Attention Proxy (replaces CRM)
class FeatureCausalAttention(nn.Module):
    def __init__(self, feature_dim, learnable_epsilon=False):
        super(FeatureCausalAttention, self).__init__()
        # Utilize learnable parameters to build proxy graphs of feature dependencies
        self.Q = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.K = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.scale = feature_dim ** 0.5
        
        self.learnable_epsilon = learnable_epsilon
        if self.learnable_epsilon:
            self.epsilon = nn.Parameter(torch.tensor(0.1))
        else:
            self.epsilon = 0.1

    def forward(self, x):
        # x: [Batch, feature_dim]
        # Attention Matrix A: [feature_dim, feature_dim]
        A = torch.nn.functional.softmax(torch.matmul(self.Q, self.K.t()) / self.scale, dim=-1)
        
        # Message passing: Purify features based on causal relations
        # Add residual connection to prevent initialization shock
        z_causal = x + self.epsilon * torch.matmul(x, A)
        
        return z_causal, A

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, use_crm=True, learnable_epsilon=False):
        super(Encoder, self).__init__()
        self.use_crm = use_crm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )
        if self.use_crm:
            self.causal_attention = FeatureCausalAttention(feature_dim, learnable_epsilon=learnable_epsilon)

    def forward(self, x, apply_crm=True):
        z = self.encoder(x)
        if self.use_crm and apply_crm:
            z, A = self.causal_attention(z)
        else:
            # Phase 1 / Identity phase: Features do not interact causally
            A = torch.eye(z.size(1)).to(z.device)
        return z, A

# Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# SCMVC Network
class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, device, use_crm=True, learnable_epsilon=False):
        super(Network, self).__init__()
        self.view = view
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim, use_crm=use_crm, learnable_epsilon=learnable_epsilon).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))

        # global features fusion layer
        self.feature_fusion_module = nn.Sequential(
            nn.Linear(self.view * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        # view-consensus features learning layer
        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )

    # global feature fusion
    def feature_fusion(self, zs, zs_gradient):
        input = torch.cat(zs, dim=1) if zs_gradient else torch.cat(zs, dim=1).detach()
        return normalize(self.feature_fusion_module(input),dim=1)

    def forward(self, xs, zs_gradient=True):
        rs = []
        xrs = []
        zs = []
        A_list = []
        for v in range(self.view):
            x = xs[v]
            z, A = self.encoders[v](x)
            xr = self.decoders[v](z)
            r = normalize(self.common_information_module(z),dim=1)

            rs.append(r)
            zs.append(z)
            xrs.append(xr)
            A_list.append(A)

        H = self.feature_fusion(zs,zs_gradient)
        return xrs,zs,rs,H,A_list
