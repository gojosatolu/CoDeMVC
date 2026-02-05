import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import argparse
import random
from loss import ContrastiveLoss
from dataloader import load_data
from causal_module import CausalDebiasedMultiViewClustering, CausalContrastiveLoss, ViewInvarianceLoss
import matplotlib.pyplot as plt

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# Cifar10
# Cifar100
# Prokaryotic
# Synthetic3d
Dataname = 'MNIST-USPS'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=0., type=float)
parser.add_argument("--pre_epochs", default=200, type=int)
parser.add_argument("--con_epochs", default=50, type=int)
parser.add_argument("--feature_dim", default=64, type=int)
parser.add_argument("--high_feature_dim", default=20, type=int)
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--causal_weight", default=0.5, type=float, help="Overall weight for causal structure (Legacy, kept for backward compatibility)")
parser.add_argument("--alpha", default=1.0, type=float, help="Weight for Disentanglement loss (Lrec + Lalign)")
parser.add_argument("--beta", default=1.0, type=float, help="Weight for Invariance loss (Linv)")
parser.add_argument("--no_counterfactual", action="store_true", help="Disable counterfactual augmentation")
parser.add_argument("--no_crm", action="store_true", help="Disable Causal Reweighting Module (CRM)")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "MNIST-USPS":
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 10 # 20
    seed = 30
if args.dataset == "CCV":
    args.con_epochs = 50 # 100
    seed = 100
    args.tune_epochs = 200
if args.dataset == "Fashion":
    args.con_epochs = 50 # 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 100
    seed = 200
    args.tune_epochs = 200
if args.dataset == "Caltech-3V":
    args.con_epochs = 100
    seed = 30
if args.dataset == "Caltech-4V":
    args.con_epochs = 100
    seed = 100
if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 1000000
if args.dataset == "Cifar10":
    args.con_epochs = 10
    seed = 10
if args.dataset == "Cifar100":
    args.con_epochs = 20
    seed = 10
if args.dataset == "Prokaryotic":
    args.con_epochs = 20
    seed = 10000
if args.dataset == "Synthetic3d":
    args.con_epochs = 100
    seed = 100

# User added: Individual configuration for new datasets
if args.dataset == "COIL20":
    # Small dataset
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "ALOI":
    # Medium dataset
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "OutdoorScene":
    # Small/Medium dataset
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 50
    seed = 10

if args.dataset == "YoutubeFace":
    # Large dataset (100k+)
    args.batch_size = 1024
    args.learning_rate = 1e-3
    args.con_epochs = 100
    seed = 10

if args.dataset == "Caltech101":
    # Medium dataset
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "NoisyBDGP":
    # Small but noisy
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 50
    seed = 10000

if args.dataset == "NoisyHW":
    # Small but noisy
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 200
    seed = 100000

if args.dataset == "TinyImage":
    # Large dataset (100k)
    args.batch_size = 2048
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "STL10":
    # Medium/Large (13k)
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "NoisyMNIST":
    # User specified settings
    args.batch_size = 256
    args.learning_rate = 5e-4
    args.con_epochs = 150
    seed = 13

if args.dataset == "MSRC-v1":
    # Small dataset (210 samples)
    # High epochs needed for convergence on small data
    args.batch_size = 64 # Small batch for small data
    args.learning_rate = 1e-3
    args.con_epochs = 200
    seed = 10

if args.dataset == "NUS-WIDE":
    # Medium dataset (2400 samples)
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "Caltech101-20":
    # Medium dataset (2386 samples)
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "Animal":
    # Large dataset (11673 samples, 20 classes, 4 views)
    # High dimensional features (2000-2689 dims per view)
    args.batch_size = 512  # Large batch for large dataset
    args.learning_rate = 1e-3
    args.con_epochs = 50  # Fewer epochs needed for large data
    seed = 10

if args.dataset == "Reuters-1200":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "BBCSport":
    args.batch_size = 128
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 42

if args.dataset == "LandUse-21":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "Scene-15":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "WebKB":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "STL10_4V":
    args.batch_size = 256
    args.learning_rate = 3e-3
    args.con_epochs = 100
    seed = 10

if args.dataset == "Yale":
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.con_epochs = 200
    seed = 10

if args.dataset == "100Leaves":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "Handwritten":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 200
    seed = 10

if args.dataset == "3Sources":
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.con_epochs = 200
    seed = 10

if args.dataset == "MNIST-10k":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "MSRC-v5":
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.con_epochs = 200
    seed = 10

if args.dataset == "Reuters-1500":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "UCI":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "NUS-WIDE-OBJ":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 50
    seed = 10

if args.dataset == "ORL":
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.con_epochs = 200
    seed = 10

if args.dataset == "EYaleB":
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.con_epochs = 100 # Slightly larger than Yale, fewer epochs needed?
    seed = 10

if args.dataset == "FashionMNIST":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "NottingHill":
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.con_epochs = 200
    seed = 10

if args.dataset == "YaleB":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 50
    seed = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def compute_view_value(rs, H, view):
    N = H.shape[0]
    w = []
    # all features are normalized
    global_sim = torch.matmul(H,H.t())
    for v in range(view):
        view_sim = torch.matmul(rs[v],rs[v].t())
        related_sim = torch.matmul(rs[v],H.t())
        # The implementation of MMD
        w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N*N)
        w.append(torch.exp(-w_v))
    w = torch.stack(w)
    w = w / torch.sum(w)
    return w.squeeze()


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        
        # Encode first
        zs = []
        for v in range(view):
            zs.append(model.encoders[v](xs[v]))
        
        # Apply causal module - no counterfactuals during pretrain for stability
        c_list, c_cf_list, z_rec_list, z_cf_list = causal_module(zs, return_counterfactuals=False)
        
        # Decode (reconstruction)
        xrs = []
        for v in range(view):
            xrs.append(model.decoders[v](zs[v]))
        
        # Reconstruction loss
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        
        # Causal losses (Reconstruction of Z + Content Consistency)
        l_rec, l_inv, l_align = causal_contrastive_criterion(zs, c_list, c_cf_list, z_rec_list)
        # Pretrain only uses disentanglement (reconstruction), inv is 0 usually if no CF.
        causal_loss_term = args.alpha * (l_rec + l_align) + args.beta * l_inv
        
        loss = sum(loss_list) + causal_loss_term
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    avg_loss = tot_loss/len(data_loader)
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(avg_loss))
    return avg_loss

def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    
    # Causal Warm-up: Enable counterfactuals after 40% of con_epochs (more stable)
    # Using 70% was too late, leaving insufficient time for the model to adapt.
    warmup_threshold = args.pre_epochs + int(0.4 * args.con_epochs)
    use_cf = (not args.no_counterfactual) and (epoch > warmup_threshold)
    
    if use_cf and epoch == warmup_threshold + 1:
        print(">>> [Causal Warm-up Finished] Style-shuffled Counterfactuals Enabled.")

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        
        # Encode features
        zs = []
        for v in range(view):
            # Only apply CRM during formal training (when counterfactuals are used)
            zs.append(model.encoders[v](xs[v], apply_crm=use_cf))
        
        # Apply causal module with dynamic counterfactual generation
        c_list, c_cf_list, z_rec_list, z_cf_list = causal_module(zs, return_counterfactuals=use_cf)
        
        # View-consensus features and global fusion
        rs = []
        for v in range(view):
            rs.append(F.normalize(model.common_information_module(zs[v]), dim=1))
        
        H = model.feature_fusion(zs, zs_gradient=True)
        
        # Input Reconstruction
        xrs = []
        for v in range(view):
            xrs.append(model.decoders[v](zs[v]))
        
        loss_list = []
        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        for v in range(view):
            loss_list.append(contrastiveloss(H, rs[v], w[v]))
            loss_list.append(mse(xs[v], xrs[v]))
        
        # Causal losses (including Invariance Loss if c_cf_list is not None)
        # causal_contrastive_criterion now returns (loss_rec, loss_inv, loss_align)
        l_rec, l_inv, l_align = causal_contrastive_criterion(zs, c_list, c_cf_list, z_rec_list)
        
        # New weighted Sum:
        # l_rec + l_align: Disentanglement quality -> Controlled by alpha
        # l_inv: Robustness to style changes -> Controlled by beta
        # We still keep args.causal_weight as a global toggle or legacy multiplier if needed, 
        # but user request is to use alpha and beta strictly. 
        # Assuming causal_weight might be passed as 1.0 or ignored in new script.
        # But to be safe and clean, we use alpha/beta directly here.
        
        causal_loss_term = args.alpha * (l_rec + l_align) + args.beta * l_inv
        
        loss = sum(loss_list) + causal_loss_term
        
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    avg_loss = tot_loss/len(data_loader)
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(avg_loss))
    return avg_loss


accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device, use_crm=not args.no_crm)
    # model = model.to(device)
    # Initialize NEW causal module (Causal Contrastive Learning)
    causal_module = CausalDebiasedMultiViewClustering(
        num_views=view, 
        feature_dim=args.feature_dim,  # Operates on encoded features
        device=device
    )
    print(model)
    model = model.to(device)
    causal_module = causal_module.to(device)
    
    state = model.state_dict()
    # Optimize both model and causal module
    optimizer = torch.optim.Adam(list(model.parameters()) + list(causal_module.parameters()), 
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)
    
    # Initialize causal criteria
    causal_contrastive_criterion = CausalContrastiveLoss(temperature=args.temperature).to(device)
    best_acc, best_nmi, best_pur = 0, 0, 0
    loss_history = []

    epoch = 1
    while epoch <= args.pre_epochs:
        loss = pretrain(epoch)
        loss_history.append(loss)
        epoch += 1
    # acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=True, epoch=epoch)

    while epoch <= args.pre_epochs + args.con_epochs:
        loss = contrastive_train(epoch)
        loss_history.append(loss)
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            torch.save(causal_module.state_dict(), './models/' + args.dataset + '_causal.pth')
        epoch += 1

    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Total Loss', color='#e74c3c', linewidth=2)
    # Mark phases
    plt.axvline(x=args.pre_epochs, color='gray', linestyle='--', label='End of Pretrain')
    warmup_pt = args.pre_epochs + int(0.4 * args.con_epochs)
    plt.axvline(x=warmup_pt, color='blue', linestyle=':', label='Causal Mechanism Enabled')
    
    plt.title(f'Training Convergence: {args.dataset}', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if not os.path.exists('figures_analysis'): os.makedirs('figures_analysis')
    plt.savefig(f'figures_analysis/{args.dataset}_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> Loss curve saved to figures_analysis/{args.dataset}_loss_curve.png")

    # The final result
    accs.append(best_acc)
    nmis.append(best_nmi)
    purs.append(best_pur)
    print('The best clustering performace: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_acc, best_nmi, best_pur))
