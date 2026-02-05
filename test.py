import argparse
import os
import torch
import numpy as np

from dataloader import load_data
from metric import valid
from network import Network

# Define the 8 target datasets
TARGET_DATASETS = [
    "ALOI",
    "Animal",
    "OutdoorScene",
    "COIL20",
    "Reuters-1200",
    "Yale",
    "EYaleB",
    "ORL"
]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_base_params(dataset_name):
    # Retrieve base parameters consistent with train.py
    params = {"feature_dim": 64, "high_feature_dim": 20}
    # Add any dataset-specific overrides if they exist in train.py (most seem to use 64/20)
    # Animal uses different batch sizes/LR but dims seem default unless specified otherwise.
    # Looking at train.py, feature_dim is an argparse with default 64.
    return params

def evaluate_dataset(name: str, args, device) -> None:
    print(f"\n>>> Processing Dataset: {name}")
    try:
        dataset, dims, view, data_size, class_num = load_data(name)
    except Exception as e:
        print(f"Error loading data for {name}: {e}")
        return

    params = get_base_params(name)
    # Initialize Network (must match train.py structure)
    # Note: Use use_crm=True as we want the full model's inference capability
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device, use_crm=True)
    model = model.to(device)
    
    ckpt_path = os.path.join(args.models_dir, f'{name}.pth')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint missing for {name}: {ckpt_path}")
        return
    
    try:
        # Load weights
        # We assume the model was trained and saved as 'model.state_dict()'
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint for {name}: {e}")
        return

    model.eval()
    print(f"Dataset: {name}")
    print(f"Datasize: {data_size}")
    print("Loading model and evaluating...")
    
    # Use valid() from metric.py as requested to get detailed output
    # eval_h=True will print low-level and consensus clustering results
    acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=True)
    print(f"Final Result: ACC = {acc:.4f} NMI = {nmi:.4f} PUR = {pur:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CoDe-MVC Test')
    # Default parameters
    parser.add_argument("--feature_dim", default=64, type=int)
    parser.add_argument("--high_feature_dim", default=20, type=int)
    parser.add_argument("--models_dir", default="./models", help="Directory where .pth models are saved")
    
    # No --run_target_set needed, we run all by default.
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(10)
    
    print(f"Running evaluation on all 8 target datasets: {TARGET_DATASETS}")
    for target_dataset in TARGET_DATASETS:
        evaluate_dataset(target_dataset, args, device)
