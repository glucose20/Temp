import os
import random
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from LLMDTA import LLMDTA as LLMDTA
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, batch2tensor, my_collate_fn

from sklearn.metrics import r2_score
from tqdm import tqdm
from math import sqrt
from scipy import stats
import csv
import wandb


def cindex_score(y, p):
    sum_m = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    sum_m += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    if pair != 0:
        return sum_m / pair
    else:
        return 0
    
def regression_scores(label, pred, is_valid=True):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    mse = ((label - pred)**2).mean(axis=0)
    rmse = sqrt(mse)
    if is_valid:
        ci = -1
    else:
        ci = cindex_score(label, pred)
    r2 = r2_score(label, pred)
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(mse, 6), round(rmse, 6), round(ci, 6), round(r2, 6), round(pearson, 6), round(spearman, 6)

def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)
    
def test(model, dataloader, is_valid=True):
    model.eval()
    preds = []
    labels = []
    for batch_i, batch_data in enumerate(dataloader):
        mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask, affinity = batch_data
        with torch.no_grad():
            # Handle both DataParallel and regular model
            if hasattr(model, 'module'):
                pred = model.module.forward(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=False)
            else:
                pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist()

    preds = np.array(preds)
    labels = np.array(labels)
    mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(labels, preds, is_valid)
    return mse_value, rmse_value, ci, r2, pearson_value, spearman_value


def train_one_epoch(model, train_loader, optimizer, criterion, hp):
    """Train model for one epoch and return metrics."""
    model.train()
    preds = []
    labels = []
    total_loss = 0.0
    
    for batch_data in train_loader:
        mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data
        
        if hasattr(model, 'module'):
            predictions, gate_info = model.module.forward(
                mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True
            )
        else:
            predictions, gate_info = model(
                mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True
            )
        
        preds += predictions.cpu().detach().numpy().reshape(-1).tolist()
        labels += affinity.cpu().detach().numpy().reshape(-1).tolist()
        
        loss = criterion(predictions.squeeze(), affinity)
        
        # Add load balance loss if applicable
        if hasattr(model, 'module') and hasattr(model.module, 'compute_load_balance_loss'):
            lb_loss = model.module.compute_load_balance_loss(gate_info['gate_weights'])
            loss = loss + hp.load_balance_weight * lb_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    
    preds = np.array(preds)
    labels = np.array(labels)
    mse, rmse, ci, r2, pearson, spearman = regression_scores(preds, labels)
    
    return {'mse': mse, 'rmse': rmse, 'ci': ci, 'r2': r2, 'pearson': pearson, 'spearman': spearman}


def auto_select_moe_config(hp, device, train_loader, valid_loader, test_loader, criterion, selection_epochs=3):
    """
    Auto-select best MoE configuration by training multiple epochs with different configs
    and selecting the one with best CI improvement over baseline.
    
    Workflow:
    - Selection phase: Train each config for `selection_epochs` epochs
      - Use VALID set during training for adjustments (like normal)
      - After all epochs done, use TEST set ONCE to get CI for selection
    - Training phase: Train with selected config, use VALID set for early stopping
    - Final: Evaluate on TEST set for final results
    
    Note: TEST set is only used twice - once for config selection, once for final results.
    
    Args:
        selection_epochs: Number of epochs to train each config before evaluation (default: 3)
    
    Returns: (best_num_experts, best_top_k, best_ci)
    """
    import time
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print(f"🔍 AUTO MOE SELECTION - Testing configs for {selection_epochs} epochs each...")
    print("=" * 70)
    
    # Define configs to test: (num_experts, top_k, description)
    configs = [
        (1, 1, "Baseline (no MoE)"),
        (4, 1, "MoE 4 experts, top-1 (sparse)"),
        (4, 2, "MoE 4 experts, top-2"),
        (6, 2, "MoE 6 experts, top-2"),
        (8, 2, "MoE 8 experts, top-2"),
    ]
    
    results = []
    baseline_ci = None
    
    # Save original hp values
    original_num_experts = hp.num_experts
    original_top_k = hp.top_k
    
    for num_exp, top_k, desc in configs:
        print(f"\n--- Testing: {desc} ---")
        
        # Update hp temporarily
        hp.num_experts = num_exp
        hp.top_k = top_k
        
        # Create fresh model
        model = nn.DataParallel(LLMDTA(hp, device))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
        
        # Train for selection_epochs, using VALID set for monitoring (like normal training)
        for ep in range(1, selection_epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, hp)
            # Use VALID set during training (like normal)
            val_mse, val_rmse, val_ci, _, _, _ = test(model, valid_loader, is_valid=True)
            print(f"    Epoch {ep}/{selection_epochs}: Train MSE={train_metrics['mse']:.4f}, Valid MSE={val_mse:.4f}, Valid CI={val_ci:.4f}")
        
        # After all selection epochs, use TEST set ONCE to evaluate for config selection
        mse, rmse, ci, r2, pearson, spearman = test(model, test_loader, is_valid=False)
        
        print(f"    ✓ Selection eval (TEST): CI={ci:.4f}, MSE={mse:.4f}")
        
        # Store baseline CI
        if num_exp == 1 and top_k == 1:
            baseline_ci = ci
        
        results.append({
            'num_experts': num_exp,
            'top_k': top_k,
            'desc': desc,
            'test_ci': ci,
            'test_mse': mse,
            'train_mse': train_metrics['mse']
        })
        
        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Calculate CI improvement and select best
    print("\n" + "=" * 70)
    print("📊 AUTO MOE SELECTION RESULTS (on TEST set):")
    print("=" * 70)
    print(f"{'Config':<30} {'Test CI':>10} {'CI Improve':>12} {'Test MSE':>10}")
    print("-" * 70)
    
    best_config = None
    best_ci_improve = 0
    
    for r in results:
        # Handle edge case where baseline_ci is 0 or None
        if baseline_ci and baseline_ci > 0:
            ci_improve = ((r['test_ci'] - baseline_ci) / baseline_ci * 100)
        else:
            ci_improve = 0
        r['ci_improve'] = ci_improve
        
        emoji = "✅" if ci_improve > 0 else "❌" if ci_improve < 0 else "➖"
        print(f"{emoji} {r['desc']:<28} {r['test_ci']:>10.4f} {ci_improve:>+11.2f}% {r['test_mse']:>10.4f}")
        
        # Select config with highest CI improvement (must be positive)
        if ci_improve > best_ci_improve:
            best_ci_improve = ci_improve
            best_config = r
    
    # If no improvement, use baseline
    if best_config is None or best_ci_improve <= 0:
        best_config = results[0]  # Baseline
        print(f"\n⚠️  No MoE config improved CI. Using Baseline (num_experts=1, top_k=1)")
    else:
        print(f"\n🏆 SELECTED: {best_config['desc']} (CI improve: {best_ci_improve:+.2f}%)")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Auto selection took {elapsed:.1f}s")
    print("=" * 70 + "\n")
    
    return best_config['num_experts'], best_config['top_k'], best_config['test_ci']

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LLMDTA model for a specific fold')
    parser.add_argument('--fold', type=int, required=True, 
                        help='Fold index to train (0-4 for 5-fold CV)')
    parser.add_argument('--cuda', type=str, default=None,
                        help='CUDA device ID (e.g., "0", "1"). Overrides hyperparameter.py setting')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name to override hyperparameter.py setting')
    parser.add_argument('--running_set', type=str, default=None,
                        help='Running set to override hyperparameter.py setting')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides hyperparameter.py setting)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size to use (overrides hyperparameter.py setting)')
    parser.add_argument('--wandb_project', type=str, default='LLMDTA',
                        help='Weights & Biases project name (default: LLMDTA)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/username (optional)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--use_esmc', type=lambda x: x.lower() == 'true', default=None,
                        help='Use ESM-C (True) or ESM2 (False). Overrides hyperparameter.py setting')
    parser.add_argument('--esmc_model', type=str, default=None, choices=['esmc_300m', 'esmc_600m', 'esmc_6b'],
                        help='ESM-C model variant (esmc_300m, esmc_600m, esmc_6b). Overrides hyperparameter.py setting')
    # MoE parameters
    parser.add_argument('--num_experts', type=int, default=None,
                        help='Number of experts in MoE (default: 4)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Number of experts to select per sample (default: 2)')
    parser.add_argument('--moe_noise_std', type=float, default=None,
                        help='Noise std for MoE exploration (default: 0.1, 0 to disable)')
    parser.add_argument('--load_balance_weight', type=float, default=None,
                        help='Weight for load balancing loss (default: 0.01, 0 to disable)')
    parser.add_argument('--auto_moe', action='store_true',
                        help='Auto-select best MoE config in first epoch based on CI improvement')
    args = parser.parse_args()
    
    fold_i = args.fold

    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    
    hp = HyperParameter()
    
    # Override ESM settings BEFORE dataset (important for path resolution)
    if args.use_esmc is not None:
        hp.use_esmc = args.use_esmc
        # Update dimension when switching between ESM2 and ESM-C
        if not hp.use_esmc:
            hp.protvec_dim = 1280  # ESM2
    
    if args.esmc_model is not None:
        hp.esmc_model = args.esmc_model
        # Update dimension based on model
        if hp.esmc_model == "esmc_300m":
            hp.protvec_dim = 960
        elif hp.esmc_model == "esmc_600m":
            hp.protvec_dim = 1152
        elif hp.esmc_model == "esmc_6b":
            hp.protvec_dim = 2560
    
    # Override CUDA device if specified
    if args.cuda is not None:
        hp.cuda = args.cuda
    
    # Override dataset if specified (this will update paths based on use_esmc)
    if args.dataset is not None:
        hp.set_dataset(args.dataset)
    else:
        # If dataset not overridden but ESM settings changed, update paths for current dataset
        if args.use_esmc is not None or args.esmc_model is not None:
            hp.set_dataset(hp.dataset)
    
    if args.running_set is not None:
        # Convert underscores to hyphens (data directories use hyphens)
        hp.running_set = args.running_set.replace('_', '-')
    # Override epochs if specified
    if args.epochs is not None:
        hp.Epoch = args.epochs
    # Override batch size if specified
    if args.batch_size is not None:
        hp.Batch_size = args.batch_size
    # Override MoE parameters if specified
    if args.num_experts is not None:
        hp.num_experts = args.num_experts
    if args.top_k is not None:
        hp.top_k = args.top_k
    if args.moe_noise_std is not None:
        hp.moe_noise_std = args.moe_noise_std
    if args.load_balance_weight is not None:
        hp.load_balance_weight = args.load_balance_weight

    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    print(f"=" * 60)
    print(f"Training Fold {fold_i}/{hp.kfold-1}")
    print(f"Dataset: {hp.dataset}-{hp.running_set}") 
    print(f"ESM Model: {'ESM-C-' + hp.esmc_model if hp.use_esmc else 'ESM2'} (dim={hp.protvec_dim})")
    print(f"MoE: num_experts={hp.num_experts}, top_k={hp.top_k}, noise={hp.moe_noise_std}, lb_weight={hp.load_balance_weight}")
    print(f"Device: {device} (CUDA_VISIBLE_DEVICES={hp.cuda})")
    print(f"Pretrain-{hp.mol2vec_dir}")
    print(f"Pretrain-{hp.protvec_dir}")
    print(f"=" * 60)
    
    # Initialize Weights & Biases
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_config = {
            'dataset': hp.dataset,
            'running_set': hp.running_set,
            'fold': fold_i,
            'epochs': hp.Epoch,
            'batch_size': hp.Batch_size,
            'learning_rate': hp.Learning_rate,
            'max_patience': hp.max_patience,
            'cuda_device': hp.cuda,
            'use_esmc': hp.use_esmc,
            'esmc_model': hp.esmc_model if hp.use_esmc else None,
            'protvec_dim': hp.protvec_dim,
            # MoE parameters (initial values, may be auto-selected)
            'num_experts': hp.num_experts,
            'top_k': hp.top_k,
            'moe_noise_std': hp.moe_noise_std,
            'load_balance_weight': hp.load_balance_weight,
            'auto_moe': args.auto_moe,
        }
        
        esm_name = f"esmc-{hp.esmc_model}" if hp.use_esmc else "esm2"
        run_name = f"{hp.dataset}-{hp.running_set}-{esm_name}-fold{fold_i}"
        if args.auto_moe:
            run_name += "-autoMoE"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=wandb_config,
            tags=[hp.dataset, hp.running_set, esm_name, f'fold{fold_i}'],
            reinit=True
        )
        print(f"Weights & Biases initialized: {args.wandb_project}")
    else:
        print("Weights & Biases logging disabled")
    
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    
    # Validate fold index
    if fold_i < 0 or fold_i >= hp.kfold:
        raise ValueError(f"Fold index must be between 0 and {hp.kfold-1}, got {fold_i}")
    
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    
    # Load data for the specified fold
    train_dir = os.path.join(dataset_root, f'fold_{fold_i}_train.csv')
    valid_dir = os.path.join(dataset_root, f'fold_{fold_i}_valid.csv')
    test_dir = os.path.join(dataset_root, f'fold_{fold_i}_test.csv')
    
    print(f"Loading fold {fold_i} data...")
    print(f"  Train: {train_dir}")
    print(f"  Valid: {valid_dir}")
    print(f"  Test:  {test_dir}")
    
    train_set = CustomDataSet(pd.read_csv(train_dir, sep=','), hp)
    valid_set = CustomDataSet(pd.read_csv(valid_dir, sep=','), hp)
    test_set = CustomDataSet(pd.read_csv(test_dir, sep=','), hp)
    train_dataset_load = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    valid_dataset_load = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    test_dataset_load = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    print(f"Dataset loaded: {len(train_set)} train, {len(valid_set)} valid, {len(test_set)} test samples")

    criterion = F.mse_loss
    
    # ============================================================
    # AUTO MOE SELECTION: Test different configs and select best
    # ============================================================
    selection_epochs = 3  # Number of epochs to train each config during selection
    if args.auto_moe and hp.Epoch > selection_epochs:
        print(f"\n🔄 Auto MoE Selection enabled - will test configs for {selection_epochs} epochs each...")
        print(f"   (Using VALID set during training, TEST set only for final selection)")
        best_num_experts, best_top_k, _ = auto_select_moe_config(
            hp, device, train_dataset_load, valid_dataset_load, test_dataset_load, criterion, selection_epochs
        )
        
        # Update hp with selected config
        hp.num_experts = best_num_experts
        hp.top_k = best_top_k
        
        # Log selected config to wandb
        if use_wandb:
            wandb.config.update({
                'auto_moe_selected_num_experts': best_num_experts,
                'auto_moe_selected_top_k': best_top_k,
            })
        
        print(f"✅ Selected MoE config: num_experts={hp.num_experts}, top_k={hp.top_k}")
        print(f"   Continuing training for {hp.Epoch} epochs with selected config...\n")
    elif args.auto_moe and hp.Epoch <= selection_epochs:
        print(f"\n⚠️  Warning: --auto_moe enabled but epochs ({hp.Epoch}) <= selection_epochs ({selection_epochs})")
        print(f"   Auto MoE selection skipped. Using default config: num_experts={hp.num_experts}, top_k={hp.top_k}\n")
    
    # Create model with (possibly updated) MoE config
    model = nn.DataParallel(LLMDTA(hp, device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))

    train_log = []     
    best_valid_mse = float('inf')  # Initialize with infinity instead of 10
    patience = 0    
    
    # Use consistent timestamp for all files
    timestamp = hp.current_time
    model_fromTrain = f'./savemodel/{hp.dataset}-{hp.running_set}-fold{fold_i}-{timestamp}.pth'
    
    # Create savemodel directory if not exists
    os.makedirs('./savemodel', exist_ok=True)
    
    print(f"Model will be saved to: {model_fromTrain}")
             
    for epoch in range(1, hp.Epoch + 1):    
        # Reset MoE usage stats at start of each epoch
        if hasattr(model.module, 'reset_usage_stats'):
            model.module.reset_usage_stats()
        
        # trainning
        model.train()
        pred = []
        label = []
        total_load_balance_loss = 0.0
        num_batches = 0
        for batch_data in train_dataset_load:
            mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask, affinity = batch_data                    
            predictions, gate_info = model.module.forward(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True)
            pred = pred + predictions.cpu().detach().numpy().reshape(-1).tolist()
            label = label + affinity.cpu().detach().numpy().reshape(-1).tolist()            
            
            # Compute main loss
            loss = criterion(predictions.squeeze(), affinity)
            
            # Optional: Add load balancing loss to encourage diverse expert usage
            load_balance_loss = model.module.compute_load_balance_loss(gate_info['gate_weights'])
            total_load_balance_loss += load_balance_loss.item()
            num_batches += 1
            
            # Combine losses (use hp.load_balance_weight, set to 0 to disable)
            total_loss = loss + hp.load_balance_weight * load_balance_loss
            
            total_loss.backward()                
            optimizer.step()
            optimizer.zero_grad()                                             
        pred = np.array(pred)
        label= np.array(label)
        mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(pred, label)
        train_log.append([mse_value, rmse_value, ci, r2, pearson_value, spearman_value])
        
        # Get MoE expert usage statistics
        moe_stats = model.module.get_expert_usage_stats() if hasattr(model.module, 'get_expert_usage_stats') else None
        avg_load_balance_loss = total_load_balance_loss / num_batches if num_batches > 0 else 0
        
        print(f'Train at fold-{fold_i} epoch-{epoch}: mse={mse_value:.4f}, rmse={rmse_value:.4f}, r2={r2:.4f}')
        if moe_stats:
            print(f'  MoE Stats: usage_rate={moe_stats["expert_usage_rate"]}, entropy={moe_stats["usage_entropy"]:.4f}, dominant_expert={moe_stats["dominant_expert"]}, load_balance_loss={avg_load_balance_loss:.4f}')
        
        # Adaptive MoE parameter adjustment
        if hasattr(model.module, 'adaptive_update'):
            adjustments = model.module.adaptive_update(epoch, hp.Epoch, moe_stats)
            # Update hp.load_balance_weight for next epoch
            if 'load_balance_weight' in adjustments:
                hp.load_balance_weight = adjustments['load_balance_weight']
            print(f'  MoE Adaptive: noise={adjustments.get("noise_std", "N/A"):.4f}, lb_weight={adjustments.get("load_balance_weight", "N/A"):.4f}, lb_adj={adjustments.get("lb_adjustment", "N/A")}')
        
        # Log training metrics to wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/mse': mse_value,
                'train/rmse': rmse_value,
                'train/ci': ci,
                'train/r2': r2,
                'train/pearson': pearson_value,
                'train/spearman': spearman_value,
            }
            # Add MoE statistics
            if moe_stats:
                log_dict['moe/load_balance_loss'] = avg_load_balance_loss
                log_dict['moe/usage_entropy'] = moe_stats['usage_entropy']
                log_dict['moe/usage_std'] = moe_stats['usage_std']
                log_dict['moe/dominant_expert'] = moe_stats['dominant_expert']
                for i, rate in enumerate(moe_stats['expert_usage_rate']):
                    log_dict[f'moe/expert_{i}_usage'] = rate
            # Add adaptive parameters
            if hasattr(model.module, 'adaptive_update'):
                log_dict['moe/adaptive_noise_std'] = adjustments.get('noise_std', 0)
                log_dict['moe/adaptive_lb_weight'] = adjustments.get('load_balance_weight', 0)
            wandb.log(log_dict)
        
        # valid
        mse, rmse, ci, r2, pearson, spearman = test(model, valid_dataset_load, is_valid=True)   
        print(f'Valid at fold-{fold_i} epoch-{epoch}: mse={mse:.4f}, rmse={rmse:.4f}, r2={r2:.4f}')
        
        # Log validation metrics to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'valid/mse': mse,
                'valid/rmse': rmse,
                'valid/r2': r2,
                'valid/pearson': pearson,
                'valid/spearman': spearman,
            })
        
        # Early stop        
        if mse < best_valid_mse :
            patience = 0
            best_valid_mse = mse
            # save model
            torch.save(model.state_dict(), model_fromTrain)
            print(f'New best! Valid at fold-{fold_i} epoch-{epoch}: mse={mse:.4f}, rmse={rmse:.4f}, r2={r2:.4f}, pearson={pearson:.4f}, spearman={spearman:.4f}')
            
            # Log best validation metrics to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'best_valid/mse': mse,
                    'best_valid/rmse': rmse,
                    'best_valid/r2': r2,
                    'best_valid/pearson': pearson,
                    'best_valid/spearman': spearman,
                })
        else:
            patience += 1
            if patience > hp.max_patience:
                print(f'Early stopping at epoch-{epoch}, patience={patience}')
                print(f'Best model saved at: {model_fromTrain}')
                break   
             
    log_dir = f"./log/{timestamp}-{hp.dataset}-{hp.running_set}-fold{fold_i}.csv"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    with open(log_dir, "w+")as f:
        writer = csv.writer(f)
        writer.writerow(["mse", "rmse",  "ci", "r2", 'pearson', 'spearman'])
        for r in train_log:
            writer.writerow(r)
    print(f'Save log over at {log_dir}')

    # Test
    print(f"\n{'='*60}")
    print(f"Testing fold {fold_i} with best model...")
    print(f"{'='*60}")
    predModel = nn.DataParallel(LLMDTA(hp, device))
    predModel.load_state_dict(torch.load(model_fromTrain))
    predModel = predModel.to(device)    
    mse, rmse, ci, r2, pearson, spearman = test(predModel, test_dataset_load, is_valid=False)
    print(f'Test at fold-{fold_i}, mse: {mse}, rmse: {rmse}, ci: {ci}, r2: {r2}, pearson: {pearson}, spearman: {spearman}\n')
    
    # Log test metrics to wandb
    if use_wandb:
        wandb.log({
            'test/mse': mse,
            'test/rmse': rmse,
            'test/ci': ci,
            'test/r2': r2,
            'test/pearson': pearson,
            'test/spearman': spearman,
        })
        wandb.summary['final_test_mse'] = mse
        wandb.summary['final_test_rmse'] = rmse
        wandb.summary['final_test_ci'] = ci
        wandb.summary['final_test_r2'] = r2
        wandb.summary['final_test_pearson'] = pearson
        wandb.summary['final_test_spearman'] = spearman
    
    # Save test results for this fold
    fold_result_file = f'./log/Test-{hp.dataset}-{hp.running_set}-fold{fold_i}-{timestamp}.csv'
    fold_result = pd.DataFrame({
        'fold': [fold_i],
        'mse': [mse], 
        'rmse': [rmse], 
        'ci': [ci], 
        'r2': [r2], 
        'pearson': [pearson], 
        'spearman': [spearman]
    })
    fold_result.to_csv(fold_result_file, index=False)
    print(f"Fold {fold_i} results saved to: {fold_result_file}")
    print(f"{'='*60}")
    print(f"Training fold {fold_i} completed successfully!")
    print(f"{'='*60}")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("Weights & Biases run finished")