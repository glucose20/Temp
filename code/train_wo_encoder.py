"""
Training script for LLMDTA without Encoder (Linear instead of CNN)
==================================================================
This variant replaces the 1D-CNN encoder with simple linear projection.
Used to evaluate the contribution of the CNN encoder component.

Usage:
    python train_wo_encoder.py --fold 0 --dataset davis --running_set novel-drug
    python train_wo_encoder.py --fold 0 --dataset davis --running_set novel-pair --epochs 100
"""

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

from LLMDTA_ablation import LLMDTA_woEncoder
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
        mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data
        with torch.no_grad():
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


if __name__ == "__main__":
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
    parser.add_argument('--max_patience', type=int, default=None,
                        help='Max patience for early stopping (overrides hyperparameter.py setting)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate to use (overrides hyperparameter.py setting)')
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
    parser.add_argument('--encoder_dropout', type=float, default=None,
                        help='Dropout rate for encoder layers (overrides hyperparameter.py setting)')
    parser.add_argument('--cross_attention_dropout', type=float, default=None,
                        help='Dropout rate for cross-attention layers (overrides hyperparameter.py setting)')
    parser.add_argument('--expert_dropout', type=float, default=None,
                        help='Dropout rate for expert networks (overrides hyperparameter.py setting)')
    # MoE parameters
    parser.add_argument('--num_experts', type=int, default=None,
                        help='Number of experts in MoE (default: 4)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Number of experts to select per sample (default: 2)')
    parser.add_argument('--moe_noise_std', type=float, default=None,
                        help='Noise std for MoE exploration (default: 0.1, 0 to disable)')
    parser.add_argument('--load_balance_weight', type=float, default=None,
                        help='Weight for load balancing loss (default: 0.01, 0 to disable)')
    args = parser.parse_args()
    
    fold_i = args.fold

    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    
    hp = HyperParameter()
    
    # Override ESM settings
    if args.use_esmc is not None:
        hp.use_esmc = args.use_esmc
        if not hp.use_esmc:
            hp.protvec_dim = 1280
    
    if args.esmc_model is not None:
        hp.esmc_model = args.esmc_model
        if hp.esmc_model == "esmc_300m":
            hp.protvec_dim = 960
        elif hp.esmc_model == "esmc_600m":
            hp.protvec_dim = 1152
        elif hp.esmc_model == "esmc_6b":
            hp.protvec_dim = 2560
    
    if args.cuda is not None:
        hp.cuda = args.cuda
    if args.dataset is not None:
        hp.set_dataset(args.dataset)
    else:
        if args.use_esmc is not None or args.esmc_model is not None:
            hp.set_dataset(hp.dataset)
    if args.running_set is not None:
        hp.running_set = args.running_set.replace('_', '-')
    if args.epochs is not None:
        hp.Epoch = args.epochs
    if args.batch_size is not None:
        hp.Batch_size = args.batch_size
     # Override max patience if specified
    if args.max_patience is not None:
        hp.max_patience = args.max_patience
    # Override learning rate if specified
    if args.learning_rate is not None:
        hp.Learning_rate = args.learning_rate
    if args.num_experts is not None:
        hp.num_experts = args.num_experts
    if args.top_k is not None:
        hp.top_k = args.top_k
    if args.load_balance_weight is not None:
        hp.load_balance_weight = args.load_balance_weight
    # Override dropout rates if specified
    if args.encoder_dropout is not None:
        hp.encoder_dropout = args.encoder_dropout
    if args.cross_attention_dropout is not None:
        hp.cross_attention_dropout = args.cross_attention_dropout
    if args.expert_dropout is not None:
        hp.expert_dropout = args.expert_dropout

    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    print(f"=" * 70)
    print(f"🔬 LLMDTA w/o Encoder (Linear instead of CNN)")
    print(f"=" * 70)
    print(f"Training Fold {fold_i}/{hp.kfold-1}")
    print(f"Dataset: {hp.dataset}-{hp.running_set}") 
    print(f"Training config: {hp.Learning_rate}-{hp.Batch_size}-{hp.Epoch}-patience{hp.max_patience}")
    print("Dropout rates: encoder-{:.2f}, cross-attention-{:.2f}, expert-{:.2f}".format(
        hp.encoder_dropout, hp.cross_attention_dropout, hp.expert_dropout))
    print(f"ESM Model: {'ESM-C-' + hp.esmc_model if hp.use_esmc else 'ESM2'} (dim={hp.protvec_dim})")
    print(f"MoE: num_experts={hp.num_experts}, top_k={hp.top_k}, noise={hp.moe_noise_std}, lb_weight={hp.load_balance_weight}")
    print(f"Device: {device} (CUDA_VISIBLE_DEVICES={hp.cuda})")
    print(f"Pretrain-{hp.mol2vec_dir}")
    print(f"Pretrain-{hp.protvec_dir}")
    print(f"=" * 70)
    
    # Initialize Weights & Biases
    use_wandb = not args.no_wandb   
    if use_wandb:
        wandb_config = {
            'dataset': hp.dataset,
            'running_set': hp.running_set,
            'fold': fold_i,
            'epochs': hp.Epoch,
            'batch_size': hp.Batch_size,
            'patience': hp.max_patience,
            'learning_rate': hp.Learning_rate,
            'max_patience': hp.max_patience,
            'cuda_device': hp.cuda,
            'use_esmc': hp.use_esmc,
            'esmc_model': hp.esmc_model if hp.use_esmc else None,
            'protvec_dim': hp.protvec_dim,
            'num_experts': hp.num_experts,
            'top_k': hp.top_k,
            'moe_noise_std': hp.moe_noise_std,
            'load_balance_weight': hp.load_balance_weight,
            'encoder_dropout': hp.encoder_dropout,
            'cross_attention_dropout': hp.cross_attention_dropout,
            'expert_dropout': hp.expert_dropout
        }
        
        esm_name = f"esmc-{hp.esmc_model}" if hp.use_esmc else "esm2"
        run_name = f"wo_encoder-{hp.dataset}-{hp.running_set}-{esm_name}-fold{fold_i}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            tags=[hp.dataset, hp.running_set, 'wo_encoder', f'fold{fold_i}'],
            reinit=True
        )
    
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    
    if fold_i < 0 or fold_i >= hp.kfold:
        raise ValueError(f"Fold index must be between 0 and {hp.kfold-1}")
    
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    
    # Load data
    train_dir = os.path.join(dataset_root, f'fold_{fold_i}_train.csv')
    valid_dir = os.path.join(dataset_root, f'fold_{fold_i}_valid.csv')
    test_dir = os.path.join(dataset_root, f'fold_{fold_i}_test.csv')
    
    print(f"Loading fold {fold_i} data...")
    train_set = CustomDataSet(pd.read_csv(train_dir, sep=','), hp)
    valid_set = CustomDataSet(pd.read_csv(valid_dir, sep=','), hp)
    test_set = CustomDataSet(pd.read_csv(test_dir, sep=','), hp)
    
    train_dataset_load = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True, 
                                    num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    valid_dataset_load = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, 
                                    num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    test_dataset_load = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, 
                                   num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    
    print(f"Dataset loaded: {len(train_set)} train, {len(valid_set)} valid, {len(test_set)} test samples")

    criterion = F.mse_loss
    
    # Create model - USE LLMDTA_woEncoder instead of LLMDTA
    model = nn.DataParallel(LLMDTA_woEncoder(hp, device))
    model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
    # 1. Use AdamW for better Weight Decay performance
    # weight_decay usually ranges from 1e-4 to 1e-2
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=hp.Learning_rate, 
        betas=(0.9, 0.999),
        weight_decay=1e-4  
    )

    # 2. Initialize Cosine Annealing Scheduler
    # T_max is the total number of epochs. eta_min is the minimum LR it will drop to.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=hp.Epoch, 
        eta_min=1e-6
    )

    train_log = []     
    best_valid_mse = float('inf')
    patience = 0    
    
    timestamp = hp.current_time
    model_save_path = f'./savemodel/wo_encoder-{hp.dataset}-{hp.running_set}-fold{fold_i}-{timestamp}.pth'
    os.makedirs('./savemodel', exist_ok=True)
    
    print(f"Model will be saved to: {model_save_path}")
             
    for epoch in range(1, hp.Epoch + 1):    
        # Reset MoE usage stats
        if hasattr(model.module, 'reset_usage_stats'):
            model.module.reset_usage_stats()
        
        # Training
        model.train()
        pred = []
        label = []
        total_load_balance_loss = 0.0
        num_batches = 0
        
        for batch_data in tqdm(train_dataset_load, desc=f'Epoch {epoch}', leave=False):
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data                    
            predictions, gate_info = model.module.forward(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True)
            pred = pred + predictions.cpu().detach().numpy().reshape(-1).tolist()
            label = label + affinity.cpu().detach().numpy().reshape(-1).tolist()            
            
            loss = criterion(predictions.squeeze(), affinity)
            
            # Load balancing loss
            load_balance_loss = model.module.compute_load_balance_loss(gate_info['gate_weights'])
            total_load_balance_loss += load_balance_loss.item()
            num_batches += 1
            
            total_loss = loss + hp.load_balance_weight * load_balance_loss
            
            total_loss.backward()                
            optimizer.step()
            optimizer.zero_grad()
            
        pred = np.array(pred)
        label = np.array(label)
        mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(pred, label)
        train_log.append([mse_value, rmse_value, ci, r2, pearson_value, spearman_value])
        
        # MoE stats
        moe_stats = model.module.get_expert_usage_stats() if hasattr(model.module, 'get_expert_usage_stats') else None
        avg_load_balance_loss = total_load_balance_loss / num_batches if num_batches > 0 else 0
        
        print(f'Train epoch-{epoch}: mse={mse_value:.4f}, rmse={rmse_value:.4f}, r2={r2:.4f}')
        if moe_stats:
            # print(f'  MoE: entropy={moe_stats["usage_entropy"]:.4f}, lb_loss={avg_load_balance_loss:.4f}')
            print(f'  MoE Stats: usage_rate={moe_stats["expert_usage_rate"]}, entropy={moe_stats["usage_entropy"]:.4f}, dominant_expert={moe_stats["dominant_expert"]}, load_balance_loss={avg_load_balance_loss:.4f}')



        # 3. Step the scheduler at the end of the epoch
        scheduler.step()
        
        # 4. Log the current learning rate to Weights & Biases
        current_lr = optimizer.param_groups[0]['lr']
        # Log to wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'learning_rate': current_lr,
                'train/mse': mse_value,
                'train/rmse': rmse_value,
                'train/ci': ci,
                'train/r2': r2,
                'train/pearson': pearson_value,
                'train/spearman': spearman_value,
            }
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
        
        # Validation
        mse, rmse, ci, r2, pearson, spearman = test(model, valid_dataset_load, is_valid=True)   
        print(f'Valid epoch-{epoch}: mse={mse:.4f}, rmse={rmse:.4f}, r2={r2:.4f}')
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
        
        # Early stopping
        if mse < best_valid_mse:
            patience = 0
            best_valid_mse = mse
            torch.save(model.state_dict(), model_save_path)
            print(f'  ✓ New best! Saved model.')
            print(f'Update best_mse, Valid at fold-{fold_i} epoch-{epoch}: mse-{mse}, rmse-{rmse}, ci-{ci}, r2-{r2}, pearson-{pearson}, spearman-{spearman}')


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
                print(f'Early stopping at epoch-{epoch}')
                break   
    
    # Save training log
    log_dir = f"./log/wo_encoder-{timestamp}-{hp.dataset}-{hp.running_set}-fold{fold_i}.csv"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    with open(log_dir, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["mse", "rmse", "ci", "r2", 'pearson', 'spearman'])
        for r in train_log:
            writer.writerow(r)
    print(f'Training log saved: {log_dir}')

    # Test
    print(f"\n{'='*60}")
    print(f"Testing with best model...")
    print(f"{'='*60}")
    
    predModel = nn.DataParallel(LLMDTA_woEncoder(hp, device))
    predModel.load_state_dict(torch.load(model_save_path))
    predModel = predModel.to(device)    
    
    mse, rmse, ci, r2, pearson, spearman = test(predModel, test_dataset_load, is_valid=False)
    # print(f'Test Results: mse={mse}, rmse={rmse}, ci={ci}, r2={r2}, pearson={pearson}, spearman={spearman}')
    print(f'Test at fold-{fold_i}, mse: {mse}, rmse: {rmse}, ci: {ci}, r2: {r2}, pearson: {pearson}, spearman: {spearman}\n')

    
    if use_wandb:
        wandb.log({
            'test/mse': mse, 'test/rmse': rmse, 'test/ci': ci,
            'test/r2': r2, 'test/pearson': pearson, 'test/spearman': spearman,
        })
        wandb.summary['final_test_mse'] = mse
        wandb.summary['final_test_rmse'] = rmse
        wandb.summary['final_test_ci'] = ci
        wandb.summary['final_test_r2'] = r2
        wandb.summary['final_test_pearson'] = pearson
        wandb.summary['final_test_spearman'] = spearman
    
    # Save test results
    result_file = f'./log/Test-wo_encoder-{hp.dataset}-{hp.running_set}-fold{fold_i}-{timestamp}.csv'
    result_df = pd.DataFrame({
        'fold': [fold_i], 'mse': [mse], 'rmse': [rmse], 
        'ci': [ci], 'r2': [r2], 'pearson': [pearson], 'spearman': [spearman]
    })
    result_df.to_csv(result_file, index=False)
    print(f"Results saved: {result_file}")
    
    print(f"\n{'='*60}")
    print(f"✅ Training wo_encoder fold {fold_i} completed!")
    print(f"{'='*60}")
    
    if use_wandb:
        wandb.finish()
