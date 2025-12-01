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

from LLMDTA_MoE import LLMDTA_MoE, load_balancing_loss
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, batch2tensor, my_collate_fn

from sklearn.metrics import r2_score
from tqdm import tqdm
from math import sqrt
from scipy import stats
import csv


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


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


def val(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    pred_list, label_list = [], []
    
    for data in dataloader:
        drug_vec, prot_vec, drug_mat, drug_mask, prot_mat, prot_mask, label = data
        label = label.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
        with torch.no_grad():
            pred, routing_weights = model(drug_vec, drug_mat, drug_mask, prot_vec, prot_mat, prot_mask)
            loss = F.mse_loss(pred, label)
        running_loss += loss.item()
        pred_list.append(pred.detach().cpu())
        label_list.append(label.detach().cpu())
    
    pred_list = torch.cat(pred_list, dim=0).numpy()
    label_list = torch.cat(label_list, dim=0).numpy()
    mse, rmse, ci, r2, pearson, spearman = regression_scores(label_list, pred_list, is_valid=True)
    running_loss = running_loss / len(dataloader)
    
    return running_loss, mse, rmse, ci, r2, pearson, spearman


def test(model, dataloader, device):
    model.eval()
    pred_list, label_list = [], []
    
    for data in dataloader:
        drug_vec, prot_vec, drug_mat, drug_mask, prot_mat, prot_mask, label = data
        label = label.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
        with torch.no_grad():
            pred, routing_weights = model(drug_vec, drug_mat, drug_mask, prot_vec, prot_mat, prot_mask)
        pred_list.append(pred.detach().cpu())
        label_list.append(label.detach().cpu())
    
    pred_list = torch.cat(pred_list, dim=0).numpy()
    label_list = torch.cat(label_list, dim=0).numpy()
    mse, rmse, ci, r2, pearson, spearman = regression_scores(label_list, pred_list, is_valid=False)
    
    return mse, rmse, ci, r2, pearson, spearman


def main(hp, fold, num_experts=4, top_k=2, lb_weight=0.001, lb_method='entropy', pretrained_path=None, seed=42):
    # Set seed for reproducibility
    set_seed(seed)
    
    print("=" * 100)
    print(f"Training LLMDTA with Mixture of Experts - {hp.dataset}/{hp.running_set}/fold{fold}")
    print(f"Seed: {seed}, Experts: {num_experts}, Top-K: {top_k}")
    print(f"LB Weight: {lb_weight}, LB Method: {lb_method}")
    print("=" * 100)
    
    # Device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    
    # Load data for the specified fold
    train_dir = os.path.join(dataset_root, f'fold_{fold}_train.csv')
    valid_dir = os.path.join(dataset_root, f'fold_{fold}_valid.csv')
    test_dir = os.path.join(dataset_root, f'fold_{fold}_test.csv')
    
    train_dataset = CustomDataSet(pd.read_csv(train_dir, sep=','), hp)
    valid_dataset = CustomDataSet(pd.read_csv(valid_dir, sep=','), hp)
    test_dataset = CustomDataSet(pd.read_csv(test_dir, sep=','), hp)
    
    train_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, 
                             collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict), 
                             num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, 
                             collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict), 
                             num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, 
                            collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict), 
                            num_workers=0)
    
    print(f"Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}, Test size: {len(test_dataset)}")
    
    # Model setup - MoE with configurable experts and top-k
    model = LLMDTA_MoE(hp, device, num_experts=num_experts, top_k=top_k).to(device)
    print(f"Model created with {model.num_experts} experts, top-{model.top_k} routing")
    
    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_pretrained_base(pretrained_path)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate)
    
    # Training
    best_val_mse = float('inf')
    best_epoch = 0
    patience = 0
    
    # Loss balancing weight with warmup
    warmup_epochs = 10  # Warmup for first 10 epochs
    print(f"Load balancing weight: {lb_weight} (with {warmup_epochs} epochs warmup)")
    print(f"Load balancing method: {lb_method}")
    
    # Adaptive lb_weight: reduce when validation improves
    adaptive_lb = True  # Enable adaptive adjustment
    lb_reduction_factor = 0.9  # Reduce by 10% when val improves
    
    print("\nStarting training...")
    for epoch in range(hp.Epoch):
        model.train()
        train_loss = 0.0
        train_pred_loss = 0.0
        train_lb_loss = 0.0
        
        # Warmup schedule for load balancing loss
        if epoch < warmup_epochs:
            current_lb_weight = lb_weight * (epoch + 1) / warmup_epochs
        else:
            current_lb_weight = lb_weight
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{hp.Epoch}"):
            drug_vec, prot_vec, drug_mat, drug_mask, prot_mat, prot_mask, label = data
            label = label.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
            
            optimizer.zero_grad()
            pred, routing_weights = model(drug_vec, drug_mat, drug_mask, prot_vec, prot_mat, prot_mask)
            
            # Main prediction loss
            pred_loss = F.mse_loss(pred, label)
            
            # Load balancing loss to encourage expert diversity
            lb_loss = load_balancing_loss(routing_weights, model.num_experts, method=lb_method)
            
            # Total loss with warmup
            loss = pred_loss + current_lb_weight * lb_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pred_loss += pred_loss.item()
            train_lb_loss += lb_loss.item()
        
        train_loss /= len(train_loader)
        train_pred_loss /= len(train_loader)
        train_lb_loss /= len(train_loader)
        
        # Validation
        val_loss, val_mse, val_rmse, val_ci, val_r2, val_pearson, val_spearman = val(model, valid_loader, device)
        
        print(f"\nEpoch {epoch+1}/{hp.Epoch}")
        print(f"Train Loss: {train_loss:.6f} (Pred: {train_pred_loss:.6f}, LB: {train_lb_loss:.6f}, LB_weight: {current_lb_weight:.6f})")
        print(f"Valid Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, RMSE: {val_rmse:.6f}, R2: {val_r2:.6f}")
        
        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1
            patience = 0
            
            # Adaptive: reduce lb_weight when validation improves
            if adaptive_lb and epoch > warmup_epochs:
                lb_weight = max(lb_weight * lb_reduction_factor, 0.0001)  # Min 0.0001
                print(f"  → Validation improved! Reducing LB weight to {lb_weight:.6f}")
            
            # Save best model
            model_name = f"{hp.dataset}-{hp.running_set}-fold{fold}-MoE-{hp.current_time}.pth"
            torch.save(model.state_dict(), f"./savemodel/{model_name}")
            print(f"✓ Model saved: {model_name}")
        else:
            patience += 1
            if patience >= hp.max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model and test
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(f"./savemodel/{model_name}"))
    
    test_mse, test_rmse, test_ci, test_r2, test_pearson, test_spearman = test(model, test_loader, device)
    
    print("\n" + "=" * 100)
    print(f"Test Results - {hp.dataset}/{hp.running_set}/fold{fold}")
    print("=" * 100)
    print(f"MSE: {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"CI: {test_ci:.6f}")
    print(f"R2: {test_r2:.6f}")
    print(f"Pearson: {test_pearson:.6f}")
    print(f"Spearman: {test_spearman:.6f}")
    print("=" * 100)
    
    # Save results
    result = {
        'dataset': hp.dataset,
        'running_set': hp.running_set,
        'fold': fold,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_ci': test_ci,
        'test_r2': test_r2,
        'test_pearson': test_pearson,
        'test_spearman': test_spearman,
        'best_epoch': best_epoch,
        'model_name': model_name
    }
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LLMDTA with Mixture of Experts')
    parser.add_argument('--dataset', type=str, default='davis', help='Dataset name: davis, kiba, metz')
    parser.add_argument('--running_set', type=str, default='warm', help='Task setting: warm, novel-drug, novel-prot, novel-pair')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation')
    parser.add_argument('--all_folds', action='store_true', help='Train all 5 folds')
    parser.add_argument('--cuda', type=str, default='0', help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (default: from hyperparameter.py)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts')
    parser.add_argument('--top_k', type=int, default=2, help='Number of top experts to select')
    parser.add_argument('--lb_weight', type=float, default=0.001, help='Load balancing loss weight (reduced default)')
    parser.add_argument('--lb_method', type=str, default='entropy', choices=['cv', 'entropy', 'importance'], 
                        help='Load balancing method: cv, entropy, or importance')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained LLMDTA model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    hp = HyperParameter()
    hp.set_dataset(args.dataset)
    hp.running_set = args.running_set
    
    # Override hyperparameters from command line arguments
    if args.cuda is not None:
        hp.cuda = args.cuda
    if args.epochs is not None:
        hp.Epoch = args.epochs
    if args.batch_size is not None:
        hp.Batch_size = args.batch_size
    if args.lr is not None:
        hp.Learning_rate = args.lr
    
    if args.all_folds:
        all_results = []
        for fold in range(hp.kfold):
            result = main(hp, fold, args.num_experts, args.top_k, args.lb_weight, args.lb_method, args.pretrained, args.seed)
            all_results.append(result)
        
        # Aggregate results
        print("\n" + "=" * 100)
        print("SUMMARY - All Folds")
        print("=" * 100)
        
        metrics = ['test_mse', 'test_rmse', 'test_ci', 'test_r2', 'test_pearson', 'test_spearman']
        for metric in metrics:
            values = [r[metric] for r in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")
        
        # Save summary
        summary_file = f"./results/{hp.dataset}_{hp.running_set}_MoE_{hp.current_time}.csv"
        os.makedirs('./results', exist_ok=True)
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nResults saved to: {summary_file}")
    else:
        main(hp, args.fold, args.num_experts, args.top_k, args.lb_weight, args.lb_method, args.pretrained, args.seed)
