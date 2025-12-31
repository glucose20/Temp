"""
Component Ablation Study for LLMDTA Model
==========================================
This script evaluates the contribution of each model component by
removing/replacing them one at a time (proper ablation study style).

Model Components being evaluated:
1. Encoder Block (1D-CNN with skip-connections)
2. Cross-Attention (Drug ↔ Protein interaction)
3. Self-Attention Pooling
4. Residual Feature Fusion (h_pre + h_post)
5. Mixture of Experts (MoE)
6. Full Pipeline vs Linear-only (AI-Bind style)

Usage:
    python ablation_component.py --dataset davis --running_set warm --fold 0
    python ablation_component.py --dataset davis --running_set novel-drug --fold 0 --all_folds
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle

from LLMDTA_ablation import get_ablation_model, get_all_variants
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn

from sklearn.metrics import r2_score
from scipy import stats
from math import sqrt
from tqdm import tqdm


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)


def cindex_score(y, p):
    sum_m = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if y[i] > y[j]:
                pair += 1
                sum_m += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    return sum_m / pair if pair != 0 else 0


def regression_scores(label, pred, compute_ci=True):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    mse = ((label - pred)**2).mean(axis=0)
    rmse = sqrt(mse)
    ci = cindex_score(label, pred) if compute_ci else -1
    r2 = r2_score(label, pred)
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return {
        'mse': round(mse, 6),
        'rmse': round(rmse, 6),
        'ci': round(ci, 6),
        'r2': round(r2, 6),
        'pearson': round(pearson, 6),
        'spearman': round(spearman, 6)
    }


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_one_epoch(model, train_loader, optimizer, criterion, hp, max_grad_norm=1.0):
    """Train model for one epoch with gradient clipping."""
    model.train()
    preds, labels = [], []
    total_loss = 0.0
    
    for batch_data in tqdm(train_loader, desc='Training', leave=False):
        mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data
        
        if hasattr(model, 'module'):
            predictions, gate_info = model.module.forward(
                mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True
            )
            lb_loss = model.module.compute_load_balance_loss(gate_info['gate_weights'])
        else:
            predictions, gate_info = model(
                mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True
            )
            lb_loss = model.compute_load_balance_loss(gate_info['gate_weights'])
        
        preds.extend(predictions.cpu().detach().numpy().reshape(-1).tolist())
        labels.extend(affinity.cpu().detach().numpy().reshape(-1).tolist())
        
        loss = criterion(predictions.squeeze(), affinity)
        if lb_loss is not None and lb_loss.item() > 0:
            loss = loss + getattr(hp, 'load_balance_weight', 0.01) * lb_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        total_loss += loss.item()
    
    return regression_scores(np.array(labels), np.array(preds), compute_ci=False)


def evaluate(model, dataloader, compute_ci=True):
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='Evaluating', leave=False):
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data
            
            if hasattr(model, 'module'):
                pred = model.module.forward(
                    mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=False
                )
            else:
                pred = model.forward(
                    mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=False
                )
            
            preds.extend(pred.cpu().numpy().reshape(-1).tolist())
            labels.extend(affinity.cpu().numpy().reshape(-1).tolist())
    
    return regression_scores(np.array(labels), np.array(preds), compute_ci=compute_ci)


def train_and_evaluate(variant_name, hp, device, train_loader, valid_loader, test_loader,
                       epochs=100, patience=20, verbose=True):
    """Full training with early stopping."""
    
    # Create model variant
    model = get_ablation_model(variant_name, hp, device)
    model = nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
    criterion = F.mse_loss
    
    best_valid_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_test_metrics = None
    
    for epoch in range(1, epochs + 1):
        if hasattr(model.module, 'reset_usage_stats'):
            model.module.reset_usage_stats()
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, hp)
        valid_metrics = evaluate(model, valid_loader, compute_ci=False)
        
        if valid_metrics['mse'] < best_valid_mse:
            best_valid_mse = valid_metrics['mse']
            best_epoch = epoch
            patience_counter = 0
            
            test_metrics = evaluate(model, test_loader, compute_ci=True)
            best_test_metrics = test_metrics.copy()
            best_test_metrics['best_epoch'] = epoch
            best_test_metrics['best_valid_mse'] = best_valid_mse
            
            if hasattr(model.module, 'get_expert_usage_stats'):
                moe_stats = model.module.get_expert_usage_stats()
                if moe_stats:
                    best_test_metrics['entropy'] = moe_stats.get('usage_entropy', 0.0)
        else:
            patience_counter += 1
        
        if verbose and epoch % 20 == 0:
            print(f"    Epoch {epoch}: Valid MSE={valid_metrics['mse']:.4f}, Best={best_valid_mse:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopped at epoch {epoch}")
            break
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return best_test_metrics


# ============================================================
# ABLATION STUDY RUNNER
# ============================================================

def run_component_ablation(hp, device, train_loader, valid_loader, test_loader,
                           epochs=100, patience=20, verbose=True):
    """Run ablation study on all model variants."""
    
    print("\n" + "="*70)
    print("🔬 COMPONENT ABLATION STUDY")
    print("="*70)
    print("Evaluating contribution of each model component...")
    print("="*70)
    
    variants = get_all_variants()
    results = []
    baseline_metrics = None
    
    for i, (variant_name, description) in enumerate(variants):
        print(f"\n[{i+1}/{len(variants)}] {description}")
        print("-"*50)
        
        set_seed(42)
        
        try:
            start_time = time.time()
            metrics = train_and_evaluate(
                variant_name, hp, device, 
                train_loader, valid_loader, test_loader,
                epochs=epochs, patience=patience, verbose=verbose
            )
            elapsed = time.time() - start_time
            
            # Store baseline for comparison
            if variant_name == 'full':
                baseline_metrics = metrics.copy()
            
            result = {
                'variant': variant_name,
                'description': description,
                **metrics,
                'training_time': round(elapsed, 2),
                'status': 'success'
            }
            
            # Calculate difference from baseline
            if baseline_metrics and variant_name != 'full':
                result['mse_diff'] = round(metrics['mse'] - baseline_metrics['mse'], 6)
                result['ci_diff'] = round(metrics['ci'] - baseline_metrics['ci'], 6)
                result['mse_diff_pct'] = round((metrics['mse'] - baseline_metrics['mse']) / baseline_metrics['mse'] * 100, 2)
                result['ci_diff_pct'] = round((metrics['ci'] - baseline_metrics['ci']) / baseline_metrics['ci'] * 100, 2)
            
            results.append(result)
            
            print(f"  ✓ CI={metrics['ci']:.4f}, MSE={metrics['mse']:.4f}, "
                  f"R²={metrics['r2']:.4f}, Epoch={metrics['best_epoch']}")
            
            if baseline_metrics and variant_name != 'full':
                diff_emoji = "📉" if result['ci_diff'] < 0 else "📈"
                print(f"  {diff_emoji} vs Full: CI {result['ci_diff']:+.4f} ({result['ci_diff_pct']:+.2f}%), "
                      f"MSE {result['mse_diff']:+.4f} ({result['mse_diff_pct']:+.2f}%)")
        
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            results.append({
                'variant': variant_name,
                'description': description,
                'status': 'failed',
                'error': str(e)
            })
            continue
    
    return results, baseline_metrics


def print_ablation_summary(results, baseline_metrics):
    """Print formatted ablation study summary."""
    
    print("\n" + "="*80)
    print("📊 ABLATION STUDY RESULTS SUMMARY")
    print("="*80)
    
    # Filter successful results only
    successful_results = [r for r in results if r.get('status', 'success') == 'success']
    
    if not successful_results:
        print("No successful experiments to report.")
        return
    
    # Sort by CI difference (most impactful first)
    sorted_results = sorted(
        [r for r in successful_results if r['variant'] != 'full'],
        key=lambda x: x.get('ci_diff', 0)
    )
    
    print(f"\n{'Component Removed':<35} {'CI':>8} {'Δ CI':>10} {'MSE':>8} {'Δ MSE':>10}")
    print("-"*80)
    
    # Print baseline first
    baseline = next((r for r in successful_results if r['variant'] == 'full'), None)
    if baseline:
        print(f"{'Full Model (Baseline)':<35} {baseline['ci']:>8.4f} {'---':>10} "
              f"{baseline['mse']:>8.4f} {'---':>10}")
        print("-"*80)
    
    # Print ablated variants
    for r in sorted_results:
        emoji = "⬇️" if r.get('ci_diff', 0) < -0.01 else "➡️" if abs(r.get('ci_diff', 0)) < 0.01 else "⬆️"
        ci_diff_str = f"{r.get('ci_diff', 0):+.4f}" if 'ci_diff' in r else "N/A"
        mse_diff_str = f"{r.get('mse_diff', 0):+.4f}" if 'mse_diff' in r else "N/A"
        
        print(f"{emoji} {r['description']:<33} {r['ci']:>8.4f} {ci_diff_str:>10} "
              f"{r['mse']:>8.4f} {mse_diff_str:>10}")
    
    # Component importance ranking
    print("\n" + "="*80)
    print("🏆 COMPONENT IMPORTANCE RANKING (by CI drop when removed)")
    print("="*80)
    
    importance = []
    for r in sorted_results:
        if 'ci_diff' in r:
            importance.append({
                'component': r['description'].replace('w/o ', ''),
                'ci_drop': -r['ci_diff'],  # Positive = important
                'mse_increase': r['mse_diff']
            })
    
    importance.sort(key=lambda x: x['ci_drop'], reverse=True)
    
    for i, item in enumerate(importance):
        bar_len = int(max(0, item['ci_drop'] * 500))  # Scale for visualization
        bar = "█" * min(bar_len, 30)
        print(f"{i+1}. {item['component']:<35} CI drop: {item['ci_drop']:+.4f} {bar}")
    
    print("\n" + "="*80)
    print("📝 INTERPRETATION:")
    print("-"*80)
    
    if importance:
        most_important = importance[0]
        least_important = importance[-1]
        
        print(f"• Most critical component: {most_important['component']}")
        print(f"  → Removing it causes CI to drop by {most_important['ci_drop']:.4f}")
        print(f"• Least critical component: {least_important['component']}")
        print(f"  → Removing it only changes CI by {least_important['ci_drop']:.4f}")
    
    print("="*80)


def save_ablation_results(results, output_path, metadata=None):
    """Save results to CSV and JSON."""
    df = pd.DataFrame(results)
    df.to_csv(output_path + '.csv', index=False)
    print(f"\n💾 Results saved to: {output_path}.csv")
    
    output_data = {'metadata': metadata or {}, 'results': results}
    with open(output_path + '.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"💾 Results saved to: {output_path}.json")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LLMDTA Component Ablation Study')
    
    parser.add_argument('--dataset', type=str, default='davis',
                        choices=['davis', 'kiba', 'metz'])
    parser.add_argument('--running_set', type=str, default='warm',
                        choices=['warm', 'novel-drug', 'novel-prot', 'novel-pair'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--all_folds', action='store_true',
                        help='Run ablation on all 5 folds and average')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    
    parser.add_argument('--output_dir', type=str, default='./ablation_results')
    parser.add_argument('--cuda', type=str, default='0')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup
    hp = HyperParameter()
    hp.set_dataset(args.dataset)
    hp.running_set = args.running_set
    hp.Batch_size = args.batch_size
    hp.Learning_rate = args.lr
    hp.cuda = args.cuda
    
    # Set optimal MoE params
    hp.num_experts = 4
    hp.top_k = 2
    hp.moe_noise_std = 0.1
    hp.load_balance_weight = 0.01
    
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"🧪 LLMDTA COMPONENT ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}-{args.running_set}")
    print(f"Folds: {'All (0-4)' if args.all_folds else args.fold}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}")
    print(f"Learning Rate: {args.lr}")
    print(f"{'='*70}")
    
    # Load data
    print("\nLoading data...")
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    
    folds = range(5) if args.all_folds else [args.fold]
    all_fold_results = []
    
    for fold in folds:
        print(f"\n{'='*70}")
        print(f"FOLD {fold}")
        print(f"{'='*70}")
        
        dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
        
        train_set = CustomDataSet(pd.read_csv(os.path.join(dataset_root, f'fold_{fold}_train.csv')), hp)
        valid_set = CustomDataSet(pd.read_csv(os.path.join(dataset_root, f'fold_{fold}_valid.csv')), hp)
        test_set = CustomDataSet(pd.read_csv(os.path.join(dataset_root, f'fold_{fold}_test.csv')), hp)
        
        # Create collate function with captured variables
        def collate_fn(batch_data):
            return my_collate_fn(batch_data, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict)
        
        train_loader = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True,
                                  drop_last=True, num_workers=0, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False,
                                  drop_last=True, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False,
                                 drop_last=True, num_workers=0, collate_fn=collate_fn)
        
        print(f"Data: {len(train_set)} train, {len(valid_set)} valid, {len(test_set)} test")
        
        # Run ablation
        results, baseline = run_component_ablation(
            hp, device, train_loader, valid_loader, test_loader,
            epochs=args.epochs, patience=args.patience, verbose=True
        )
        
        # Add fold info
        for r in results:
            r['fold'] = fold
        
        all_fold_results.extend(results)
        
        # Print summary for this fold
        print_ablation_summary(results, baseline)
        
        # Save per-fold results
        output_path = os.path.join(args.output_dir,
            f'ablation_{args.dataset}_{args.running_set}_fold{fold}_{timestamp}')
        
        metadata = {
            'dataset': args.dataset,
            'running_set': args.running_set,
            'fold': fold,
            'epochs': args.epochs,
            'patience': args.patience,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'timestamp': timestamp
        }
        save_ablation_results(results, output_path, metadata)
    
    # Aggregate results across folds
    if args.all_folds and len(folds) > 1:
        print("\n" + "="*80)
        print("📊 AGGREGATED RESULTS (Mean ± Std across all folds)")
        print("="*80)
        
        # Filter only successful results for aggregation
        successful_fold_results = [r for r in all_fold_results if r.get('status', 'success') == 'success']
        
        if not successful_fold_results:
            print("No successful experiments to aggregate.")
        else:
            df = pd.DataFrame(successful_fold_results)
            
            # Group by variant and compute mean/std
            agg_results = []
            for variant in df['variant'].unique():
                variant_df = df[df['variant'] == variant]
                agg = {
                    'variant': variant,
                    'description': variant_df['description'].iloc[0],
                    'ci_mean': variant_df['ci'].mean(),
                    'ci_std': variant_df['ci'].std(),
                    'mse_mean': variant_df['mse'].mean(),
                    'mse_std': variant_df['mse'].std(),
                    'r2_mean': variant_df['r2'].mean(),
                    'r2_std': variant_df['r2'].std(),
                }
                if 'ci_diff' in variant_df.columns:
                    agg['ci_diff_mean'] = variant_df['ci_diff'].mean()
                    agg['ci_diff_std'] = variant_df['ci_diff'].std()
                agg_results.append(agg)
            
            # Print aggregated table
            print(f"\n{'Component':<35} {'CI (mean±std)':>18} {'MSE (mean±std)':>18}")
            print("-"*80)
            
            for r in agg_results:
                ci_str = f"{r['ci_mean']:.4f}±{r['ci_std']:.4f}"
                mse_str = f"{r['mse_mean']:.4f}±{r['mse_std']:.4f}"
                print(f"{r['description']:<35} {ci_str:>18} {mse_str:>18}")
            
            # Save aggregated results
            agg_df = pd.DataFrame(agg_results)
            agg_path = os.path.join(args.output_dir,
                f'ablation_{args.dataset}_{args.running_set}_AGGREGATED_{timestamp}')
            agg_df.to_csv(agg_path + '.csv', index=False)
            print(f"\n💾 Aggregated results saved to: {agg_path}.csv")
    
    print(f"\n{'='*70}")
    print(f"✅ ABLATION STUDY COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
