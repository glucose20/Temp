"""
Evaluate trained LLMDTA model on validation/test set
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
from math import sqrt
from tqdm import tqdm

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from LLMDTA import LLMDTA
from hyperparameter_full import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn


def cindex_score(y, p):
    """Concordance index"""
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


def regression_scores(label, pred):
    """Calculate all regression metrics"""
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    
    mse = ((label - pred)**2).mean(axis=0)
    rmse = sqrt(mse)
    ci = cindex_score(label, pred)
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


def evaluate_model(model_path, dataset='davis', data_root=None, batch_size=256):
    """Evaluate model on validation set"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating Model on {dataset.upper()} Dataset")
    print(f"{'='*80}\n")
    
    SEED = 0
    
    # Initialize hyperparameters
    hp = HyperParameter()
    hp.dataset = dataset
    hp.Batch_size = batch_size
    if data_root:
        hp.data_root = data_root
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset}")
    print(f"Batch size: {batch_size}\n")
    
    # Load pretrained features
    print("Loading pretrained features...")
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = pickle.load(open(hp.mol2vec_dir, 'rb'))
    protvec_dict = pickle.load(open(hp.protvec_dir, 'rb'))
    print(f"✓ Loaded {len(drug_df)} drugs and {len(prot_df)} proteins\n")
    
    # Load dataset
    data_file = f'{hp.data_root}/{dataset}.txt'
    if not os.path.exists(data_file):
        print(f"ERROR: Dataset file not found: {data_file}")
        return None
    
    print(f"Loading dataset from {data_file}...")
    df = pd.read_csv(data_file, sep=' ', header=None)
    df.columns = hp.dataset_columns
    df = df[['drug_id', 'prot_id', 'label']]
    print(f"✓ Loaded {len(df)} samples\n")
    
    # Split into train and validation (same as training)
    print("Splitting dataset (80/20)...")
    train_df, valid_df = train_test_split(
        df, 
        test_size=hp.valid_ratio, 
        random_state=SEED,
        shuffle=True
    )
    print(f"✓ Train: {len(train_df)} samples")
    print(f"✓ Valid: {len(valid_df)} samples\n")
    
    # Create validation dataset
    valid_dataset = CustomDataSet(valid_df, hp)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=hp.Batch_size, 
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict)
    )
    
    # Load model
    print("Loading model...")
    model = nn.DataParallel(LLMDTA(hp, device))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully\n")
    
    # Evaluate
    print(f"{'='*80}")
    print("Evaluating on Validation Set...")
    print(f"{'='*80}\n")
    
    preds = []
    labels = []
    
    with torch.no_grad():
        for batch_data in tqdm(valid_loader, desc="Evaluating"):
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data
            
            pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
            preds += pred.cpu().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist()
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    # Calculate metrics
    metrics = regression_scores(labels, preds)
    
    # Print results
    print(f"\n{'='*80}")
    print("Validation Results")
    print(f"{'='*80}\n")
    print(f"Samples evaluated: {len(labels)}")
    print(f"\nMetrics:")
    print(f"  MSE:      {metrics['mse']:.6f}")
    print(f"  RMSE:     {metrics['rmse']:.6f}")
    print(f"  CI:       {metrics['ci']:.6f}")
    print(f"  R²:       {metrics['r2']:.6f}")
    print(f"  Pearson:  {metrics['pearson']:.6f}")
    print(f"  Spearman: {metrics['spearman']:.6f}")
    
    print(f"\nPrediction statistics:")
    print(f"  Min:  {preds.min():.3f}")
    print(f"  Max:  {preds.max():.3f}")
    print(f"  Mean: {preds.mean():.3f}")
    print(f"  Std:  {preds.std():.3f}")
    
    print(f"\nLabel statistics:")
    print(f"  Min:  {labels.min():.3f}")
    print(f"  Max:  {labels.max():.3f}")
    print(f"  Mean: {labels.mean():.3f}")
    print(f"  Std:  {labels.std():.3f}")
    
    print(f"\n{'='*80}\n")
    
    return metrics, preds, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LLMDTA model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model .pth file')
    parser.add_argument('--dataset', type=str, default='davis',
                        help='Dataset name (davis/kiba/metz)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to dataset root directory')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.dataset, args.data_root, args.batch_size)
