"""
Train LLMDTA model on full dataset without fold split.
This creates models like All-davis, All-kiba, All-metz.
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
from sklearn.model_selection import train_test_split

from LLMDTA import LLMDTA as LLMDTA
from hyperparameter_full import HyperParameter
from MyDataset import CustomDataSet, batch2tensor, my_collate_fn

from sklearn.metrics import r2_score
from tqdm import tqdm
from math import sqrt
from scipy import stats
import csv


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
            pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist()

    preds = np.array(preds)
    labels = np.array(labels)
    mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(labels, preds, is_valid)
    return mse_value, rmse_value, ci, r2, pearson_value, spearman_value


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LLMDTA model on full dataset')
    parser.add_argument('--cuda', type=str, default=None,
                        help='CUDA device ID (e.g., "0", "1")')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name: davis, kiba, or metz')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to dataset root directory (e.g., /kaggle/input/llmdta/dta-origin-dataset)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size to use')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    args = parser.parse_args()

    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    
    hp = HyperParameter()
    
    # Override parameters from command line
    if args.cuda is not None:
        hp.cuda = args.cuda
    if args.data_root is not None:
        hp.data_root = args.data_root
    if args.dataset is not None:
        hp.set_dataset(args.dataset)
    if args.epochs is not None:
        hp.Epoch = args.epochs
    if args.batch_size is not None:
        hp.Batch_size = args.batch_size
    if args.lr is not None:
        hp.Learning_rate = args.lr
    
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"Training LLMDTA on Full {hp.dataset.upper()} Dataset")
    print(f"{'='*80}")
    print(f"Dataset: {hp.dataset}")
    print(f"Data root: {hp.data_root}")
    print(f"Running set: {hp.running_set}")
    print(f"Device: {device}")
    print(f"Epochs: {hp.Epoch}")
    print(f"Batch size: {hp.Batch_size}")
    print(f"Learning rate: {hp.Learning_rate}")
    print(f"{'='*80}\n")
    
    # Load pretrained features
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    
    # Load full dataset
    data_file = f'{hp.data_root}/{hp.dataset}.txt'
    if not os.path.exists(data_file):
        print(f"ERROR: Dataset file not found: {data_file}")
        print(f"Please make sure you have {hp.dataset}.txt in {hp.data_root}/")
        sys.exit(1)
    
    # Read dataset with correct number of columns
    df = pd.read_csv(data_file, sep=' ', header=None)
    df.columns = hp.dataset_columns
    
    # Keep only needed columns for training
    df = df[['drug_id', 'prot_id', 'label']]
    
    print(f"Loaded {len(df)} samples from {data_file}")
    
    # Split into train and validation (80/20)
    train_df, valid_df = train_test_split(
        df, 
        test_size=hp.valid_ratio, 
        random_state=SEED,
        shuffle=True
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    
    # Load drug and protein data
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    
    # Create datasets (CustomDataSet only takes dataset and hp)
    train_dataset = CustomDataSet(train_df, hp)
    valid_dataset = CustomDataSet(valid_df, hp)
    
    # Create dataloaders (collate_fn needs all required parameters)
    train_dataset_load = DataLoader(
        train_dataset, 
        batch_size=hp.Batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict)
    )
    valid_dataset_load = DataLoader(
        valid_dataset, 
        batch_size=hp.Batch_size, 
        shuffle=False,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict)
    )
    
    # Initialize model
    model = nn.DataParallel(LLMDTA(hp, device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
    criterion = F.mse_loss
    
    # Model save path
    model_save_path = f'./savemodel/All-{hp.dataset}-{hp.current_time}.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"\nModel will be saved to: {model_save_path}\n")
    
    # Training loop
    best_valid_mse = float('inf')
    patience = 0
    train_log = []
    
    for epoch in range(hp.Epoch):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{hp.Epoch}")
        print(f"{'='*80}")
        
        # Training
        model.train()
        pred = []
        label = []
        
        for batch_i, batch_data in enumerate(tqdm(train_dataset_load, desc="Training")):
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data
            
            predictions = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
            pred += predictions.cpu().detach().numpy().reshape(-1).tolist()
            label += affinity.cpu().detach().numpy().reshape(-1).tolist()
            
            loss = criterion(predictions.squeeze(), affinity)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        pred = np.array(pred)
        label = np.array(label)
        mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(label, pred)
        train_log.append([epoch, mse_value, rmse_value, ci, r2, pearson_value, spearman_value])
        
        print(f'\nTraining Results:')
        print(f'  MSE: {mse_value:.6f}')
        print(f'  RMSE: {rmse_value:.6f}')
        print(f'  R²: {r2:.6f}')
        print(f'  Pearson: {pearson_value:.6f}')
        print(f'  Spearman: {spearman_value:.6f}')
        
        # Validation
        mse, rmse, ci, r2, pearson, spearman = test(model, valid_dataset_load, is_valid=True)
        
        print(f'\nValidation Results:')
        print(f'  MSE: {mse:.6f}')
        print(f'  RMSE: {rmse:.6f}')
        print(f'  R²: {r2:.6f}')
        print(f'  Pearson: {pearson:.6f}')
        print(f'  Spearman: {spearman:.6f}')
        
        # Early stopping
        if mse < best_valid_mse:
            patience = 0
            best_valid_mse = mse
            torch.save(model.state_dict(), model_save_path)
            print(f'\n✓ Best model updated! MSE improved to {mse:.6f}')
            print(f'  Model saved to: {model_save_path}')
        else:
            patience += 1
            print(f'\n✗ No improvement. Patience: {patience}/{hp.max_patience}')
            if patience > hp.max_patience:
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                break
    
    # Save training log
    log_dir = f"./log/All-{hp.dataset}-{hp.current_time}-train.csv"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    
    with open(log_dir, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mse", "rmse", "ci", "r2", "pearson", "spearman"])
        for r in train_log:
            writer.writerow(r)
    
    print(f"\n{'='*80}")
    print(f"Training Completed!")
    print(f"{'='*80}")
    print(f"Best validation MSE: {best_valid_mse:.6f}")
    print(f"Model saved to: {model_save_path}")
    print(f"Training log saved to: {log_dir}")
    print(f"{'='*80}\n")
