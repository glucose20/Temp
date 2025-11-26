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
            pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist()

    preds = np.array(preds)
    labels = np.array(labels)
    mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(labels, preds, is_valid)
    return mse_value, rmse_value, ci, r2, pearson_value, spearman_value

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

    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    print(f"=" * 60)
    print(f"Training Fold {fold_i}/{hp.kfold-1}")
    print(f"Dataset: {hp.dataset}-{hp.running_set}") 
    print(f"ESM Model: {'ESM-C-' + hp.esmc_model if hp.use_esmc else 'ESM2'} (dim={hp.protvec_dim})")
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
        }
        
        esm_name = f"esmc-{hp.esmc_model}" if hp.use_esmc else "esm2"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{hp.dataset}-{hp.running_set}-{esm_name}-fold{fold_i}",
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

    model = nn.DataParallel(LLMDTA(hp, device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
    criterion = F.mse_loss

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
        # trainning
        model.train()
        pred = []
        label = []             
        for batch_data in train_dataset_load:
            mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask, affinity = batch_data                    
            predictions = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
            pred = pred + predictions.cpu().detach().numpy().reshape(-1).tolist()
            label = label + affinity.cpu().detach().numpy().reshape(-1).tolist()            
            
            loss = criterion(predictions.squeeze(), affinity)
            loss.backward()                
            optimizer.step()
            optimizer.zero_grad()                                             
        pred = np.array(pred)
        label= np.array(label)
        mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(pred, label)
        train_log.append([mse_value, rmse_value, ci, r2, pearson_value, spearman_value])
        print(f'Traing Log at fold-{fold_i} epoch-{epoch}: mse-{mse_value}, rmse-{rmse_value}, r2-{r2}')
        
        # Log training metrics to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/mse': mse_value,
                'train/rmse': rmse_value,
                'train/ci': ci,
                'train/r2': r2,
                'train/pearson': pearson_value,
                'train/spearman': spearman_value,
            })
        
        # valid
        mse, rmse, ci, r2, pearson, spearman = test(model, valid_dataset_load, is_valid=True)   
        print(f'Valid at fold-{fold_i}: mse-{mse}')
        
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
                print(f'Traing stop at epoch-{epoch}, model save at-{model_fromTrain}')
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