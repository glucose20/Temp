import os
import random
import sys
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
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    
    hp = HyperParameter()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"Dataset-{hp.dataset}-{hp.running_set}") 
    print(f"Pretrain-{hp.mol2vec_dir}-{hp.protvec_dir}")
    fold_metrics = {'mse':[], 'rmse':[], 'ci':[], 'r2':[], 'pearson':[], 'spearman':[]}
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)    # TODO change coldstart v2
    
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    
    for fold_i in range(hp.kfold):                               
        train_dir = os.path.join(dataset_root, f'fold_{fold_i}_train.csv')
        valid_dir = os.path.join(dataset_root, f'fold_{fold_i}_valid.csv')
        test_dir = os.path.join(dataset_root, f'fold_{fold_i}_test.csv')                   
        train_set = CustomDataSet(pd.read_csv(train_dir, sep=','), hp)
        valid_set = CustomDataSet(pd.read_csv(valid_dir, sep=','), hp)
        test_set = CustomDataSet(pd.read_csv(test_dir, sep=','), hp)
        train_dataset_load = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
        valid_dataset_load = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
        test_dataset_load = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
        print("load dataset finished")
    
        model = nn.DataParallel(LLMDTA(hp, device))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
        criterion = F.mse_loss
    
        train_log = []     
        best_valid_mse = 10  
        patience = 0    
        model_fromTrain = f'./savemodel/{hp.dataset}-{hp.running_set}-fold{fold_i}-{hp.current_time}.pth'
                 
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
            
            # valid
            mse, rmse, ci, r2, pearson, spearman = test(model, valid_dataset_load, is_valid=True)   
            print(f'Valid at fold-{fold_i}: mse-{mse}') 
            # Early stop        
            if mse < best_valid_mse :
                patience = 0
                best_valid_mse = mse
                # save model
                torch.save(model.state_dict(), model_fromTrain)
                print(f'Update best_mse, Valid at fold-{fold_i} epoch-{epoch}: mse-{mse}, rmse-{rmse}, ci-{ci}, r2-{r2}, pearson-{pearson}, spearman-{spearman}')
            else:
                patience += 1
                if patience > hp.max_patience:
                    print(f'Traing stop at epoch-{epoch}, model save at-{model_fromTrain}')
                    break   
                 
        log_dir = f"./log/{hp.current_time}-{hp.dataset}-{hp.running_set}-fold{fold_i}.csv"
        with open(log_dir, "w+")as f:
            writer = csv.writer(f)
            writer.writerow(["mse", "rmse",  "ci", "r2", 'pearson', 'spearman'])
            for r in train_log:
                writer.writerow(r)
        print(f'Save log over at {log_dir}')

        # Test
        predModel = nn.DataParallel(LLMDTA(hp, device))
        predModel.load_state_dict(torch.load(model_fromTrain))
        predModel = predModel.to(device)    
        mse, rmse, ci, r2, pearson, spearman = test(predModel, test_dataset_load, is_valid=False)
        print(f'Test at fold-{fold_i}, mse: {mse}, rmse: {rmse}, ci: {ci}, r2: {r2}, pearson: {pearson}, spearman: {spearman}\n')
        fold_metrics['mse'].append(mse)
        fold_metrics['rmse'].append(rmse)
        fold_metrics['ci'].append(ci)
        fold_metrics['r2'].append(r2)
        fold_metrics['pearson'].append(pearson)
        fold_metrics['spearman'].append(spearman)
        
    # save training log
    fold_test_metrics = pd.DataFrame(fold_metrics)    
    fold_test_metrics.to_csv(f'./log/Test-{hp.dataset}-{hp.running_set}-{hp.current_time}.csv', index=False)    
    mean_values = fold_test_metrics.mean()
    variance_values = fold_test_metrics.var()    
    print(f"Dataset-{hp.dataset}-{hp.running_set}")
    print(f"Mean Values:{pd.concat([mean_values, variance_values], axis=1)}") 