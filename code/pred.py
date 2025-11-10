import os
import numpy as np 
import pandas as pd 
import pickle
import random
import esm
from tqdm import tqdm
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence
from rdkit import Chem
from LLMDTA import LLMDTA as LLMDTA
from hyperparameter4pred import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn4pred
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datetime import datetime
import utils


def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)
    
    
def test(model, dataloader):
    model.eval()
    preds = []    
    for batch_data in tqdm(dataloader):
        mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask = batch_data
        with torch.no_grad():
            pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()            

    preds = np.array(preds)
    return preds


SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(4)

hp = HyperParameter()
os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sep = hp.sep
pred_task_name = hp.pred_dataset
word2vec_pth = hp.word2vec_pth
if os.path.exists(hp.pred_pair_pth):    
    pair_dir = hp.pred_pair_pth
    col_name = hp.pair_col_name
    pred_pairs = pd.read_csv(pair_dir)    
    mol2vec_dict = utils.get_mol2vec(word2vec_pth, pair_dir, pred_task_name, sep=sep, col_names=col_name)
    protvec_dict = utils.get_esm_pretrain(pair_dir, pred_task_name, sep=sep, col_names=col_name)
else:        
    drug_dir = hp.pred_drug_dir
    prot_dir = hp.pred_prot_dir
    d_col_names=hp.d_col_name
    p_col_names=hp.p_col_name
    pred_pairs = utils.getPairs(drug_dir=drug_dir, prot_dir=prot_dir, sep=sep, d_col_names=d_col_names, p_col_names=p_col_names)
    mol2vec_dict = utils.get_mol2vec(word2vec_pth, drug_dir, pred_task_name, sep=sep, col_names=d_col_names)
    protvec_dict = utils.get_esm_pretrain(prot_dir, pred_task_name, sep=sep, col_names=p_col_names)

# pretrained model pth 
model_fromTrain = hp.model_fromTrain

pred_set = CustomDataSet(pred_pairs, hp)
test_dataset_load = DataLoader(pred_set, batch_size=1, 
                            shuffle=False, drop_last=False, num_workers=0, 
                            collate_fn=lambda x: my_collate_fn4pred(x, device, hp, mol2vec_dict, protvec_dict))
print('Load pred data over.')

# Test
predModel = nn.DataParallel(LLMDTA(hp, device))  
predModel.load_state_dict(torch.load(model_fromTrain, map_location=device))
print('Load model over.')
predModel = predModel.to(device)    
pred = test(predModel, test_dataset_load)

output_df = pred_pairs[['drug_id', 'prot_id']]    
output_df.loc[:, 'pred'] = pd.Series([round(p, 3) for p in pred])
date_str = datetime.now().strftime('%b%d_%H-%M-%S')
save_path = f'./Pred_{pred_task_name}_{date_str}.csv'
output_df.to_csv(f'{save_path}', index=False)
print(f'Predict over {save_path}.')
