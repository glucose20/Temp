import torch
from torch.utils.data import Dataset
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence
from rdkit import Chem
import pickle
import numpy as np

def batch2tensor(batch_data, device):
    mol_vec, mol_mat, mol_mask, prot_vec, prot_mat, prot_mask, label = batch_data      
    mol_mat = mol_mat.to(device).to(torch.float32)
    mol_mask = mol_mask.to(device).to(torch.float32)
    prot_mat = prot_mat.to(device).to(torch.float32)
    prot_mask = prot_mask.to(device).to(torch.float32)
    mol_vec = mol_vec.to(device).to(torch.float32)
    prot_vec = prot_vec.to(device).to(torch.float32)
    label = label.to(device).to(torch.float32)
    return mol_vec, mol_mat, mol_mask, prot_vec, prot_mat, prot_mask, label

def matrix_pad(arr, max_len):   
    dim = arr.shape[-1]
    len = arr.shape[0]
    if len < max_len:            
        new_arr = np.zeros((max_len, dim))
        vec_mask = np.zeros((max_len))                            
        new_arr[:len] = arr
        vec_mask[:len] = 1
        return new_arr, vec_mask
    else:
        new_arr = arr[:max_len]
        vec_mask = np.ones((max_len))  
        return new_arr, vec_mask
    
def my_collate_fn(batch_data, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, isEsm=False):
    batch_size = len(batch_data)
    drug_max = hp.drug_max_len
    drug_substruc_max = hp.substructure_max_len
    protein_max = hp.prot_max_len
    mol2vec_dim = hp.mol2vec_dim
    protvec_dim = hp.protvec_dim
    
    # Mat for pretrain feat
    b_drug_vec = torch.zeros((batch_size, mol2vec_dim), dtype=torch.float32)
    b_prot_vec = torch.zeros((batch_size, protvec_dim), dtype=torch.float32)
    b_drug_mask = torch.zeros((batch_size, drug_substruc_max), dtype=torch.float32)
    b_prot_mask = torch.zeros((batch_size, protein_max), dtype=torch.float32)    
    b_drug_mat = torch.zeros((batch_size, drug_substruc_max, mol2vec_dim), dtype=torch.float32)
    b_prot_mat = torch.zeros((batch_size, protein_max, protvec_dim), dtype=torch.float32)
    # label
    b_label = torch.zeros(batch_size)
    
    for i, pair in enumerate(batch_data):        
        drug_id, prot_id, label = pair[-3], pair[-2], pair[-1]
        drug_smiles = drug_df.loc[drug_df['drug_id'] == drug_id, 'drug_seq'].iloc[0]
        prot_seq = prot_df.loc[prot_df['prot_id'] == prot_id, 'prot_seq'].iloc[0]        
        drug_id = str(drug_id)
        prot_id = str(prot_id)
        drug_vec = mol2vec_dict["vec_dict"][drug_id]
        prot_vec = protvec_dict["vec_dict"][prot_id]
        drug_mat = mol2vec_dict["mat_dict"][drug_id]
        prot_mat = protvec_dict["mat_dict"][prot_id]
        drug_mat_pad, drug_mask = matrix_pad(drug_mat, drug_substruc_max)        
        prot_mat_pad, prot_mask = matrix_pad(prot_mat, protein_max)
        
        b_drug_vec[i] = torch.from_numpy(drug_vec)
        b_prot_vec[i] = torch.from_numpy(prot_vec)
        b_drug_mat[i] = torch.from_numpy(drug_mat_pad)
        b_drug_mask[i] = torch.from_numpy(drug_mask)
        b_prot_mat[i] = torch.from_numpy(prot_mat_pad)
        b_prot_mask[i] = torch.from_numpy(prot_mask)
        b_label[i] = label
        
    b_drug_vec = b_drug_vec.to(device)
    b_prot_vec = b_prot_vec.to(device)
    b_drug_mat = b_drug_mat.to(device)
    b_drug_mask = b_drug_mask.to(device)
    b_prot_mat = b_prot_mat.to(device)
    b_prot_mask = b_prot_mask.to(device)
    b_label = b_label.to(device)
    return b_drug_vec, b_prot_vec, b_drug_mat, b_drug_mask, b_prot_mat, b_prot_mask, b_label


def my_collate_fn4pred(batch_data, device, hp, mol2vec_dict, protvec_dict, isEsm=False):
    batch_size = len(batch_data)
    drug_max = hp.drug_max_len
    drug_substruc_max = hp.substructure_max_len
    protein_max = hp.prot_max_len
    mol2vec_dim = hp.mol2vec_dim
    protvec_dim = hp.protvec_dim
    
    # Mat for pretrain feat
    b_drug_vec = torch.zeros((batch_size, mol2vec_dim), dtype=torch.float32)
    b_prot_vec = torch.zeros((batch_size, protvec_dim), dtype=torch.float32)
    b_drug_mask = torch.zeros((batch_size, drug_substruc_max), dtype=torch.float32)
    b_prot_mask = torch.zeros((batch_size, protein_max), dtype=torch.float32)    
    b_drug_mat = torch.zeros((batch_size, drug_substruc_max, mol2vec_dim), dtype=torch.float32)
    b_prot_mat = torch.zeros((batch_size, protein_max, protvec_dim), dtype=torch.float32)
    # label
    b_label = torch.zeros(batch_size)
    
    for i, pair in enumerate(batch_data):        
        drug_id, prot_id = pair[0], pair[1]     
        drug_id = str(drug_id)
        prot_id = str(prot_id)
        drug_vec = mol2vec_dict["vec_dict"][drug_id]
        prot_vec = protvec_dict["vec_dict"][prot_id]
        drug_mat = mol2vec_dict["mat_dict"][drug_id]
        prot_mat = protvec_dict["mat_dict"][prot_id]
        drug_mat_pad, drug_mask = matrix_pad(drug_mat, drug_substruc_max)        
        prot_mat_pad, prot_mask = matrix_pad(prot_mat, protein_max)
        
        b_drug_vec[i] = torch.from_numpy(drug_vec)
        b_prot_vec[i] = torch.from_numpy(prot_vec)
        b_drug_mat[i] = torch.from_numpy(drug_mat_pad)
        b_drug_mask[i] = torch.from_numpy(drug_mask)
        b_prot_mat[i] = torch.from_numpy(prot_mat_pad)
        b_prot_mask[i] = torch.from_numpy(prot_mask)
        
    b_drug_vec = b_drug_vec.to(device)
    b_prot_vec = b_prot_vec.to(device)
    b_drug_mat = b_drug_mat.to(device)
    b_drug_mask = b_drug_mask.to(device)
    b_prot_mat = b_prot_mat.to(device)
    b_prot_mask = b_prot_mask.to(device)
    return b_drug_vec, b_prot_vec, b_drug_mat, b_drug_mask, b_prot_mat, b_prot_mask



def my_collate4predict(batch_data, device, hp, drug_feat_tool, prot_feat_tool):
    batch_size = len(batch_data)
    drug_max = hp.drug_max_len
    drug_substruc_max = hp.substructure_max_len
    protein_max = hp.prot_max_len
    mol2vec_dim = hp.mol2vec_dim
    protvec_dim = hp.protvec_dim
    
    # Mat for pretrain feat
    b_drug_vec = torch.zeros((batch_size, mol2vec_dim), dtype=torch.float32)
    b_prot_vec = torch.zeros((batch_size, protvec_dim), dtype=torch.float32)
    b_drug_mask = torch.zeros((batch_size, drug_substruc_max), dtype=torch.float32)
    b_prot_mask = torch.zeros((batch_size, protein_max), dtype=torch.float32)    
    b_drug_mat = torch.zeros((batch_size, drug_substruc_max, mol2vec_dim), dtype=torch.float32)
    b_prot_mat = torch.zeros((batch_size, protein_max, protvec_dim), dtype=torch.float32)

    
    for i, pair in enumerate(batch_data):                
        drug_smiles, prot_seq = pair[-2], pair[-1]               
        drug_vec, drug_mat, _ = drug_feat_tool.get(drug_smiles)        
        prot_vec, prot_mat, _ = prot_feat_tool.get(prot_seq)   # tensor     
        drug_mat_pad, drug_mask = matrix_pad(drug_mat, drug_substruc_max)        
        prot_mat_pad, prot_mask = matrix_pad(prot_mat, protein_max)
        
        b_drug_vec[i] = torch.from_numpy(drug_vec)
        b_prot_vec[i] = torch.from_numpy(prot_vec)
        b_drug_mat[i] = torch.from_numpy(drug_mat_pad)
        b_drug_mask[i] = torch.from_numpy(drug_mask)
        b_prot_mat[i] = torch.from_numpy(prot_mat_pad)
        b_prot_mask[i] = torch.from_numpy(prot_mask)        
        
    b_drug_vec = b_drug_vec.to(device)
    b_prot_vec = b_prot_vec.to(device)
    b_drug_mat = b_drug_mat.to(device)
    b_drug_mask = b_drug_mask.to(device)
    b_prot_mat = b_prot_mat.to(device)
    b_prot_mask = b_prot_mask.to(device)
        
    return b_drug_vec, b_prot_vec, b_drug_mat, b_drug_mask, b_prot_mat, b_prot_mask


class CustomDataSet(Dataset):
    def __init__(self, dataset, hp):    
        self.hp = hp
        self.dataset = dataset
        # self.dataset.columns = hp.dataset_columns
        
    def __getitem__(self, index):
        # drug_id, prot_id, label
        return self.dataset.iloc[index,:]

    def __len__(self):
        return len(self.dataset)