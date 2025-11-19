import torch
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np 
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence
from rdkit import Chem
import os

# ============================================================================
# ESM-C (Cambrian) - New EvolutionaryScale Package
# ============================================================================
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    ESMC_AVAILABLE = True
except ImportError:
    ESMC_AVAILABLE = False
    print("Warning: ESM-C not available. Install with: pip install esm")

# ============================================================================
# ESM2 (Legacy) - Facebook Research Package
# ============================================================================
try:
    import esm
    ESM2_AVAILABLE = True
except ImportError:
    ESM2_AVAILABLE = False
    print("Warning: ESM2 not available. Install with: pip install fair-esm")


def get_esmc_pretrain(df_dir, db_name, model_name="esmc_300m", sep=' ', header=None, 
                      col_names=['drug_id', 'prot_id', 'drug_smile', 'prot_seq', 'label'], 
                      is_save=True, device='cuda'):
    """
    Extract protein embeddings using ESM-C (Cambrian) models.
    
    Args:
        df_dir: Path to input CSV file
        db_name: Database name for output file
        model_name: ESM-C model variant ('esmc_300m', 'esmc_600m', 'esmc_6b')
        sep: CSV separator
        header: CSV header row
        col_names: Column names
        is_save: Whether to save embeddings
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with 'dataset', 'vec_dict', 'mat_dict', 'length_dict'
    """
    if not ESMC_AVAILABLE:
        raise ImportError("ESM-C not available. Install with: pip install esm")
    
    # Check for cached embeddings
    file_path = f'./data/{db_name}_esmc_pretrain.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Load ESM-C pretrained feature: {file_path}")
            return data
    
    print(f"Loading ESM-C model: {model_name}...")
    model = ESMC.from_pretrained(model_name).to(device)
    model.eval()
    
    # Load and prepare data
    df = pd.read_csv(df_dir, sep=sep, header=header)
    df.columns = col_names
    df.drop_duplicates(subset='prot_id', inplace=True)
    prot_ids = df['prot_id'].tolist()
    prot_seqs = df['prot_seq'].tolist()
    
    emb_dict = {}
    emb_mat_dict = {}
    length_target = {}
    
    print(f"Extracting ESM-C embeddings for {len(prot_ids)} proteins...")
    for idx in tqdm(range(len(prot_ids))):
        prot_id = str(prot_ids[idx])
        # ESM-C supports up to 2048 tokens (vs ESM2's 1022)
        seq = prot_seqs[idx][:2048]
        length_target[prot_id] = len(seq)
        
        try:
            # Create protein object
            protein = ESMProtein(sequence=seq)
            
            # Encode protein
            protein_tensor = model.encode(protein)
            
            # Get embeddings
            with torch.no_grad():
                logits_output = model.logits(
                    protein_tensor,
                    LogitsConfig(return_embeddings=True)
                )
            
            # ESM-C returns embeddings shape: (seq_len, d_model)
            # d_model: 960 (300M), 1152 (600M), 2560 (6B)
            embeddings = logits_output.embeddings.cpu().numpy()
            
            # Store per-sequence representation (mean over sequence)
            emb_dict[prot_id] = embeddings.mean(axis=0)
            
            # Store full sequence embeddings matrix
            emb_mat_dict[prot_id] = embeddings
            
        except Exception as e:
            print(f"Error processing {prot_id}: {e}")
            # Use zero embeddings as fallback
            dim = 960 if model_name == "esmc_300m" else (1152 if model_name == "esmc_600m" else 2560)
            emb_dict[prot_id] = np.zeros(dim)
            emb_mat_dict[prot_id] = np.zeros((len(seq), dim))
    
    # Save embeddings
    dump_data = {
        "dataset": db_name,
        "vec_dict": emb_dict,
        "mat_dict": emb_mat_dict,
        "length_dict": length_target,
        "model": model_name  # Store model info
    }
    
    if is_save:
        os.makedirs('./data', exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(dump_data, f)
        print(f"Saved ESM-C embeddings to: {file_path}")
    
    return dump_data


def get_esm_pretrain(df_dir, db_name, sep=' ', header=None, 
                     col_names=['drug_id', 'prot_id', 'drug_smile', 'prot_seq', 'label'], 
                     is_save=True, use_esmc=True, esmc_model="esmc_300m"):
    """
    Main function to extract protein embeddings.
    Automatically uses ESM-C if available, otherwise falls back to ESM2.
    
    Args:
        use_esmc: If True, use ESM-C; if False, use ESM2 (legacy)
        esmc_model: ESM-C model variant ('esmc_300m', 'esmc_600m', 'esmc_6b')
    """
    
    # Try ESM-C first if requested and available
    if use_esmc and ESMC_AVAILABLE:
        print("Using ESM-C (Cambrian) for protein embeddings...")
        return get_esmc_pretrain(df_dir, db_name, model_name=esmc_model, 
                                sep=sep, header=header, col_names=col_names, 
                                is_save=is_save)
    
    # Fallback to ESM2 (Legacy)
    if not ESM2_AVAILABLE:
        raise ImportError("Neither ESM-C nor ESM2 available. Install one with: pip install esm OR pip install fair-esm")
    
    print("Using ESM2 (Legacy) for protein embeddings...")
    
    # ========== ESM2 Legacy Code (Commented but functional) ==========
    file_path = f'./data/{db_name}_esm_pretrain.pkl'
    if os.path.exists(file_path):          
        with open(file_path, 'rb+') as f:
            data = pickle.load(f)
            print(f"Load ESM2 pretrained feature: {file_path}.")
            return data
        
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results 
    
    df = pd.read_csv(df_dir, sep=sep)
    df.columns = col_names
    df.drop_duplicates(subset='prot_id', inplace=True)
    prot_ids = df['prot_id'].tolist()
    prot_seqs = df['prot_seq'].tolist()
    data = []
    prot_size = len(prot_ids)
    for i in range(prot_size):
        seq_len = min(len(prot_seqs[i]),1022)
        data.append((prot_ids[i], prot_seqs[i][:seq_len]))
    
    emb_dict = {}
    emb_mat_dict = {}
    length_target = {}

    for d in tqdm(data):
        prot_id = d[0]
        batch_labels, batch_strs, batch_tokens = batch_converter([d])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33].numpy()

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

        emb_dict[prot_id] = sequence_representations[0]
        emb_mat_dict[prot_id] = token_representations[0]
        length_target[prot_id] = len(d[1])
    
    # save
    dump_data = {
        "dataset": db_name,
        "vec_dict": emb_dict,
        "mat_dict": emb_mat_dict,
        "length_dict": length_target
    }   
    if is_save:
        with open(f'./data/{db_name}_esm_pretrain.pkl', 'wb+') as f:
            pickle.dump(dump_data, f)    
    return dump_data


def get_mol2vec(word2vec_pth, df_dir, db_name, sep=' ', col_names=['drug_id', 'prot_id', 'drug_smile', 'prot_seq', 'label'], embedding_dimension=300, is_debug=False, is_save=True):    
    file_path = f'./data/{db_name}_mol_pretrain.pkl'
    if os.path.exists(file_path):          
        with open(file_path, 'rb+') as f:
            data = pickle.load(f)
            print(f"Load pretrained feature: {file_path}.")
            return data
        
    mol2vec_model = word2vec.Word2Vec.load(word2vec_pth)    
    df = pd.read_csv(df_dir, sep=sep)
    df.columns = col_names
    df.drop_duplicates(subset='drug_id', inplace=True)    
    drug_ids = df['drug_id'].tolist()
    drug_seqs = df['drug_smile'].tolist()
    
    emb_dict = {}
    emb_mat_dict = {}
    length_dict = {}
    
    percent_unknown = []
    bad_mol = 0
    
    # get pretrain feature
    for idx in tqdm(range(len(drug_ids))):
        flag = 0
        mol_miss_words = 0
        
        drug_id = str(drug_ids[idx])
        molecule = Chem.MolFromSmiles(drug_seqs[idx])
        length_dict
        try:
            # Get fingerprint from molecule
            sub_structures = mol2alt_sentence(molecule, 2)
        except Exception as e: 
            if is_debug: 
                print (e)
            percent_unknown.append(100)
            continue    
        
        # 存储该分子的子结构特征矩阵
        emb_mat = np.zeros((len(sub_structures), embedding_dimension))
        length_dict[drug_id] = len(sub_structures)
        
        # 遍历分子中每个子结构
        for i, sub in enumerate(sub_structures):
            # Check to see if substructure exists
            try:
                emb_dict[drug_id] = emb_dict.get(drug_id, np.zeros(embedding_dimension)) + mol2vec_model.wv[sub]  
                emb_mat[i] = mol2vec_model.wv[sub]  
            # If not, replace with UNK (unknown)
            except Exception as e:
                if is_debug : 
                    print ("Sub structure not found")
                    print (e)
                emb_dict[drug_id] = emb_dict.get(drug_id, np.zeros(embedding_dimension)) + mol2vec_model.wv['UNK']
                emb_mat[i] = mol2vec_model.wv['UNK']                
                flag = 1
                mol_miss_words = mol_miss_words + 1        
        emb_mat_dict[drug_id] = emb_mat
        
        percent_unknown.append((mol_miss_words / len(sub_structures)) * 100)
        if flag == 1:
            bad_mol = bad_mol + 1
            
    print(f'All Bad Mol: {bad_mol}, Avg Miss Rate: {sum(percent_unknown)/len(percent_unknown)}%')        
    dump_data = {
        "dataset": db_name,
        "vec_dict": emb_dict,
        "mat_dict": emb_mat_dict,
        "length_dict": length_dict
    }   
    if is_save: 
        with open(f'./data/{db_name}_mol_pretrain.pkl', 'wb+') as f:
            pickle.dump(dump_data, f)    
    return dump_data


def getPairs(drug_dir, prot_dir, sep, d_col_names, p_col_names):
    d_df = pd.read_csv(drug_dir, sep=sep)
    d_df.columns = d_col_names
    p_df = pd.read_csv(prot_dir, sep=sep)
    p_df.columns = p_col_names
    
    pair_dict = {'drug_id':[], 'prot_id':[], 'drug_smile':[], 'prot_seq':[]}
    for i, row_d in d_df.iterrows():
        for j, row_p in p_df.iterrows():
            pair_dict['drug_id'].append(row_d['drug_id'])
            pair_dict['prot_id'].append(row_p['prot_id'])
            pair_dict['drug_smile'].append(row_d['drug_smile'])
            pair_dict['prot_seq'].append(row_p['prot_seq'])
            
    pair_df = pd.DataFrame(pair_dict, index=None)  
    return pair_df

