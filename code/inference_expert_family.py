import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LLMDTA import LLMDTA
from MyDataset import CustomDataSet, my_collate_fn
from hyperparameter import HyperParameter
from sklearn.metrics import r2_score
from math import sqrt
from scipy import stats

# Protein family classification logic
RTK_PROTEINS = [
    'EGFR', 'ERBB', 'HER2', 'HER3', 'HER4', 'VEGFR', 'KDR', 'FLT1', 'FLT4',
    'PDGFR', 'PDGFRA', 'PDGFRB', 'FGFR', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4',
    'INSR', 'IGF1R', 'INSRR', 'KIT', 'FLT3', 'CSF1R', 'NTRK', 'TRKA', 'TRKB', 'TRKC',
    'MET', 'RON', 'AXL', 'MER', 'TYRO3', 'RET', 'ALK', 'ROS1', 'LTK',
    'DDR1', 'DDR2', 'EPHA', 'EPHB', 'TIE1', 'TIE2', 'TEK'
]

# Non-receptor Tyrosine Kinases
TK_PROTEINS = [
    'SRC', 'FYN', 'YES', 'LCK', 'LYN', 'HCK', 'FGR', 'BLK', 'YRK',
    'ABL', 'ABL1', 'ABL2', 'ARG', 'BCR-ABL',
    'BTK', 'ITK', 'TEC', 'BMX', 'TXK', 'RLK',
    'SYK', 'ZAP70', 'ZAP',
    'JAK', 'JAK1', 'JAK2', 'JAK3', 'TYK2',
    'CSK', 'CTK', 'MATK',
    'FAK', 'PYK2', 'PTK2',
    'ACK', 'TNK1', 'TNK2',
    'FES', 'FER',
    'BRK', 'FRK', 'SRMS', 'PTK6'
]

# Serine/Threonine Kinases
STK_PROTEINS = [
    'CDK', 'CDK1', 'CDK2', 'CDK3', 'CDK4', 'CDK5', 'CDK6', 'CDK7', 'CDK8', 'CDK9',
    'MAPK', 'ERK', 'ERK1', 'ERK2', 'JNK', 'JNK1', 'JNK2', 'JNK3', 'p38',
    'RAF', 'ARAF', 'BRAF', 'CRAF', 'RAF1', 'KSR',
    'MEK', 'MEK1', 'MEK2', 'MKK', 'MAP2K',
    'AKT', 'AKT1', 'AKT2', 'AKT3', 'PKB',
    'PKA', 'PRKACA', 'PRKACB', 'PKC', 'PKG', 'ROCK', 'ROCK1', 'ROCK2',
    'GSK', 'GSK3', 'GSK3A', 'GSK3B',
    'CK1', 'CK2', 'CSNK', 'CSNK1', 'CSNK2',
    'PLK', 'PLK1', 'PLK2', 'PLK3', 'PLK4',
    'AURK', 'AURKA', 'AURKB', 'AURKC',
    'CHK', 'CHK1', 'CHK2', 'CHEK',
    'DAPK', 'DAPK1', 'DAPK2', 'DAPK3',
    'CAMK', 'CAMK1', 'CAMK2', 'CAMK4', 'CAMKK',
    'AMPK', 'PRKAA', 'STK11', 'LKB1',
    'DYRK', 'CLK', 'PIM', 'PAK', 'MINK', 'TNIK',
    'ASK', 'TAK', 'MLK', 'MEKK', 'MAP3K',
    'RIPK', 'IRAK', 'IKK', 'TBK1',
    'WEE', 'MYT1', 'TTK', 'BUB', 'NEK',
    'LATS', 'MST', 'STK', 'MAST', 'MARK', 'BRSK', 'NUAK'
]


# GPCR-related 
GPCR_KEYWORDS = ['GPCR', 'receptor', 'GPR', 'ADORA', 'ADRB', 'DRD', 'HTR', 'CHRM']

# Lipid kinases
LIPID_KINASES = ['PI3K', 'PIK3', 'PIKK', 'ATM', 'ATR', 'DNAPK', 'mTOR', 'FRAP']

def classify_protein(prot_name):
    """Classify protein into family based on name."""
    name_upper = prot_name.upper()
    for rtk in RTK_PROTEINS:
        if rtk in name_upper:
            return 'RTK', 'Receptor Tyrosine Kinase'
    for tk in TK_PROTEINS:
        if tk in name_upper:
            return 'TK', 'Tyrosine Kinase (non-receptor)'
    for stk in STK_PROTEINS:
        if stk in name_upper:
            return 'STK', 'Serine/Threonine Kinase'
    for lk in LIPID_KINASES:
        if lk in name_upper:
            return 'LK', 'Lipid/Atypical Kinase'
    for gpcr in GPCR_KEYWORDS:
        if gpcr in name_upper:
            return 'GPCR', 'GPCR'
    return 'Other', 'Other'

def regression_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    mse = ((label - pred) ** 2).mean(axis=0)
    rmse = sqrt(mse)
    ci = cindex_score(label, pred)
    r2 = r2_score(label, pred)
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(mse, 6), round(rmse, 6), round(ci, 6), round(r2, 6), round(pearson, 6), round(spearman, 6)

def cindex_score(y, p):
    sum_m = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if y[i] > y[j]:
                pair += 1
                sum_m += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    return sum_m / pair if pair != 0 else 0

def test(models, dataloader, test_df):
    """Run inference for all models and track metrics."""
    results = []
    batch_idx = 0

    for batch_data in dataloader:
        mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data
        # prot_ids = prot_info['prot_id'].values
        with torch.no_grad():
            batch_preds = []
            for model in models:
                pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
                batch_preds.append(pred.cpu().numpy().reshape(-1))
            batch_preds = np.array(batch_preds)  # Shape: (num_models, batch_size)

            # Get corresponding samples from test_df
            batch_size = batch_preds.shape[1]
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + batch_size, len(test_df))

            sample_idx = start_idx + i
            if sample_idx >= len(test_df):
                break
                
            row = test_df.iloc[sample_idx]
            prot_id = row['prot_id']
            drug_id = row['drug_id']
            
            family, subfamily = classify_protein(prot_id)

            for i, prot_id in enumerate(prot_ids):
                family, subfamily = classify_protein(prot_id)
                for model_idx, pred in enumerate(batch_preds[:, i]):
                    results.append({
                        'protein': prot_id,
                        'family': family,
                        'subfamily': subfamily,
                        'expert': model_idx,
                        'prediction': pred,
                        'label': affinity[i].item()
                    })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Inference for LLMDTA model with fixed experts")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--cuda', type=str, default=None, help="CUDA device ID (e.g., '0', '1')")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    # Load hyperparameters
    hp = HyperParameter()
    if args.cuda is not None:
        hp.cuda = args.cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    print("Loading DAVIS warm test set...")
    test_df = pd.read_csv(hp.test_dir)
    mol2vec_dict = pd.read_pickle(hp.mol2vec_dir)
    protvec_dict = pd.read_pickle(hp.protvec_dir)
    test_set = CustomDataSet(test_df, hp)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: my_collate_fn(x, device, hp, test_df, test_df, mol2vec_dict, protvec_dict),
    )
    print(f"Loaded {len(test_set)} test samples")

    # Initialize models with different fixed experts
    models = []
    for fixed_expert in range(4):  # Assuming 4 experts
        print(f"Initializing model with fixed expert {fixed_expert}...")
        model = nn.DataParallel(LLMDTA(hp, device))
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.module.gating.fixed_expert = fixed_expert
        model = model.to(device)
        models.append(model)

    # Run inference
    print("Running inference...")
    results = test(models, test_loader, test_df)

    # Save results
    results_file = "./inference_results.csv"
    results.to_csv(results_file, index=False)
    print(f"Inference results saved to: {results_file}")