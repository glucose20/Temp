"""
Test trained kd_bind model with a single drug-protein pair
"""

import os
import sys
import pickle
import torch
import pandas as pd
import numpy as np

# Add code directory to path
sys.path.append('./code')

from LLMDTA import LLMDTA
from hyperparameter_full import HyperParameter


def load_pretrain_data(dataset_name='kd_bind'):
    """Load pretrained drug and protein embeddings"""
    mol2vec_path = f'./data/{dataset_name}/{dataset_name}_drug_pretrain.pkl'
    protvec_path = f'./data/{dataset_name}/{dataset_name}_esm_pretrain.pkl'
    
    print(f"Loading pretrained features from:")
    print(f"  Drug embeddings: {mol2vec_path}")
    print(f"  Protein embeddings: {protvec_path}")
    
    with open(mol2vec_path, 'rb') as f:
        mol2vec_dict = pickle.load(f)
    with open(protvec_path, 'rb') as f:
        protvec_dict = pickle.load(f)
    
    print(f"✓ Loaded {len(mol2vec_dict['vec_dict'])} drug embeddings")
    print(f"✓ Loaded {len(protvec_dict['vec_dict'])} protein embeddings")
    
    return mol2vec_dict, protvec_dict


def prepare_sample(drug_id, prot_id, mol2vec_dict, protvec_dict, hp, device):
    """Prepare tensors for a single drug-protein pair"""
    
    # Convert to string to match embedding keys
    drug_id = str(drug_id)
    prot_id = str(prot_id)
    
    # Get embeddings
    drug_vec = mol2vec_dict["vec_dict"][drug_id]
    prot_vec = protvec_dict["vec_dict"][prot_id]
    drug_mat = mol2vec_dict["mat_dict"][drug_id]
    prot_mat = protvec_dict["mat_dict"][prot_id]
    
    # Handle vec that might be 2D
    if drug_vec.ndim > 1:
        drug_vec = drug_vec.mean(axis=0)
    if prot_vec.ndim > 1:
        prot_vec = prot_vec.mean(axis=0)
    
    # Get lengths
    drug_len = mol2vec_dict["length_dict"].get(drug_id, drug_mat.shape[0])
    prot_len = protvec_dict["length_dict"].get(prot_id, prot_mat.shape[0])
    
    # Pad drug matrix
    drug_max_len = hp.drug_max_len
    if drug_mat.shape[0] < drug_max_len:
        pad_len = drug_max_len - drug_mat.shape[0]
        drug_mat = np.vstack([drug_mat, np.zeros((pad_len, drug_mat.shape[1]))])
    else:
        drug_mat = drug_mat[:drug_max_len]
    
    # Pad protein matrix
    prot_max_len = hp.prot_max_len
    if prot_mat.shape[0] < prot_max_len:
        pad_len = prot_max_len - prot_mat.shape[0]
        prot_mat = np.vstack([prot_mat, np.zeros((pad_len, prot_mat.shape[1]))])
    else:
        prot_mat = prot_mat[:prot_max_len]
    
    # Create masks
    drug_mask = np.array([1] * min(drug_len, drug_max_len) + [0] * max(0, drug_max_len - drug_len))
    prot_mask = np.array([1] * min(prot_len, prot_max_len) + [0] * max(0, prot_max_len - prot_len))
    
    # Convert to tensors and add batch dimension
    drug_vec_tensor = torch.FloatTensor(drug_vec).unsqueeze(0).to(device)
    prot_vec_tensor = torch.FloatTensor(prot_vec).unsqueeze(0).to(device)
    drug_mat_tensor = torch.FloatTensor(drug_mat).unsqueeze(0).to(device)
    prot_mat_tensor = torch.FloatTensor(prot_mat).unsqueeze(0).to(device)
    drug_mask_tensor = torch.FloatTensor(drug_mask).unsqueeze(0).to(device)
    prot_mask_tensor = torch.FloatTensor(prot_mask).unsqueeze(0).to(device)
    
    return (drug_vec_tensor, prot_vec_tensor, drug_mat_tensor, prot_mat_tensor,
            drug_mask_tensor, prot_mask_tensor)


def test_single_sample(model_path, drug_id, prot_id, dataset_name='kd_bind', actual_affinity=None):
    """
    Test model with a single drug-protein pair
    
    Args:
        model_path: Path to trained model .pth file
        drug_id: Drug ID (will be converted to string)
        prot_id: Protein ID (will be converted to string)
        dataset_name: Dataset name for loading embeddings
        actual_affinity: Optional actual pKd value for comparison
    """
    
    print(f"{'='*80}")
    print(f"Testing Model: {model_path}")
    print(f"{'='*80}\n")
    
    # Load hyperparameters
    hp = HyperParameter()
    hp.dataset = dataset_name
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load pretrained embeddings
    mol2vec_dict, protvec_dict = load_pretrain_data(dataset_name)
    
    # Convert IDs to string for lookup
    drug_id_str = str(drug_id)
    prot_id_str = str(prot_id)
    
    # Check if IDs exist in embeddings
    if drug_id_str not in mol2vec_dict['vec_dict']:
        print(f"❌ Drug ID '{drug_id_str}' not found in embeddings!")
        return
    if prot_id_str not in protvec_dict['vec_dict']:
        print(f"❌ Protein ID '{prot_id_str}' not found in embeddings!")
        return
    
    print(f"Testing drug-protein pair:")
    print(f"  Drug ID:    {drug_id_str}")
    print(f"  Protein ID: {prot_id_str}")
    if actual_affinity is not None:
        print(f"  Actual pKd: {actual_affinity:.3f}")
    print()
    
    # Load model
    print("Loading model...")
    model = LLMDTA(hp, device).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check checkpoint structure and load accordingly
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # Checkpoint is the model state dict directly
        state_dict = checkpoint
    
    # Remove 'module.' prefix if exists (from DataParallel)
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded\n")
    
    # Prepare input
    print("Preparing input...")
    drug_vec, prot_vec, drug_mat, prot_mat, drug_mask, prot_mask = prepare_sample(
        drug_id, prot_id, mol2vec_dict, protvec_dict, hp, device
    )
    print("✓ Input prepared\n")
    
    # Run prediction
    print("Running prediction...")
    with torch.no_grad():
        # CRITICAL: Correct argument order: drug_vec, drug_mat, drug_mask, prot_vec, prot_mat, prot_mask
        prediction = model(drug_vec, drug_mat, drug_mask, prot_vec, prot_mat, prot_mask)
        predicted_affinity = prediction.item()
    
    print(f"{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Predicted pKd: {predicted_affinity:.3f}")
    
    if actual_affinity is not None:
        error = abs(predicted_affinity - actual_affinity)
        print(f"Actual pKd:    {actual_affinity:.3f}")
        print(f"Error:         {error:.3f}")
        print(f"Relative error: {error/actual_affinity*100:.1f}%")
    
    # Interpret binding affinity
    print(f"\nBinding strength interpretation:")
    if predicted_affinity >= 8.0:
        print(f"  🔥 Very High Affinity (pKd ≥ 8.0)")
    elif predicted_affinity >= 7.0:
        print(f"  ✅ High Affinity (7.0 ≤ pKd < 8.0)")
    elif predicted_affinity >= 6.0:
        print(f"  ⚠️  Moderate Affinity (6.0 ≤ pKd < 7.0)")
    else:
        print(f"  ❌ Low Affinity (pKd < 6.0)")
    
    print(f"{'='*80}\n")
    
    return predicted_affinity


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained model with single sample')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model .pth file')
    parser.add_argument('--drug-id', type=str, required=True,
                       help='Drug ID')
    parser.add_argument('--prot-id', type=str, required=True,
                       help='Protein ID')
    parser.add_argument('--dataset', type=str, default='kd_bind',
                       help='Dataset name (default: kd_bind)')
    parser.add_argument('--actual', type=float, default=None,
                       help='Optional actual pKd value for comparison')
    
    args = parser.parse_args()
    
    test_single_sample(
        model_path=args.model,
        drug_id=args.drug_id,
        prot_id=args.prot_id,
        dataset_name=args.dataset,
        actual_affinity=args.actual
    )
