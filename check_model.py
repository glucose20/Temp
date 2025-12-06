"""
Script to check and validate a trained LLMDTA model
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import pickle

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from LLMDTA import LLMDTA
from hyperparameter_full import HyperParameter

def check_model(model_path):
    """Check if model file is valid"""
    print(f"\n{'='*80}")
    print(f"Checking Model: {model_path}")
    print(f"{'='*80}\n")
    
    # 1. Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found!")
        return False
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✓ Model file exists")
    print(f"  File size: {file_size:.2f} MB")
    
    # 2. Try to load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"\n✓ Model checkpoint loaded successfully")
        
        # Check checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"  Checkpoint keys: {list(checkpoint.keys())}")
            
            # Count parameters
            total_params = 0
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    params = value.numel()
                    total_params += params
                    print(f"  - {key}: {value.shape} ({params:,} params)")
            
            print(f"\n  Total parameters: {total_params:,}")
        else:
            print(f"  Checkpoint type: {type(checkpoint)}")
        
    except Exception as e:
        print(f"\n❌ ERROR loading checkpoint: {e}")
        return False
    
    # 3. Try to initialize model architecture
    try:
        print(f"\n{'='*80}")
        print("Testing Model Architecture")
        print(f"{'='*80}\n")
        
        hp = HyperParameter()
        hp.dataset = 'davis'
        device = torch.device('cpu')
        
        model = nn.DataParallel(LLMDTA(hp, device))
        model = model.to(device)
        
        print(f"✓ Model architecture initialized")
        
        # Count model parameters
        model_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {model_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"\n❌ ERROR initializing model: {e}")
        return False
    
    # 4. Try to load weights into model
    try:
        print(f"\n{'='*80}")
        print("Loading Weights into Model")
        print(f"{'='*80}\n")
        
        model.load_state_dict(checkpoint)
        print(f"✓ Weights loaded successfully into model")
        
        # Set to eval mode
        model.eval()
        print(f"✓ Model set to evaluation mode")
        
    except Exception as e:
        print(f"\n❌ ERROR loading weights: {e}")
        print(f"\nPossible reasons:")
        print(f"  - Model architecture mismatch")
        print(f"  - Checkpoint was saved with different hyperparameters")
        return False
    
    print(f"\n{'='*80}")
    print(f"✅ Model Check PASSED - Model is valid!")
    print(f"{'='*80}\n")
    
    return True


def test_inference(model_path, dataset='davis'):
    """Test model inference with sample data"""
    print(f"\n{'='*80}")
    print(f"Testing Model Inference")
    print(f"{'='*80}\n")
    
    try:
        # Initialize
        hp = HyperParameter()
        hp.dataset = dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {device}")
        
        # Load model
        model = nn.DataParallel(LLMDTA(hp, device))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        
        # Create dummy input
        batch_size = 2
        mol_vec = torch.randn(batch_size, hp.mol2vec_dim).to(device)
        prot_vec = torch.randn(batch_size, hp.protvec_dim).to(device)
        mol_mat = torch.randn(batch_size, hp.drug_max_len, hp.mol2vec_dim).to(device)
        mol_mat_mask = torch.ones(batch_size, hp.drug_max_len).to(device)
        prot_mat = torch.randn(batch_size, hp.prot_max_len, hp.protvec_dim).to(device)
        prot_mat_mask = torch.ones(batch_size, hp.prot_max_len).to(device)
        
        print(f"\n✓ Created dummy input tensors:")
        print(f"  mol_vec: {mol_vec.shape}")
        print(f"  prot_vec: {prot_vec.shape}")
        print(f"  mol_mat: {mol_mat.shape}")
        print(f"  prot_mat: {prot_mat.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values: {output.cpu().numpy().flatten()}")
        
        print(f"\n{'='*80}")
        print(f"✅ Inference Test PASSED - Model can make predictions!")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check LLMDTA model validity')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model .pth file')
    parser.add_argument('--dataset', type=str, default='davis',
                        help='Dataset name (davis/kiba/metz)')
    parser.add_argument('--test-inference', action='store_true',
                        help='Also test inference with dummy data')
    
    args = parser.parse_args()
    
    # Check model
    success = check_model(args.model)
    
    # Test inference if requested
    if success and args.test_inference:
        test_inference(args.model, args.dataset)
