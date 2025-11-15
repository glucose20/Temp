#!/bin/bash
################################################################################
# LLMDTA Server Setup Script
# Automatically sets up environment and prepares data for experiments
################################################################################

set -e

echo "============================================================"
echo "LLMDTA - Server Setup Script"
echo "============================================================"
echo ""

# 1. Install dependencies
echo "Step 1: Installing Python dependencies..."
pip install -q numpy pandas scipy scikit-learn tqdm gensim matplotlib mol2vec psutil

# Install PyTorch (adjust based on CUDA version)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install RDKit (via conda if available, otherwise pip)
if command -v conda &> /dev/null; then
    echo "Installing RDKit via conda..."
    conda install -y -c conda-forge rdkit
else
    echo "Installing RDKit via pip..."
    pip install rdkit
fi

# Install ESM
echo "Installing ESM (fair-esm)..."
pip install fair-esm

echo "✓ Dependencies installed"
echo ""

# 2. Download dataset from Kaggle
echo "Step 2: Downloading dataset from Kaggle..."

# Check if kagglehub is installed
if ! python -c "import kagglehub" 2>/dev/null; then
    echo "Installing kagglehub..."
    pip install kagglehub
fi

# Download pretrained features
python << 'PYTHON_SCRIPT'
import kagglehub
import shutil
import os

print("Downloading LLMDTA dataset from Kaggle...")
path = kagglehub.dataset_download("christang0002/llmdta")
print(f"Downloaded to: {path}")

pretrain_dir = f'{path}/pretrain-feature/pretrained-feature'

# Copy pretrained features for each dataset
for dataset in ['davis', 'kiba', 'metz']:
    src = os.path.join(pretrain_dir, dataset)
    dst = f'./data/{dataset}'
    
    if os.path.exists(dst):
        print(f"Removing existing {dst}...")
        shutil.rmtree(dst)
    
    print(f"Copying {dataset} pretrained features...")
    shutil.copytree(src, dst)
    print(f"✓ Copied {dataset}")

print("\n✓ Pretrained features downloaded and copied")
PYTHON_SCRIPT

echo ""

# 3. Extract 5-fold datasets
echo "Step 3: Extracting 5-fold datasets..."

if [ -f "./data/dta-5fold-dataset/davis.tar.gz" ]; then
    echo "Extracting davis.tar.gz..."
    tar -xzf ./data/dta-5fold-dataset/davis.tar.gz -C ./data/dta-5fold-dataset/
    echo "✓ davis extracted"
fi

if [ -f "./data/dta-5fold-dataset/kiba.tar.gz" ]; then
    echo "Extracting kiba.tar.gz..."
    tar -xzf ./data/dta-5fold-dataset/kiba.tar.gz -C ./data/dta-5fold-dataset/
    echo "✓ kiba extracted"
fi

if [ -f "./data/dta-5fold-dataset/metz.tar.gz" ]; then
    echo "Extracting metz.tar.gz..."
    tar -xzf ./data/dta-5fold-dataset/metz.tar.gz -C ./data/dta-5fold-dataset/
    echo "✓ metz extracted"
fi

echo ""

# 4. Create necessary directories
echo "Step 4: Creating directories..."
mkdir -p ./log
mkdir -p ./savemodel
mkdir -p ./results
echo "✓ Directories created"
echo ""

# 5. Make scripts executable
echo "Step 5: Setting script permissions..."
chmod +x scripts/*.sh
echo "✓ Scripts are now executable"
echo ""

# 6. Verify setup
echo "Step 6: Verifying setup..."

# Check PyTorch and CUDA
python << 'PYTHON_SCRIPT'
import torch
import sys

print("\nPyTorch Information:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("  WARNING: CUDA not available! Training will be very slow on CPU.")
    print("  Please install CUDA-enabled PyTorch.")
PYTHON_SCRIPT

# Check data files
echo ""
echo "Data files check:"
for dataset in davis kiba metz; do
    drug_pkl="./data/${dataset}/${dataset}_drug_pretrain.pkl"
    prot_pkl="./data/${dataset}/${dataset}_esm_pretrain.pkl"
    drugs_csv="./data/dta-5fold-dataset/${dataset}/${dataset}_drugs.csv"
    prots_csv="./data/dta-5fold-dataset/${dataset}/${dataset}_prots.csv"
    
    echo -n "  $dataset: "
    if [ -f "$drug_pkl" ] && [ -f "$prot_pkl" ] && [ -f "$drugs_csv" ] && [ -f "$prots_csv" ]; then
        echo "✓ All files present"
    else
        echo "✗ Missing files"
        [ ! -f "$drug_pkl" ] && echo "    Missing: $drug_pkl"
        [ ! -f "$prot_pkl" ] && echo "    Missing: $prot_pkl"
        [ ! -f "$drugs_csv" ] && echo "    Missing: $drugs_csv"
        [ ! -f "$prots_csv" ] && echo "    Missing: $prots_csv"
    fi
done

echo ""
echo "============================================================"
echo "Setup completed successfully!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Test with a quick run:"
echo "     python code/train.py --fold 0 --cuda 0 --dataset davis --running_set warm --epochs 2"
echo ""
echo "  2. Run all experiments:"
echo "     bash scripts/run_all_experiments.sh          # Sequential"
echo "     bash scripts/run_all_experiments_parallel.sh # Parallel (multi-GPU)"
echo ""
echo "  3. Or run one dataset at a time:"
echo "     bash scripts/run_single_dataset.sh davis"
echo ""
echo "For more details, see EXPERIMENT_GUIDE.md"
echo "============================================================"
