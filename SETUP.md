# LLMDTA Setup Guide

This guide will help you set up the LLMDTA repository for drug-target affinity prediction.

## Prerequisites

- Python 3.7-3.9 (recommended for compatibility with torch 1.8.2)
- pip
- Git

## Installation Steps

### 1. Install Python Dependencies

Install the basic requirements:
```bash
pip install -r requirements.txt
```

### 2. Install RDKit

RDKit requires special installation. Choose the appropriate method for your system:

**For Windows:**
```bash
conda install -c conda-forge rdkit
```
Or download from: https://www.rdkit.org/docs/Install.html

**For Linux/Mac:**
```bash
conda install -c conda-forge rdkit
```

### 3. Install ESM2

ESM2 (Evolutionary Scale Modeling) needs to be installed from the Facebook Research repository:

```bash
pip install fair-esm
```

Or clone and install from source:
```bash
git clone https://github.com/facebookresearch/esm.git
cd esm
pip install -e .
```

### 4. Verify Installation

You can verify the installation by running:
```python
import torch
import rdkit
import esm
print("All packages installed successfully!")
```

## Dataset Setup

### For Training

1. Download datasets from: https://www.kaggle.com/datasets/christang0002/llmdta/data
2. Extract the datasets to the `data/` directory
3. Update `code/hyperparameter.py` with your dataset paths

### For Prediction

The repository includes example prediction data in:
- `data/EGFR-Case/` - Example EGFR case study
- `data/simple-Case/` - Simple prediction example

## Configuration

Before running training or prediction, configure the hyperparameters:

- **Training**: Edit `code/hyperparameter.py`
- **Prediction**: Edit `code/hyperparameter4pred.py`

Key settings to configure:
- `data_root`: Path to your dataset
- `dataset`: Dataset name (davis, kiba, or metz)
- `cuda`: GPU device number (use "cpu" for CPU-only)

## Usage

### Training
```bash
python code/train.py
```

### Prediction
```bash
python code/pred.py
```

## Notes

- The code uses PyTorch 1.8.2, which may require specific CUDA versions
- RDKit installation can be tricky on Windows - consider using conda
- ESM2 models are large and will be downloaded automatically on first use
- Pre-trained models are available in the `savemodel/` directory

## Troubleshooting

1. **RDKit import errors**: Use conda to install RDKit instead of pip
2. **CUDA errors**: Check your PyTorch CUDA version compatibility
3. **ESM2 errors**: Ensure you have sufficient disk space (models are large)
4. **Memory errors**: Reduce batch size in hyperparameter.py


