# LLMDTA Quick Start Guide

## Repository Setup Complete ✅

The LLMDTA repository has been cloned and basic setup files have been created.

## Repository Structure

```
EXPLAIN/
├── code/                    # Main code directory
│   ├── train.py            # Training script
│   ├── pred.py             # Prediction script
│   ├── hyperparameter.py   # Training configuration
│   ├── hyperparameter4pred.py  # Prediction configuration
│   └── ...
├── code_prepareEmb/         # Embedding preparation notebooks
├── data/                    # Dataset directory
│   ├── dta-5fold-dataset/   # Training datasets (compressed)
│   ├── EGFR-Case/           # Example prediction case
│   └── simple-Case/         # Simple prediction example
├── savemodel/               # Pre-trained models
├── requirements.txt         # Python dependencies
└── README.md               # Original README
```

## Installation Checklist

- [ ] **Python Environment**: Set up Python 3.9 (recommended) or use Python 3.10+ with updated requirements
- [ ] **Basic Dependencies**: Install from `requirements.txt` or `requirements_updated.txt`
- [ ] **RDKit**: Install via conda (`conda install -c conda-forge rdkit`)
- [ ] **ESM2**: Install from GitHub (`pip install fair-esm` or clone from Facebook Research)

## Quick Test

### 1. Check Python Version
```bash
python --version
# Should be 3.7-3.9 for best compatibility
```

### 2. Install Dependencies
```bash
# For Python 3.7-3.9
pip install -r requirements.txt

# For Python 3.10+
pip install -r requirements_updated.txt
```

### 3. Install Special Packages
```bash
# RDKit (use conda if possible)
conda install -c conda-forge rdkit

# ESM2
pip install fair-esm
```

### 4. Verify Installation
```python
python -c "import torch; import pandas; import numpy; print('Basic packages OK')"
```

## Running the Code

### Training
1. Configure `code/hyperparameter.py`
2. Ensure datasets are in the correct location
3. Run: `python code/train.py`

### Prediction
1. Configure `code/hyperparameter4pred.py`
2. Prepare your drug/target data
3. Run: `python code/pred.py`

## Configuration Files

### Training Configuration (`code/hyperparameter.py`)
Key settings:
- `data_root`: Dataset root path
- `dataset`: 'davis', 'kiba', or 'metz'
- `running_set`: 'warm', 'novel-drug', 'novel-prot', or 'novel-pair'
- `cuda`: GPU device (e.g., "0" or "cpu")

### Prediction Configuration (`code/hyperparameter4pred.py`)
Key settings:
- `pred_dataset`: Name for saving results
- `pred_pair_pth`: Path to drug-target pairs file
- `model_fromTrain`: Path to trained model
- `cuda`: GPU device

## Example Datasets

The repository includes:
- **Pre-trained models** in `savemodel/` for davis, kiba, and metz datasets
- **Example prediction cases** in `data/EGFR-Case/` and `data/simple-Case/`
- **Compressed datasets** in `data/dta-5fold-dataset/` (need extraction)

## Troubleshooting

See `INSTALLATION_NOTES.md` for detailed troubleshooting information.

## Next Steps

1. Set up Python 3.9 environment (if using Python 3.12)
2. Install all dependencies
3. Extract training datasets if needed
4. Configure hyperparameters
5. Run training or prediction

## Resources

- Original README: `README.md`
- Setup Guide: `SETUP.md`
- Installation Notes: `INSTALLATION_NOTES.md`
- GitHub Repository: https://github.com/Chris-Tang6/LLMDTA


