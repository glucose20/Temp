# Installation Notes

## ⚠️ Important: Python Version Compatibility

The original requirements are designed for **Python 3.7-3.9**. If you're using Python 3.10 or newer (including Python 3.12), you may encounter compatibility issues.

### Recommended Approach

**Option 1: Use Python 3.9 (Recommended)**
1. Install Python 3.9
2. Create a virtual environment:
   ```bash
   python3.9 -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

**Option 2: Use Updated Versions (Python 3.10+)**
1. Try installing with updated requirements:
   ```bash
   pip install -r requirements_updated.txt
   ```
2. Note: This may require code modifications for compatibility

### Special Package Installations

#### RDKit
RDKit is required but cannot be installed via pip on Windows easily. Use conda:

```bash
conda install -c conda-forge rdkit
```

Or download pre-built wheels from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#rdkit

#### ESM2
Install from Facebook Research:
```bash
pip install fair-esm
```

Or clone and install:
```bash
git clone https://github.com/facebookresearch/esm.git
cd esm
pip install -e .
```

### Current Status

- ✅ Repository cloned successfully
- ⚠️ Python 3.12 detected - compatibility issues expected
- ⚠️ Some packages need manual installation (RDKit, ESM2)

### Next Steps

1. **Set up Python 3.9 environment** (recommended)
2. **Install RDKit** using conda
3. **Install ESM2** from GitHub
4. **Install remaining requirements**
5. **Configure hyperparameters** in `code/hyperparameter.py` or `code/hyperparameter4pred.py`

### Testing Installation

After installation, test with:
```python
import torch
print(f"PyTorch version: {torch.__version__}")

try:
    import rdkit
    print("RDKit installed successfully")
except ImportError:
    print("RDKit not installed - use conda to install")

try:
    import esm
    print("ESM2 installed successfully")
except ImportError:
    print("ESM2 not installed - install from GitHub")
```


