# Setup Guide for Google Colab

This guide helps you set up and run the prediction code on Google Colab.

## Quick Start

### 1. Clone the Repository

```bash
!git clone https://github.com/glucose20/Temp.git
%cd Temp
```

### 2. Download Required Model File

The `model_300dim.pkl` file (~73MB) is required for mol2vec embeddings but is not included in the repository due to size.

#### Option A: Download from Kaggle (Recommended)

```bash
# Install Kaggle CLI
!pip install -q kaggle

# Upload your kaggle.json to Colab (download from https://www.kaggle.com/settings)
# Then run:
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and extract the dataset
!kaggle datasets download -d christang0002/llmdta
!unzip -q llmdta.zip

# Copy the model file to the correct location
!cp model_300dim.pkl /content/Temp/data/
```

#### Option B: Download from Google Drive

If you have the file in Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from your Google Drive
!cp "/content/drive/MyDrive/path/to/model_300dim.pkl" /content/Temp/data/
```

### 3. Install Dependencies

```bash
!pip install -r requirements.txt
```

### 4. Prepare Your Data

Upload your drug and protein data files:

- `data/simple-Case/drugs.csv` - with columns: drug_id, drug_smile
- `data/simple-Case/proteins.csv` - with columns: prot_id, prot_seq

### 5. Run Prediction

```bash
%cd code
!python pred.py
```

## Troubleshooting

### FileNotFoundError: model_300dim.pkl

If you see this error, make sure you've completed Step 2 above.

### CUDA Out of Memory

If you encounter memory issues:
- Use a smaller batch size
- Use Colab Pro with more RAM/GPU
- Process data in smaller chunks

### Import Errors

Make sure all dependencies are installed:
```bash
!pip install torch esm-py pandas numpy gensim rdkit mol2vec
```

## File Structure

```
Temp/
├── code/
│   ├── pred.py              # Main prediction script
│   ├── hyperparameter4pred.py  # Configuration
│   └── utils.py             # Utility functions
├── data/
│   ├── model_300dim.pkl     # Mol2vec model (need to download)
│   └── simple-Case/
│       ├── drugs.csv        # Your drug data
│       └── proteins.csv     # Your protein data
└── savemodel/
    └── *.pth                # Trained model weights
```

## Notes

- The model file `model_300dim.pkl` must be downloaded separately
- Make sure you have the correct trained model weights in `savemodel/`
- Default configuration uses GPU if available, falls back to CPU
