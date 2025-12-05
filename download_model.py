"""
Script to download the required model_300dim.pkl file.
This file is needed for mol2vec embeddings.

Since the file is large (~73MB), it's not included in the git repository.
You can download it from:
https://www.kaggle.com/datasets/christang0002/llmdta/data

Or use this script to check if the file exists and provide download instructions.
"""

import os
import sys

def check_model_file():
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'data', 'model_300dim.pkl')
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
        print(f"✓ model_300dim.pkl found at: {model_path}")
        print(f"  File size: {file_size:.2f} MB")
        return True
    else:
        print(f"✗ model_300dim.pkl NOT found at: {model_path}")
        print("\nPlease download the file from:")
        print("https://www.kaggle.com/datasets/christang0002/llmdta/data")
        print(f"\nAnd place it in: {os.path.dirname(model_path)}/")
        print("\nAlternatively, you can use Google Colab to download it:")
        print("!pip install kaggle")
        print("!kaggle datasets download -d christang0002/llmdta")
        print("!unzip llmdta.zip -d ./data/")
        return False

if __name__ == "__main__":
    if not check_model_file():
        sys.exit(1)
    else:
        print("\nAll required model files are present!")
