"""
System Check Script for LLMDTA Training
This script checks if your device can run train.py
"""

import sys
import os
import platform

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 7:
        print("[OK] Python version is compatible")
        return True
    else:
        print("[X] Python version should be 3.7 or higher")
        return False

def check_packages():
    """Check if required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'scipy': 'SciPy',
        'tqdm': 'tqdm',
        'gensim': 'Gensim',
        'rdkit': 'RDKit',
        'mol2vec': 'Mol2vec'
    }
    
    missing = []
    installed = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            installed.append(name)
            print(f"[OK] {name} is installed")
        except ImportError:
            missing.append(name)
            print(f"[X] {name} is NOT installed")
    
    return len(missing) == 0, missing

def check_pytorch():
    """Check PyTorch installation and CUDA availability"""
    try:
        import torch
        print(f"\nPyTorch Version: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"[OK] CUDA is available")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            
            # Check if GPU has enough memory (estimate: need at least 2GB)
            if torch.cuda.get_device_properties(0).total_memory / (1024**3) >= 2:
                print("[OK] GPU memory appears sufficient (>= 2GB)")
                return True, 'cuda'
            else:
                print("[WARNING] GPU memory may be limited (< 2GB), training might be slow")
                return True, 'cuda'
        else:
            print("[WARNING] CUDA is NOT available - will use CPU (training will be slower)")
            return True, 'cpu'
    except ImportError:
        print("[X] PyTorch is NOT installed")
        return False, None

def check_data_files():
    """Check if required data files exist"""
    from hyperparameter import HyperParameter
    
    hp = HyperParameter()
    missing_files = []
    
    # Check data files
    data_files = {
        'drugs_dir': hp.drugs_dir,
        'prots_dir': hp.prots_dir,
        'mol2vec_dir': hp.mol2vec_dir,
        'protvec_dir': hp.protvec_dir
    }
    
    print(f"\nDataset: {hp.dataset}")
    print(f"Running set: {hp.running_set}")
    print(f"Data root: {hp.data_root}")
    
    for name, path in data_files.items():
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024**2)  # MB
            print(f"[OK] {name}: {path} ({file_size:.2f} MB)")
        else:
            print(f"[X] {name}: {path} - NOT FOUND")
            missing_files.append(path)
    
    # Check fold files
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    if os.path.exists(dataset_root):
        print(f"[OK] Dataset directory exists: {dataset_root}")
        for fold in range(hp.kfold):
            train_file = os.path.join(dataset_root, f'fold_{fold}_train.csv')
            valid_file = os.path.join(dataset_root, f'fold_{fold}_valid.csv')
            test_file = os.path.join(dataset_root, f'fold_{fold}_test.csv')
            
            if os.path.exists(train_file) and os.path.exists(valid_file) and os.path.exists(test_file):
                print(f"[OK] Fold {fold} files exist")
            else:
                print(f"[X] Fold {fold} files missing")
                missing_files.extend([train_file, valid_file, test_file])
    else:
        print(f"[X] Dataset directory NOT FOUND: {dataset_root}")
        missing_files.append(dataset_root)
    
    return len(missing_files) == 0, missing_files

def estimate_memory_requirements():
    """Estimate memory requirements"""
    from hyperparameter import HyperParameter
    
    hp = HyperParameter()
    batch_size = hp.Batch_size
    
    # Estimate per-batch memory (in MB)
    # Drug: batch_size * max_len * dim * 4 bytes (float32)
    drug_mat_mem = batch_size * hp.substructure_max_len * hp.mol2vec_dim * 4 / (1024**2)
    # Protein: batch_size * max_len * dim * 4 bytes
    prot_mat_mem = batch_size * hp.prot_max_len * hp.protvec_dim * 4 / (1024**2)
    # Vectors
    drug_vec_mem = batch_size * hp.mol2vec_dim * 4 / (1024**2)
    prot_vec_mem = batch_size * hp.protvec_dim * 4 / (1024**2)
    
    batch_memory = drug_mat_mem + prot_mat_mem + drug_vec_mem + prot_vec_mem
    
    # Model parameters estimate (rough)
    # Encoder layers, MLP, attention layers
    model_memory = 50  # Rough estimate in MB
    
    # Gradient and optimizer states (typically 2-3x model size)
    optimizer_memory = model_memory * 3
    
    total_per_batch = batch_memory + model_memory + optimizer_memory
    
    print(f"\nMemory Estimates:")
    print(f"  Batch size: {batch_size}")
    print(f"  Per-batch data: ~{batch_memory:.2f} MB")
    print(f"  Model parameters: ~{model_memory} MB")
    print(f"  Optimizer states: ~{optimizer_memory} MB")
    print(f"  Total per batch: ~{total_per_batch:.2f} MB")
    print(f"  Recommended GPU memory: >= 4 GB")
    print(f"  Recommended RAM (CPU): >= 8 GB")

def check_system_resources():
    """Check system resources"""
    import psutil
    
    # RAM
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024**3)
    ram_available_gb = ram.available / (1024**3)
    
    print(f"\nSystem Resources:")
    print(f"  Total RAM: {ram_total_gb:.2f} GB")
    print(f"  Available RAM: {ram_available_gb:.2f} GB")
    
    if ram_total_gb >= 8:
        print("[OK] RAM appears sufficient (>= 8 GB)")
    else:
        print("[WARNING] RAM may be limited (< 8 GB), consider reducing batch size")
    
    # CPU
    cpu_count = psutil.cpu_count()
    print(f"  CPU Cores: {cpu_count}")
    print(f"  Note: Training uses {4} threads (set in train.py)")

def main():
    """Main check function"""
    print("=" * 60)
    print("LLMDTA Training System Check")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        all_checks_passed = False
    
    # Check packages
    print("\n2. Checking required packages...")
    packages_ok, missing = check_packages()
    if not packages_ok:
        all_checks_passed = False
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install them using: pip install -r requirements.txt")
    
    # Check PyTorch and CUDA
    print("\n3. Checking PyTorch and CUDA...")
    pytorch_ok, device_type = check_pytorch()
    if not pytorch_ok:
        all_checks_passed = False
    
    # Check data files
    print("\n4. Checking data files...")
    try:
        data_ok, missing_files = check_data_files()
        if not data_ok:
            all_checks_passed = False
            print(f"\nMissing files: {len(missing_files)} files")
    except Exception as e:
        print(f"[WARNING] Could not check data files: {e}")
        all_checks_passed = False
    
    # Estimate memory
    print("\n5. Memory requirements...")
    try:
        estimate_memory_requirements()
    except Exception as e:
        print(f"[WARNING] Could not estimate memory: {e}")
    
    # Check system resources
    print("\n6. System resources...")
    try:
        check_system_resources()
    except ImportError:
        print("[WARNING] psutil not installed - skipping system resource check")
        print("  Install with: pip install psutil")
    except Exception as e:
        print(f"[WARNING] Could not check system resources: {e}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("[OK] SYSTEM CHECK PASSED - Your device should be able to run train.py")
        if device_type == 'cpu':
            print("[WARNING] Note: Training on CPU will be slower than GPU")
            print("  Consider using a smaller batch size (e.g., 8) for CPU training")
        else:
            print("[OK] GPU detected - Training should run efficiently")
    else:
        print("[X] SYSTEM CHECK FAILED - Some requirements are missing")
        print("  Please install missing packages or fix data file paths")
    print("=" * 60)

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("Note: psutil not installed. Install with: pip install psutil")
    
    main()

