"""
Parse test results from .out files and aggregate them.
Usage:
  python code/parse_out_files.py --all
  python code/parse_out_files.py --dataset davis --running_set novel-drug
  python code/parse_out_files.py --list
"""
import os
import re
import argparse
import pandas as pd
import glob

def find_all_combinations(log_dir='./log'):
    """
    Scan log directory to find all unique dataset-running_set combinations from .out files.
    
    Returns:
        List of (dataset, running_set) tuples
    """
    pattern = f"{log_dir}/*_fold_*.out"
    files = glob.glob(pattern)
    
    combinations = set()
    
    # Parse filenames: {dataset}_{running_set}_fold_{X}_{jobid}.out
    for file in files:
        basename = os.path.basename(file)
        # Pattern: dataset_running_set_fold_X_jobid.out
        # running_set can be: warm, novel_drug, novel_prot, novel_pair
        match = re.match(r'(davis|kiba|metz)_(warm|novel_drug|novel_prot|novel_pair)_fold_\d+_\d+\.out', basename)
        if match:
            dataset = match.group(1)
            running_set = match.group(2).replace('_', '-')  # Convert novel_drug -> novel-drug
            combinations.add((dataset, running_set))
    
    return sorted(list(combinations))

def parse_test_results_from_out(out_file):
    """
    Parse test results from a .out file.
    
    Looks for line like:
    Test at fold-X, mse: 0.631866, rmse: 0.7949, ci: 0.663934, r2: 0.063495, pearson: 0.444615, spearman: 0.295378
    
    Returns:
        dict with fold and metrics, or None if not found
    """
    try:
        with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Pattern to match test results
        pattern = r'Test at fold-(\d+),\s+mse:\s+([\d.]+),\s+rmse:\s+([\d.]+),\s+ci:\s+([\d.]+),\s+r2:\s+([\d.]+),\s+pearson:\s+([\d.]+),\s+spearman:\s+([\d.]+)'
        
        match = re.search(pattern, content)
        if match:
            return {
                'fold': int(match.group(1)),
                'mse': float(match.group(2)),
                'rmse': float(match.group(3)),
                'ci': float(match.group(4)),
                'r2': float(match.group(5)),
                'pearson': float(match.group(6)),
                'spearman': float(match.group(7))
            }
        return None
    except Exception as e:
        print(f"  [ERROR] Failed to parse {os.path.basename(out_file)}: {e}")
        return None

def aggregate_from_out_files(dataset, running_set, log_dir='./log', verbose=True):
    """
    Parse .out files and aggregate results for a dataset-running_set combination.
    
    Args:
        dataset: Dataset name (davis, kiba, metz)
        running_set: Task setting (warm, novel-drug, novel-prot, novel-pair)
        log_dir: Directory containing .out files
        verbose: Print detailed output
    
    Returns:
        summary DataFrame or None if no files found
    """
    # Convert novel-drug -> novel_drug for filename matching
    file_running_set = running_set.replace('-', '_')
    
    # Find all .out files for this combination
    pattern = f"{log_dir}/{dataset}_{file_running_set}_fold_*.out"
    out_files = glob.glob(pattern)
    
    if not out_files:
        if verbose:
            print(f"[WARNING] No .out files found for {dataset}-{running_set}")
            print(f"  Pattern: {pattern}")
        return None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing: {dataset.upper()} - {running_set}")
        print(f"{'='*70}")
        print(f"Found {len(out_files)} .out files:")
        for f in sorted(out_files):
            print(f"  - {os.path.basename(f)}")
    
    # Parse results from each file
    results = []
    for out_file in sorted(out_files):
        result = parse_test_results_from_out(out_file)
        if result:
            results.append(result)
            if verbose:
                print(f"  [OK] Parsed fold {result['fold']}: MSE={result['mse']:.6f}, CI={result['ci']:.6f}")
        else:
            if verbose:
                print(f"  [WARNING] Could not parse results from {os.path.basename(out_file)}")
    
    if not results:
        print(f"  [WARNING] No valid results parsed for {dataset}-{running_set}")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('fold').reset_index(drop=True)
    
    # Calculate statistics
    metrics = ['mse', 'rmse', 'ci', 'r2', 'pearson', 'spearman']
    mean_values = df[metrics].mean()
    std_values = df[metrics].std()
    var_values = df[metrics].var()
    
    summary = pd.DataFrame({
        'metric': metrics,
        'mean': mean_values.values,
        'std': std_values.values,
        'var': var_values.values
    })
    
    # Save aggregated results
    output_file = f"{log_dir}/Test-{dataset}-{running_set}-AGGREGATED.csv"
    df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n[SUCCESS] Aggregated results saved to: {output_file}")
        print(f"\nResults by fold:")
        print("-" * 70)
        print(df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print("STATISTICS (Mean +/- Std)")
        print(f"{'='*70}")
        for _, row in summary.iterrows():
            print(f"  {row['metric']:10s}: {row['mean']:.6f} +/- {row['std']:.6f} (var={row['var']:.8f})")
        print(f"{'='*70}")
    
    # Save summary
    summary_file = f"{log_dir}/Test-{dataset}-{running_set}-SUMMARY.csv"
    summary.to_csv(summary_file, index=False)
    
    if verbose:
        print(f"[SUCCESS] Summary statistics saved to: {summary_file}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parse and aggregate test results from .out files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate specific dataset and setting
  python code/parse_out_files.py --dataset davis --running_set novel-drug
  
  # Aggregate all combinations found in log/
  python code/parse_out_files.py --all
  
  # List all available combinations
  python code/parse_out_files.py --list
        """
    )
    
    parser.add_argument('--dataset', type=str,
                        help='Dataset name (davis, kiba, metz)')
    parser.add_argument('--running_set', type=str,
                        help='Task setting (warm, novel-drug, novel-prot, novel-pair)')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='Directory containing .out files (default: ./log)')
    parser.add_argument('--all', action='store_true',
                        help='Aggregate all dataset-running_set combinations found')
    parser.add_argument('--list', action='store_true',
                        help='List all available combinations')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode - minimal output')
    
    args = parser.parse_args()
    
    # Check log directory exists
    if not os.path.exists(args.log_dir):
        print(f"[ERROR] Log directory '{args.log_dir}' does not exist!")
        exit(1)
    
    # List mode
    if args.list:
        print(f"\nScanning log directory: {args.log_dir}")
        print("=" * 70)
        combinations = find_all_combinations(args.log_dir)
        
        if not combinations:
            print("No .out files found in log directory.")
            print(f"Looking for pattern: {args.log_dir}/*_fold_*.out")
        else:
            print(f"Found {len(combinations)} dataset-running_set combinations:\n")
            
            # Group by dataset
            datasets = {}
            for dataset, running_set in combinations:
                if dataset not in datasets:
                    datasets[dataset] = []
                datasets[dataset].append(running_set)
            
            for dataset in sorted(datasets.keys()):
                print(f"  {dataset.upper()}:")
                for setting in sorted(datasets[dataset]):
                    print(f"    - {setting}")
                print()
        
        exit(0)
    
    # Aggregate all mode
    if args.all:
        print(f"\nScanning log directory: {args.log_dir}")
        print("=" * 70)
        combinations = find_all_combinations(args.log_dir)
        
        if not combinations:
            print("No .out files found to parse.")
            exit(1)
        
        print(f"Found {len(combinations)} combinations to process\n")
        
        success_count = 0
        failed_count = 0
        
        for dataset, running_set in combinations:
            result = aggregate_from_out_files(dataset, running_set, args.log_dir, verbose=not args.quiet)
            if result is not None:
                success_count += 1
            else:
                failed_count += 1
        
        print(f"\n{'='*70}")
        print("AGGREGATION COMPLETE")
        print(f"{'='*70}")
        print(f"  [SUCCESS] Successfully aggregated: {success_count}")
        if failed_count > 0:
            print(f"  [FAILED] Failed: {failed_count}")
        print(f"{'='*70}\n")
        
    # Single combination mode
    elif args.dataset and args.running_set:
        result = aggregate_from_out_files(args.dataset, args.running_set, args.log_dir, verbose=not args.quiet)
        if result is None:
            exit(1)
    
    # Error: missing required arguments
    else:
        parser.print_help()
        print("\n[ERROR] Please specify either:")
        print("  1. --dataset and --running_set for specific aggregation")
        print("  2. --all to aggregate all combinations")
        print("  3. --list to list available combinations")
        exit(1)
