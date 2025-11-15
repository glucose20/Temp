"""
Script to aggregate results from individual fold training runs.
Usage: python code/aggregate_results.py --dataset davis --running_set novel-pair
"""
import os
import argparse
import pandas as pd
import glob

def aggregate_fold_results(dataset, running_set, log_dir='./log'):
    """
    Aggregate test results from all folds into a single summary file.
    
    Args:
        dataset: Dataset name (davis, kiba, metz)
        running_set: Task setting (warm, novel-drug, novel-prot, novel-pair)
        log_dir: Directory containing fold result files
    """
    # Find all fold result files matching the pattern
    pattern = f"{log_dir}/Test-{dataset}-{running_set}-fold*-*.csv"
    fold_files = glob.glob(pattern)
    
    if not fold_files:
        print(f"No fold result files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(fold_files)} fold result files:")
    for f in sorted(fold_files):
        print(f"  - {f}")
    
    # Read and concatenate all fold results
    fold_dfs = []
    for file in sorted(fold_files):
        df = pd.read_csv(file)
        fold_dfs.append(df)
    
    # Combine all folds
    all_folds = pd.concat(fold_dfs, ignore_index=True)
    all_folds = all_folds.sort_values('fold')
    
    # Calculate statistics
    metrics = ['mse', 'rmse', 'ci', 'r2', 'pearson', 'spearman']
    mean_values = all_folds[metrics].mean()
    std_values = all_folds[metrics].std()
    var_values = all_folds[metrics].var()
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'metric': metrics,
        'mean': mean_values.values,
        'std': std_values.values,
        'var': var_values.values
    })
    
    # Output file
    output_file = f"{log_dir}/Test-{dataset}-{running_set}-AGGREGATED.csv"
    all_folds.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print(f"Aggregated results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("=" * 60)
    print(f"SUMMARY: {dataset}-{running_set}")
    print("=" * 60)
    print(all_folds.to_string(index=False))
    print("\n" + "=" * 60)
    print("STATISTICS (Mean ± Std)")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"{row['metric']:10s}: {row['mean']:.6f} ± {row['std']:.6f} (var={row['var']:.6f})")
    print("=" * 60)
    
    # Save summary statistics
    summary_file = f"{log_dir}/Test-{dataset}-{running_set}-SUMMARY.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate fold results')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (davis, kiba, metz)')
    parser.add_argument('--running_set', type=str, required=True,
                        help='Task setting (warm, novel-drug, novel-prot, novel-pair)')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='Directory containing log files (default: ./log)')
    
    args = parser.parse_args()
    
    aggregate_fold_results(args.dataset, args.running_set, args.log_dir)
