import re
import os
import pandas as pd

def parse_log_file(log_path):
    """Parse log file to extract test results."""
    results = {
        'fold': None,
        'MSE': None,
        'RMSE': None,
        'CI': None,
        'R2': None,
        'Pearson': None,
        'Spearman': None
    }
    
    # Extract fold number from filename
    match = re.search(r'fold_(\d+)', log_path)
    if match:
        results['fold'] = int(match.group(1))
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find Test Results section (get the last occurrence)
        test_section_idx = -1
        for i, line in enumerate(lines):
            if 'Test Results' in line:
                test_section_idx = i
        
        if test_section_idx >= 0:
            # Read next 10 lines for metrics
            for j in range(1, 11):
                if test_section_idx + j < len(lines):
                    metric_line = lines[test_section_idx + j]
                    
                    # Check RMSE before MSE to avoid matching MSE in RMSE
                    if 'RMSE:' in metric_line:
                        results['RMSE'] = float(re.search(r'RMSE:\s+([\d.]+)', metric_line).group(1))
                    elif 'MSE:' in metric_line:
                        results['MSE'] = float(re.search(r'MSE:\s+([\d.]+)', metric_line).group(1))
                    elif 'CI:' in metric_line:
                        results['CI'] = float(re.search(r'CI:\s+([\d.]+)', metric_line).group(1))
                    elif 'R2:' in metric_line:
                        results['R2'] = float(re.search(r'R2:\s+([\d.]+)', metric_line).group(1))
                    elif 'Pearson:' in metric_line:
                        results['Pearson'] = float(re.search(r'Pearson:\s+([\d.]+)', metric_line).group(1))
                    elif 'Spearman:' in metric_line:
                        results['Spearman'] = float(re.search(r'Spearman:\s+([\d.]+)', metric_line).group(1))
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
    
    return results


def aggregate_results(log_dir='./log', pattern='davis_warm_fold_*.txt'):
    """Aggregate results from all log files."""
    import glob
    
    log_files = sorted(glob.glob(os.path.join(log_dir, pattern)))
    
    if not log_files:
        print(f"No log files found matching pattern: {pattern}")
        return None
    
    print(f"Found {len(log_files)} log files:")
    for f in log_files:
        print(f"  - {os.path.basename(f)}")
    
    # Parse all log files
    all_results = []
    for log_file in log_files:
        results = parse_log_file(log_file)
        if results['MSE'] is not None:  # Valid result
            all_results.append(results)
        else:
            print(f"Warning: No test results found in {os.path.basename(log_file)}")
    
    if not all_results:
        print("No valid results found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    df = df.sort_values('fold')
    
    return df


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL FOLD RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("AVERAGE RESULTS (Mean ± Std)")
    print("=" * 80)
    
    metrics = ['MSE', 'RMSE', 'CI', 'R2', 'Pearson', 'Spearman']
    for metric in metrics:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        print(f"{metric:10s}: {mean_val:.6f} ± {std_val:.6f}")
    
    print("=" * 80)


def save_results(df, output_file='./log/davis_warm_summary.csv'):
    """Save results to CSV."""
    # Individual results
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Summary statistics
    summary_file = output_file.replace('.csv', '_stats.csv')
    stats_df = df.describe()
    stats_df.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate training results from log files')
    parser.add_argument('--log_dir', type=str, default='./log', help='Directory containing log files')
    parser.add_argument('--pattern', type=str, default='davis_warm_fold_*.txt', 
                       help='Pattern to match log files')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output CSV file (default: auto-generated)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for results (default: same as log_dir)')
    
    args = parser.parse_args()
    
    # Aggregate results
    df = aggregate_results(args.log_dir, args.pattern)
    
    if df is not None:
        # Print summary
        print_summary(df)
        
        # Save results
        if args.output is None:
            # Auto-generate output filename from pattern
            base_name = args.pattern.replace('fold_*.txt', 'summary.csv')
            output_dir = args.output_dir if args.output_dir else args.log_dir
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, base_name)
        else:
            output_file = args.output
        
        save_results(df, output_file)
