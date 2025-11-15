"""
Generate final summary report for all LLMDTA experiments
Aggregates results from all 60 runs (3 datasets × 4 settings × 5 folds)
"""
import os
import pandas as pd
import glob
from datetime import datetime

DATASETS = ['davis', 'kiba', 'metz']
SETTINGS = ['warm', 'novel-drug', 'novel-prot', 'novel-pair']
METRICS = ['mse', 'rmse', 'ci', 'r2', 'pearson', 'spearman']

def generate_summary_report(log_dir='./log'):
    """Generate comprehensive summary report for all experiments"""
    
    print("=" * 80)
    print("LLMDTA - Final Experiment Summary Report")
    print("=" * 80)
    print()
    
    # Collect all aggregated results
    all_results = []
    
    for dataset in DATASETS:
        for setting in SETTINGS:
            # Try to find the aggregated file
            pattern = f"{log_dir}/Test-{dataset}-{setting}-AGGREGATED.csv"
            files = glob.glob(pattern)
            
            if not files:
                print(f"⚠ Warning: Missing results for {dataset} - {setting}")
                continue
            
            # Read the aggregated results
            df = pd.read_csv(files[0])
            
            # Calculate statistics
            stats = {
                'dataset': dataset,
                'setting': setting,
                'num_folds': len(df)
            }
            
            for metric in METRICS:
                if metric in df.columns:
                    stats[f'{metric}_mean'] = df[metric].mean()
                    stats[f'{metric}_std'] = df[metric].std()
            
            all_results.append(stats)
            
            # Print individual result
            print(f"\n{dataset.upper()} - {setting}")
            print("-" * 60)
            for metric in METRICS:
                if f'{metric}_mean' in stats:
                    mean = stats[f'{metric}_mean']
                    std = stats[f'{metric}_std']
                    print(f"  {metric:10s}: {mean:.6f} ± {std:.6f}")
    
    # Create summary DataFrame
    if not all_results:
        print("\n⚠ No results found! Make sure experiments have been run.")
        return
    
    summary_df = pd.DataFrame(all_results)
    
    # Save summary to CSV
    output_file = f"{log_dir}/FINAL_SUMMARY_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print(f"Summary report saved to: {output_file}")
    print("=" * 80)
    
    # Print comparison table
    print("\n\nComparison Table (Mean MSE)")
    print("=" * 80)
    pivot_mse = summary_df.pivot(index='dataset', columns='setting', values='mse_mean')
    print(pivot_mse.to_string())
    
    print("\n\nComparison Table (Mean CI)")
    print("=" * 80)
    pivot_ci = summary_df.pivot(index='dataset', columns='setting', values='ci_mean')
    print(pivot_ci.to_string())
    
    # Best results
    print("\n\nBest Results by Metric")
    print("=" * 80)
    
    # Best MSE (lowest)
    best_mse = summary_df.loc[summary_df['mse_mean'].idxmin()]
    print(f"Best MSE: {best_mse['dataset']} - {best_mse['setting']} = {best_mse['mse_mean']:.6f}")
    
    # Best CI (highest)
    best_ci = summary_df.loc[summary_df['ci_mean'].idxmax()]
    print(f"Best CI:  {best_ci['dataset']} - {best_ci['setting']} = {best_ci['ci_mean']:.6f}")
    
    # Best R2 (highest)
    best_r2 = summary_df.loc[summary_df['r2_mean'].idxmax()]
    print(f"Best R²:  {best_r2['dataset']} - {best_r2['setting']} = {best_r2['r2_mean']:.6f}")
    
    print("\n" + "=" * 80)
    print("Report generation completed!")
    print("=" * 80)
    
    return summary_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate final summary report')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='Directory containing log files')
    
    args = parser.parse_args()
    
    summary_df = generate_summary_report(args.log_dir)
