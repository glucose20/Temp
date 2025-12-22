#!/usr/bin/env python
"""
A/B Testing Analysis Script
Compare results from different MoE configurations

Usage:
    python scripts/analyze_ab_results.py --results_dir ./ab_results/2024-12-22_10-30-00
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path


def extract_test_results(log_file):
    """Extract test metrics from log file."""
    results = {}
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract test results: "Test at fold-0, mse: 0.234, rmse: 0.484, ci: 0.876, r2: 0.65, pearson: 0.81, spearman: 0.79"
    test_pattern = r"Test at fold-(\d+), mse: ([\d.]+), rmse: ([\d.]+), ci: ([\d.]+), r2: ([\d.]+), pearson: ([\d.]+), spearman: ([\d.]+)"
    match = re.search(test_pattern, content)
    
    if match:
        results['fold'] = int(match.group(1))
        results['mse'] = float(match.group(2))
        results['rmse'] = float(match.group(3))
        results['ci'] = float(match.group(4))
        results['r2'] = float(match.group(5))
        results['pearson'] = float(match.group(6))
        results['spearman'] = float(match.group(7))
    
    # Extract best validation MSE
    best_valid_pattern = r"Update best_mse, Valid at fold-\d+ epoch-(\d+): mse-([\d.]+)"
    best_matches = re.findall(best_valid_pattern, content)
    if best_matches:
        last_best = best_matches[-1]
        results['best_epoch'] = int(last_best[0])
        results['best_valid_mse'] = float(last_best[1])
    
    # Extract MoE stats from last epoch
    moe_pattern = r"MoE Stats:.*entropy=([\d.]+).*usage_std.*"
    moe_matches = re.findall(r"entropy=([\d.]+)", content)
    if moe_matches:
        results['final_entropy'] = float(moe_matches[-1])
    
    return results


def analyze_results(results_dir):
    """Analyze all log files in results directory."""
    results_dir = Path(results_dir)
    
    all_results = []
    
    for log_file in results_dir.glob("*.log"):
        config_name = log_file.stem
        results = extract_test_results(log_file)
        results['config'] = config_name
        all_results.append(results)
    
    if not all_results:
        print(f"No log files found in {results_dir}")
        return None
    
    df = pd.DataFrame(all_results)
    
    # Reorder columns
    cols = ['config', 'mse', 'rmse', 'ci', 'r2', 'pearson', 'spearman', 'best_epoch', 'best_valid_mse', 'final_entropy']
    df = df[[c for c in cols if c in df.columns]]
    
    return df


def print_comparison(df):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("A/B TESTING RESULTS COMPARISON")
    print("=" * 100)
    
    # Sort by MSE (lower is better)
    df_sorted = df.sort_values('mse')
    
    print("\n📊 Sorted by MSE (lower is better):")
    print("-" * 100)
    print(df_sorted.to_string(index=False))
    
    # Highlight best results
    print("\n" + "=" * 100)
    print("🏆 BEST RESULTS:")
    print("=" * 100)
    
    metrics = {
        'mse': ('lowest', df['mse'].idxmin()),
        'rmse': ('lowest', df['rmse'].idxmin()),
        'ci': ('highest', df['ci'].idxmax()),
        'r2': ('highest', df['r2'].idxmax()),
        'pearson': ('highest', df['pearson'].idxmax()),
        'spearman': ('highest', df['spearman'].idxmax()),
    }
    
    for metric, (direction, idx) in metrics.items():
        if metric in df.columns:
            best_config = df.loc[idx, 'config']
            best_value = df.loc[idx, metric]
            print(f"  {metric.upper():10s}: {best_value:.6f} ({direction}) -> {best_config}")
    
    # Calculate improvement over baseline
    print("\n" + "=" * 100)
    print("📈 IMPROVEMENT OVER BASELINE:")
    print("=" * 100)
    
    baseline_row = df[df['config'].str.contains('baseline', case=False)]
    if len(baseline_row) > 0:
        baseline_mse = baseline_row['mse'].values[0]
        baseline_ci = baseline_row['ci'].values[0]
        
        for idx, row in df.iterrows():
            if 'baseline' not in row['config'].lower():
                mse_improvement = (baseline_mse - row['mse']) / baseline_mse * 100
                ci_improvement = (row['ci'] - baseline_ci) / baseline_ci * 100
                print(f"  {row['config']:25s}: MSE {mse_improvement:+.2f}%, CI {ci_improvement:+.2f}%")
    else:
        print("  (No baseline found for comparison)")
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Analyze A/B testing results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing log files from ab_testing.ps1')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (optional)')
    args = parser.parse_args()
    
    df = analyze_results(args.results_dir)
    
    if df is not None:
        print_comparison(df)
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
