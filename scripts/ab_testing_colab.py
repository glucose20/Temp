"""
A/B Testing Script for Google Colab
Run this script on Colab to compare Baseline vs MoE configurations

Usage on Colab:
    !python scripts/ab_testing_colab.py --dataset davis --running_set novel-pair --fold 0 --epochs 100
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import pandas as pd
import re

def run_experiment(config_name, args_list, results_dir):
    """Run a single experiment and save results."""
    log_file = os.path.join(results_dir, f"{config_name}.log")
    
    print(f"\n{'='*70}")
    print(f"🚀 Running: {config_name}")
    print(f"{'='*70}")
    
    cmd = ["python", "code/train_no_3_epochs.py"] + args_list
    print(f"Command: {' '.join(cmd)}")
    
    # Run and capture output
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')  # Print to console
            f.write(line)        # Write to log
        
        process.wait()
    
    print(f"\n✅ {config_name} completed! Log saved to: {log_file}")
    return log_file


def extract_test_results(log_file):
    """Extract test metrics from log file."""
    results = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return results
    
    # Extract test results
    test_pattern = r"Test at fold-(\d+), mse: ([\d.]+), rmse: ([\d.]+), ci: ([\d.]+), r2: ([\d.-]+), pearson: ([\d.-]+), spearman: ([\d.-]+)"
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
    
    # Extract final MoE entropy
    entropy_matches = re.findall(r"entropy=([\d.]+)", content)
    if entropy_matches:
        results['final_entropy'] = float(entropy_matches[-1])
    
    return results


def print_comparison(all_results):
    """Print formatted comparison table."""
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No results to compare!")
        return df
    
    print("\n" + "=" * 100)
    print("📊 A/B TESTING RESULTS COMPARISON")
    print("=" * 100)
    
    # Sort by MSE
    df_sorted = df.sort_values('mse') if 'mse' in df.columns else df
    print("\nSorted by MSE (lower is better):")
    print("-" * 100)
    print(df_sorted.to_string(index=False))
    
    # Best results
    print("\n" + "=" * 100)
    print("🏆 BEST RESULTS:")
    print("=" * 100)
    
    if 'mse' in df.columns:
        best_idx = df['mse'].idxmin()
        print(f"  Best MSE:  {df.loc[best_idx, 'mse']:.6f} -> {df.loc[best_idx, 'config']}")
    if 'ci' in df.columns:
        best_idx = df['ci'].idxmax()
        print(f"  Best CI:   {df.loc[best_idx, 'ci']:.6f} -> {df.loc[best_idx, 'config']}")
    if 'r2' in df.columns:
        best_idx = df['r2'].idxmax()
        print(f"  Best R2:   {df.loc[best_idx, 'r2']:.6f} -> {df.loc[best_idx, 'config']}")
    
    # Improvement over baseline
    baseline_rows = df[df['config'].str.contains('baseline', case=False)]
    if len(baseline_rows) > 0 and 'mse' in df.columns:
        baseline_mse = baseline_rows['mse'].values[0]
        baseline_ci = baseline_rows['ci'].values[0] if 'ci' in baseline_rows.columns else 0
        
        print("\n" + "=" * 100)
        print("📈 IMPROVEMENT OVER BASELINE:")
        print("=" * 100)
        
        for idx, row in df.iterrows():
            if 'baseline' not in row['config'].lower():
                mse_imp = (baseline_mse - row['mse']) / baseline_mse * 100
                ci_imp = (row['ci'] - baseline_ci) / baseline_ci * 100 if baseline_ci > 0 else 0
                emoji = "✅" if mse_imp > 0 else "❌"
                print(f"  {emoji} {row['config']:30s}: MSE {mse_imp:+.2f}%, CI {ci_imp:+.2f}%")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='A/B Testing for MoE on Colab')
    parser.add_argument('--dataset', type=str, default='davis', help='Dataset name')
    parser.add_argument('--running_set', type=str, default='novel-pair', help='Running set')
    parser.add_argument('--fold', type=int, default=0, help='Fold index')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device ID (default: 0 for Colab)')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer configs')
    args = parser.parse_args()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"./ab_results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 70)
    print("🧪 A/B TESTING: Baseline vs MoE")
    print("=" * 70)
    print(f"Dataset:     {args.dataset}-{args.running_set}")
    print(f"Fold:        {args.fold}")
    print(f"Epochs:      {args.epochs}")
    print(f"CUDA:        {args.cuda}")
    print(f"Results dir: {results_dir}")
    print("=" * 70)
    
    # Common arguments
    common_args = [
        "--fold", str(args.fold),
        "--dataset", args.dataset,
        "--running_set", args.running_set,
        "--epochs", str(args.epochs),
        "--cuda", args.cuda,
        "--no_wandb"
    ]
    
    # Define experiments
    experiments = [
        # Test A: Baseline (no MoE)
        ("baseline", common_args + [
            "--num_experts", "1",
            "--top_k", "1",
            "--load_balance_weight", "0"
        ]),
        
        # Test B: MoE with 4 experts, top-2
        ("moe_4exp_top2", common_args + [
            "--num_experts", "4",
            "--top_k", "2",
            "--load_balance_weight", "0.01",
            "--moe_noise_std", "0.1"
        ]),
    ]
    
    # Add more experiments if not quick mode
    if not args.quick:
        experiments.extend([
            # Test C: Sparse MoE (top-1)
            ("moe_4exp_top1", common_args + [
                "--num_experts", "4",
                "--top_k", "1",
                "--load_balance_weight", "0.02",
                "--moe_noise_std", "0.15"
            ]),
            
            # Test D: MoE with 6 experts, top-2
            ("moe_6exp_top2", common_args + [
                "--num_experts", "6",
                "--top_k", "2",
                "--load_balance_weight", "0.01",
                "--moe_noise_std", "0.1"
            ]),
            
            # Test E: MoE with 8 experts
            ("moe_8exp_top2", common_args + [
                "--num_experts", "8",
                "--top_k", "2",
                "--load_balance_weight", "0.01",
                "--moe_noise_std", "0.1"
            ]),
        ])
    
    # Run experiments
    all_results = []
    log_files = {}
    
    for config_name, exp_args in experiments:
        log_file = run_experiment(config_name, exp_args, results_dir)
        log_files[config_name] = log_file
        
        # Extract results
        results = extract_test_results(log_file)
        results['config'] = config_name
        all_results.append(results)
    
    # Print comparison
    df = print_comparison(all_results)
    
    # Save summary
    summary_file = os.path.join(results_dir, "summary.csv")
    df.to_csv(summary_file, index=False)
    print(f"\n📁 Summary saved to: {summary_file}")
    
    print("\n" + "=" * 70)
    print("🎉 A/B Testing Complete!")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    # Save full log of main execution with parameterized filename (4 params)
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--running_set', type=str, default='novel-pair')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    args, unknown = parser.parse_known_args()
    ab_results_dir = "./ab_results"
    os.makedirs(ab_results_dir, exist_ok=True)
    log_file = os.path.join(ab_results_dir, f"ab_testing_main_{args.dataset}_{args.running_set}_fold{args.fold}_epochs{args.epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_file, "w") as f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = f
        try:
            main()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    print(f"Full main log saved to: {log_file}")