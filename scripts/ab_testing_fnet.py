"""Run a fair Baseline vs MoE vs FNet+MoE comparison for one fold.

Baseline and MoE use code/train.py; FNet+MoE uses code/train_fnet.py.
Run this script from the project root (the directory containing code/).
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

import pandas as pd


CONFIGS = ("baseline", "moe", "fnet_moe")


def run_experiment(name, trainer, arguments, results_dir):
    log_file = os.path.join(results_dir, f"{name}.log")
    command = [sys.executable, trainer, *arguments]
    print(f"\n{'=' * 70}\nRunning {name}: {' '.join(command)}\n{'=' * 70}")
    with open(log_file, "w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            output.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return log_file


def extract_results(log_file):
    with open(log_file, encoding="utf-8", errors="ignore") as source:
        content = source.read()
    result = {}
    patterns = {
        "test": r"Test at fold-(\d+), mse: ([\d.eE+-]+), rmse: ([\d.eE+-]+), ci: ([\d.eE+-]+), r2: ([\d.eE+-]+), pearson: ([\d.eE+-]+), spearman: ([\d.eE+-]+)",
        "parameters": r"Model parameters: total=(\d+), trainable=(\d+)",
        "training_time": r"Training time: ([\d.]+) seconds",
        "test_time": r"Test time: ([\d.]+) seconds",
    }
    match = re.search(patterns["test"], content)
    if match:
        keys = ("fold", "mse", "rmse", "ci", "r2", "pearson", "spearman")
        values = [int(match.group(1)), *map(float, match.groups()[1:])]
        result.update(zip(keys, values))
    match = re.search(patterns["parameters"], content)
    if match:
        result.update(total_parameters=int(match.group(1)), trainable_parameters=int(match.group(2)))
    for key in ("training_time", "test_time"):
        match = re.search(patterns[key], content)
        if match:
            result[f"{key}_seconds"] = float(match.group(1))
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare baseline, MoE, and FNet+MoE")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--running_set", default="novel-pair")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_patience", type=int, default=20)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--moe_noise_std", type=float, default=0.1)
    parser.add_argument("--load_balance_weight", type=float, default=0.01)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--cross_attention_dropout", type=float, default=0.1)
    parser.add_argument("--expert_dropout", type=float, default=0.1)
    parser.add_argument("--models", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable AMP for FNet+MoE")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--results_root", default="fnet_ab_results")
    parser.add_argument('--mol_embed_type', type=str, default='mol2vec', choices=['mol2vec', 'molformer'],)
    parser.add_argument('--use_esmc', action='store_true', help='Use ESM-C (True) or ESM2 (False). Overrides hyperparameter.py setting')
    parser.add_argument('--esmc_model', type=str, default=None, choices=['esmc_300m', 'esmc_600m', 'esmc_6b', 'esm3'], help='ESM-C model variant (esmc_300m, esmc_600m, esmc_6b). Overrides hyperparameter.py setting')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(
        args.results_root,
        f"{timestamp}_{args.dataset}_{args.running_set}_fold{args.fold}",
    )
    os.makedirs(results_dir, exist_ok=True)

    common = [
        "--fold", str(args.fold), "--dataset", args.dataset,
        "--running_set", args.running_set, "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate), "--batch_size", str(args.batch_size),
        "--max_patience", str(args.max_patience), "--cuda", args.cuda,
        "--encoder_dropout", str(args.encoder_dropout),
        "--cross_attention_dropout", str(args.cross_attention_dropout),
        "--expert_dropout", str(args.expert_dropout),
        "--mol_embed_type", args.mol_embed_type,
        "--esmc_model", args.esmc_model,
    ]
    if args.use_esmc:
        common.append("--use_esmc")
    if args.no_wandb:
        common.append("--no_wandb")
    moe = [
        "--num_experts", str(args.num_experts), "--top_k", str(args.top_k),
        "--moe_noise_std", str(args.moe_noise_std),
        "--load_balance_weight", str(args.load_balance_weight),
    ]
    specifications = {
        "baseline": ("code/train.py", common + [
            "--num_experts", "1", "--top_k", "1",
            "--moe_noise_std", "0", "--load_balance_weight", "0",
        ]),
        "moe": ("code/train.py", common + moe),
        "fnet_moe": ("code/train_fnet.py", common + moe),
    }
    if args.amp:
        specifications["fnet_moe"][1].extend(["--amp", "--amp_dtype", args.amp_dtype])

    rows = []
    for name in args.models:
        trainer, arguments = specifications[name]
        log_file = run_experiment(name, trainer, arguments, results_dir)
        rows.append({"config": name, "trainer": trainer, **extract_results(log_file)})

    summary = pd.DataFrame(rows)
    summary_file = os.path.join(results_dir, "summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"\n{summary.to_string(index=False)}\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
