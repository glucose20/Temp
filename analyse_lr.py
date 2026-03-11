import csv
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# =========================
# USER CONFIG (EDIT HERE)
# =========================


# FOLDER_PATTERN = "sweep_ab_metz_novel-drug_fold0_b256" # dataset_runningSet
FOLDER_PATTERN = "sweep_ab_metz_novel-pair_fold0_b256" # dataset_runningSet
# FOLDER_PATTERN = "sweep_ab_metz_novel-prot_fold0_b256" # dataset_runningSet
# FOLDER_PATTERN = "sweep_ab_metz_warm_fold0_b256" # dataset_runningSet

# FOLDER_PATTERN = "davis_novel-drug_fold0_b256" # dataset_runningSet
# FOLDER_PATTERN = "davis_novel-pair_fold0_b256" # dataset_runningSet
# FOLDER_PATTERN = "davis_novel-prot_fold0_b256" # dataset_runningSet
# FOLDER_PATTERN = "davis_warm_b256" # dataset_runningSet


# FOLDER_PATTERN = "kiba_novel-drug_fold0_b256" # dataset_runningSet
# FOLDER_PATTERN = "kiba_novel-pair_fold0_b256" # dataset_runningSet
# FOLDER_PATTERN = "kiba_novel-prot_fold0_b256" # dataset_runningSet
# FOLDER_PATTERN = "kiba_warm_fold0_b256" # dataset_runningSet


ROOT_DIR = Path("sweep_results")          # root directory containing all experiment folders
CONFIG_PREFIX = "moe_"        # only configs starting with this
METRIC = "ci"                 # metric to MAXIMISE
OUT_DIR = Path("analysis_results")   # output directory
# =========================


# Example folder:
# <jobid>_<timestamp>_davis_warm_b256_lr5e-4_moestd0.1_lbw0.01
FOLDER_RE = re.compile(
    r"""
    ^[^_]+_[^_]+_
    (?P<dataset>[A-Za-z0-9-]+)_
    (?P<runset>[A-Za-z0-9-]+)
    (?P<rest>.*)$
    """,
    re.VERBOSE,
)

LR_RE = re.compile(
    r"(?:^|_)lr(?P<lr>[0-9.]+(?:e-?\d+)?)(?:_|$)",
    re.IGNORECASE,
)


METRICS = ["mse","rmse","r2","pearson","spearman"]
 
@dataclass
class Candidate:
    dataset: str
    runset: str
    config: str
    lr: str
    ci: float
    folder: str
    log_file: str
    metrics : Dict[str, float] = None


def parse_folder(folder_name: str) -> Optional[Tuple[str, str, str]]:
    m = FOLDER_RE.match(folder_name)
    if not m:
        return None

    lr_match = LR_RE.search(folder_name)
    if not lr_match:
        return None

    return (
        m.group("dataset"),
        m.group("runset"),
        lr_match.group("lr"),
    )


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    folders = [
        p for p in ROOT_DIR.iterdir()
        if p.is_dir() and FOLDER_PATTERN in p.name
    ]

    candidates: List[Candidate] = []

    for folder in folders:
        parsed = parse_folder(folder.name)
        if not parsed:
            continue

        dataset, runset, lr = parsed

        for csv_file in folder.glob("*.csv"): #only summary is csv
            for row in read_csv(csv_file):
                cfg = row.get("config", "").strip()

                # uncomment this to compare with baseline scores
                # if not cfg.startswith(CONFIG_PREFIX):
                #     continue

                ci_val = row.get("ci", "").strip()
                if not ci_val:
                    continue

                try:
                    ci = float(ci_val)
                except ValueError:
                    continue

                metrics = {k: float(v) for k, v in row.items() if k in METRICS and v}

                candidates.append(
                    Candidate(
                        dataset=dataset,
                        runset=runset,
                        config=cfg,
                        lr=lr,
                        ci=ci,
                        folder=folder.name,
                        log_file=csv_file.name,
                        metrics=metrics
                    )
                )

    if not candidates:
        print("⚠ No valid logs found")
        return

    # Pick BEST (MAX ci) per config
    best: Dict[str, Candidate] = {}
    for c in candidates:
        if c.config not in best or c.ci > best[c.config].ci:
            best[c.config] = c

    # Write output
    out_path = OUT_DIR / f"presweep_best_exp-lr_{FOLDER_PATTERN}_by_ci.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "running_set",
                "config",
                "best_lr",
                "best_ci",
                "source_folder",
                "source_log",
                *METRICS,
            ],
        )
        writer.writeheader()
        for c in sorted(best.values(), key=lambda x: x.config):
            writer.writerow(
                {
                    "dataset": c.dataset,
                    "running_set": c.runset,
                    "config": c.config,
                    "best_lr": c.lr,
                    "best_ci": f"{c.ci:.6f}",
                    "source_folder": c.folder,
                    "source_log": c.log_file,
                    **{k: f"{v:.6f}" for k, v in c.metrics.items()},
                }
            )

    print(f"✓ Written: {out_path}")
    print(f"✓ Configs summarised: {len(best)}")
    print(f"✓ Len candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
