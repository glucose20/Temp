import pandas as pd
from pathlib import Path

# Set the path to your main results directory
# base_path = Path("./sweep_extra_results_2")
# base_path = Path("./auto_moe_results")
# base_path = Path("./sweep_pre_results")
# base_path = Path("./ablation_results")

all_data = []

# Recursively find all 'summary.csv' files
# Use rglob to search through all subdirectories
base_paths = [Path("./sweep_array_results_0"), Path("./sweep_array_results_1")]
for base_path in base_paths:
    for file_path in base_path.rglob("summary.csv"):
    # for file_path in base_path.rglob("ablation_davis_novel-pair*.csv"):
        # Load the CSV?
        df = pd.read_csv(file_path)

        # Extract metadata from folder structure
        # Example structure: root/date_timestamp_dataset_runSet_foldX_batchsized_[...]_learningrate_moestd_lbw/summary.csv
        string = file_path.parent.name
        parts = string.split('_')
        df['dataset'] = parts[2]  # Assuming dataset is the 3rd part
        df['running_set'] = parts[3]  # Assuming runSet is the 4th part
        df['fold'] = parts[4]  # Assuming fold is the 5th
        df['batch_size'] = parts[5]  # Assuming batch size is the 6th


        df["learning_rate"] = parts[-6]  # Assuming learning rate is the 3rd last part
        df["moe_std"] = parts[-5]  # Assuming moe_std is the 2nd last part
        df["lbw"] = parts[-4]  # Assuming lbw is the last part

        df["encoder_dropout"] = parts[-3]  # Assuming encoder_dropout is the 3rd last part
        df["cross_attention_dropout"] = parts[-2]  # Assuming cross_attention_dropout is the 2nd last part
        df["expert_dropout"] = parts[-1]  # Assuming expert_dropout is the last part
        
        all_data.append(df)

# Combine all into one master DataFrame
master_df = pd.concat(all_data, ignore_index=True)

# --- Analysis Examples ---

# 1. Average performance per configuration across all folds
summary_stats = master_df.groupby(['dataset', 'running_set', 'config'])[['mse', 'rmse', 'pearson', 'spearman', 'ci', 'r2', 'final_entropy']].agg(['mean', 'std', 'count'])
# summary_stats = master_df.groupby(['variant', 'description'])[['ci', 'mse', 'r2', 'ci_diff']].agg(['mean', 'std'])
print("Summary Statistics per Configuration:")
print(summary_stats)


# 2. Find the best configuration for each dataset based on MSE
# best_configs = master_df.loc[master_df.groupby('dataset')['mse'].idxmin()]

# print("Master Dataframe Shape:", master_df.shape)
# print("\nTop 5 Results by Pearson Correlation:")
# print(master_df.sort_values(by='pearson', ascending=False).head())


# save the master dataframe to a CSV for further analysis if needed
master_df.to_csv("sweep_analyse.csv", index=False)
summary_stats.to_csv("sweep_all_summary_stats.csv")
# best_ci_per_group.to_csv("sweep_best_ci_means_pre.csv", index=False)
# summary_stats.to_csv("ablation_davis_novel-pair_AGGREGATED.csv")