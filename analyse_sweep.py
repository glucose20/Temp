import pandas as pd
from pathlib import Path

# Set the path to your main results directory
# base_path = Path("./sweep_array_results_3")
# base_path = Path("./auto_moe_results")
base_path = Path("./sweep_pre_results")
# base_path = Path("./ablation_results")

all_data = []

# Recursively find all 'summary.csv' files
# Use rglob to search through all subdirectories
for file_path in base_path.rglob("summary.csv"):
# for file_path in base_path.rglob("ablation_davis_novel-prot*.csv"):
    # Load the CSV?
    df = pd.read_csv(file_path)
    
    # # Extract metadata from folder structure
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
summary_stats = master_df.groupby(['dataset', 'running_set', 'config', 'learning_rate', 'batch_size'])[['mse', 'rmse', 'pearson', 'spearman', 'ci', 'r2', 'final_entropy']].agg(['mean', 'std', 'count'])
# summary_stats = master_df.groupby(['variant', 'description'])[['ci', 'mse', 'r2', 'ci_diff', 'mse_diff']].agg(['mean', 'std'])
print("Summary Statistics per Configuration:")
# print(summary_stats)

# Best CI mean per dataset + running_set
summary_stats_reset = summary_stats.reset_index()
summary_stats_reset['ci_mean'] = summary_stats_reset[('ci', 'mean')]
# Ensure we only consider entries where CI std is not NaN
summary_stats_reset['ci_std'] = summary_stats_reset[('ci', 'std')]
summary_stats_reset['ci_count'] = summary_stats_reset[('ci', 'count')]
filtered_stats = summary_stats_reset[
    (summary_stats_reset['ci_std'].notna()) & 
    (summary_stats_reset['ci_count'] == 5)
]

# Best CI mean per dataset + running_set among entries with non-NaN ci_std and ci_count == 5
best_ci_per_group = filtered_stats.loc[
    filtered_stats.groupby(['dataset', 'running_set'])['ci_mean'].idxmax()
]

# Drop all columns with 'count' in their name
best_ci_per_group = best_ci_per_group.drop(columns=[col for col in best_ci_per_group.columns if 'count' in str(col)])

print("\nBest CI mean per dataset+running_set:")
print(best_ci_per_group)

# # 2. Find the best configuration for each dataset based on MSE
# # best_configs = master_df.loc[master_df.groupby('dataset')['mse'].idxmin()]

# # print("Master Dataframe Shape:", master_df.shape)
# # print("\nTop 5 Results by Pearson Correlation:")
# # print(master_df.sort_values(by='pearson', ascending=False).head())


# # save the master dataframe to a CSV for further analysis if needed
master_df.to_csv("sweep_analyse.csv", index=False)
# summary_stats.to_csv("sweep_automoe_summary_stats_m.csv")
summary_stats.to_csv("sweep_preset_summary_stats_m.csv")
# best_ci_per_group.to_csv("sweep_best_ci_means_automoe_m.csv", index=False)
best_ci_per_group.to_csv("sweep_best_ci_means_pre_m.csv", index=False)
# summary_stats.to_csv("ablation_davis_novel-prot_AGGREGATED.csv")