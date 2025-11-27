"""
Analyze the routing behavior of LLMDTA-MoE model.
This script helps understand which experts specialize in which types of drug-protein interactions.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from LLMDTA_MoE import LLMDTA_MoE
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn


def analyze_expert_usage(model, dataloader, device):
    """Analyze which experts are used most frequently."""
    model.eval()
    
    expert_usage_count = np.zeros(model.num_experts)
    total_samples = 0
    
    with torch.no_grad():
        for data in dataloader:
            drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, label = [d.to(device) for d in data]
            
            pred, routing_weights, top_k_indices, normalized_weights = model.forward_with_routing_info(
                drug, drug_mat, drug_mask, protein, prot_mat, prot_mask
            )
            
            # Count expert usage
            for i in range(top_k_indices.size(0)):
                for k in range(model.top_k):
                    expert_idx = top_k_indices[i, k].item()
                    expert_usage_count[expert_idx] += 1
            
            total_samples += top_k_indices.size(0)
    
    # Normalize to percentage
    expert_usage_pct = (expert_usage_count / (total_samples * model.top_k)) * 100
    
    return expert_usage_count, expert_usage_pct


def analyze_routing_distribution(model, dataloader, device, num_samples=1000):
    """Analyze the distribution of routing weights."""
    model.eval()
    
    all_routing_weights = []
    all_labels = []
    all_predictions = []
    all_top_k_indices = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for data in dataloader:
            if samples_collected >= num_samples:
                break
                
            drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, label = [d.to(device) for d in data]
            
            pred, routing_weights, top_k_indices, normalized_weights = model.forward_with_routing_info(
                drug, drug_mat, drug_mask, protein, prot_mat, prot_mask
            )
            
            batch_size = routing_weights.size(0)
            samples_to_take = min(batch_size, num_samples - samples_collected)
            
            all_routing_weights.append(routing_weights[:samples_to_take].cpu().numpy())
            all_labels.append(label[:samples_to_take].cpu().numpy())
            all_predictions.append(pred[:samples_to_take].cpu().numpy())
            all_top_k_indices.append(top_k_indices[:samples_to_take].cpu().numpy())
            
            samples_collected += samples_to_take
    
    routing_weights = np.vstack(all_routing_weights)
    labels = np.vstack(all_labels).flatten()
    predictions = np.vstack(all_predictions).flatten()
    top_k_indices = np.vstack(all_top_k_indices)
    
    return routing_weights, labels, predictions, top_k_indices


def plot_expert_usage(expert_usage_pct, save_path='expert_usage.png'):
    """Plot expert usage distribution."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(expert_usage_pct)), expert_usage_pct, color='steelblue', alpha=0.8)
    plt.xlabel('Expert Index', fontsize=12)
    plt.ylabel('Usage Percentage (%)', fontsize=12)
    plt.title('Expert Usage Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(expert_usage_pct)))
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, pct in enumerate(expert_usage_pct):
        plt.text(i, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Expert usage plot saved to {save_path}")
    plt.close()


def plot_routing_heatmap(routing_weights, save_path='routing_heatmap.png'):
    """Plot heatmap of routing weights."""
    plt.figure(figsize=(12, 8))
    
    # Sample subset if too many samples
    if routing_weights.shape[0] > 100:
        indices = np.random.choice(routing_weights.shape[0], 100, replace=False)
        routing_weights_subset = routing_weights[indices]
    else:
        routing_weights_subset = routing_weights
    
    sns.heatmap(routing_weights_subset.T, cmap='YlOrRd', cbar_kws={'label': 'Routing Weight'},
                xticklabels=False, yticklabels=[f'Expert {i}' for i in range(routing_weights.shape[1])])
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Expert', fontsize=12)
    plt.title('Routing Weights Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Routing heatmap saved to {save_path}")
    plt.close()


def analyze_expert_specialization(routing_weights, labels, predictions, top_k_indices, num_experts):
    """Analyze what types of interactions each expert specializes in."""
    expert_stats = {i: {'labels': [], 'errors': [], 'count': 0} for i in range(num_experts)}
    
    for i in range(len(labels)):
        for k in range(top_k_indices.shape[1]):
            expert_idx = top_k_indices[i, k]
            expert_stats[expert_idx]['labels'].append(labels[i])
            expert_stats[expert_idx]['errors'].append(abs(predictions[i] - labels[i]))
            expert_stats[expert_idx]['count'] += 1
    
    # Compute statistics for each expert
    results = []
    for expert_idx in range(num_experts):
        if expert_stats[expert_idx]['count'] > 0:
            labels_arr = np.array(expert_stats[expert_idx]['labels'])
            errors_arr = np.array(expert_stats[expert_idx]['errors'])
            
            results.append({
                'Expert': expert_idx,
                'Usage Count': expert_stats[expert_idx]['count'],
                'Avg Label': np.mean(labels_arr),
                'Std Label': np.std(labels_arr),
                'Avg Error': np.mean(errors_arr),
                'Median Label': np.median(labels_arr)
            })
    
    df = pd.DataFrame(results)
    return df


def main():
    # Configuration
    hp = HyperParameter()
    hp.dataset = 'davis'
    hp.running_set = 'warm'
    fold = 0
    
    model_path = './savemodel/davis-warm-fold0-MoE-Nov27_10-30-45.pth'  # Update with your model path
    
    # Device setup
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = LLMDTA_MoE(hp, device, num_experts=4, top_k=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = CustomDataSet(hp, fold, 'test')
    test_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False,
                             collate_fn=my_collate_fn, num_workers=0)
    
    # Analysis 1: Expert Usage
    print("\n" + "="*80)
    print("ANALYSIS 1: Expert Usage Distribution")
    print("="*80)
    expert_usage_count, expert_usage_pct = analyze_expert_usage(model, test_loader, device)
    
    for i, (count, pct) in enumerate(zip(expert_usage_count, expert_usage_pct)):
        print(f"Expert {i}: {int(count)} times ({pct:.2f}%)")
    
    plot_expert_usage(expert_usage_pct, f'results/expert_usage_{hp.dataset}_{hp.running_set}_fold{fold}.png')
    
    # Analysis 2: Routing Distribution
    print("\n" + "="*80)
    print("ANALYSIS 2: Routing Weight Distribution")
    print("="*80)
    routing_weights, labels, predictions, top_k_indices = analyze_routing_distribution(
        model, test_loader, device, num_samples=500
    )
    
    print(f"Analyzed {len(labels)} samples")
    print(f"Routing weights shape: {routing_weights.shape}")
    print(f"Average routing weights per expert:")
    for i in range(routing_weights.shape[1]):
        print(f"  Expert {i}: {routing_weights[:, i].mean():.4f} ± {routing_weights[:, i].std():.4f}")
    
    plot_routing_heatmap(routing_weights, f'results/routing_heatmap_{hp.dataset}_{hp.running_set}_fold{fold}.png')
    
    # Analysis 3: Expert Specialization
    print("\n" + "="*80)
    print("ANALYSIS 3: Expert Specialization")
    print("="*80)
    specialization_df = analyze_expert_specialization(
        routing_weights, labels, predictions, top_k_indices, model.num_experts
    )
    
    print(specialization_df.to_string(index=False))
    specialization_df.to_csv(f'results/expert_specialization_{hp.dataset}_{hp.running_set}_fold{fold}.csv', index=False)
    
    # Analysis 4: Routing vs Performance
    print("\n" + "="*80)
    print("ANALYSIS 4: Correlation between Routing Entropy and Prediction Error")
    print("="*80)
    
    # Calculate entropy of routing weights
    entropy = -np.sum(routing_weights * np.log(routing_weights + 1e-10), axis=1)
    errors = np.abs(predictions - labels)
    
    correlation = np.corrcoef(entropy, errors)[0, 1]
    print(f"Correlation between routing entropy and prediction error: {correlation:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(entropy, errors, alpha=0.5, s=20)
    plt.xlabel('Routing Entropy', fontsize=12)
    plt.ylabel('Absolute Prediction Error', fontsize=12)
    plt.title('Routing Entropy vs Prediction Error', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/entropy_vs_error_{hp.dataset}_{hp.running_set}_fold{fold}.png', dpi=300, bbox_inches='tight')
    print(f"Entropy vs error plot saved")
    plt.close()
    
    print("\n" + "="*80)
    print("Analysis complete! Check the 'results/' folder for plots and CSV files.")
    print("="*80)


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    main()
