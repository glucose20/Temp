# LLMDTA with Mixture of Experts (MoE)

## Overview

This implementation extends the original LLMDTA model with a **Mixture of Experts (MoE)** architecture. Instead of using a single MLP for prediction, the model uses multiple expert networks that view drug-protein interactions from different perspectives.

## Architecture

### Key Components

1. **Shared Representation Learning**
   - Drug Encoder: Processes molecular embeddings (Mol2Vec)
   - Protein Encoder: Processes protein embeddings (ESM2)
   - Bilinear Attention: Captures cross-modal interactions

2. **Mixture of Experts Layer**
   - **Gating Network**: Routes combined representation (h_pre + h_post) to appropriate experts
   - **Multiple Experts**: Each expert is an MLP that predicts binding affinity from a unique perspective
   - **Top-K Routing**: Selects the top-k most relevant experts for each input
   - **Weighted Aggregation**: Final prediction is the weighted sum of selected expert outputs

3. **Load Balancing Loss**
   - Encourages the gating network to distribute inputs evenly across all experts
   - Prevents expert collapse (where only a few experts are used)

## Model Parameters

- **num_experts**: Number of expert MLPs (default: 4)
- **top_k**: Number of experts to activate per input (default: 2)
- **lb_weight**: Weight for load balancing loss (default: 0.01)

## Usage

### Training Single Fold

```bash
python code/train_moe.py --dataset davis --running_set warm --fold 0
```

### Training All Folds

```bash
python code/train_moe.py --dataset davis --running_set warm --all_folds
```

### Command-line Arguments

- `--dataset`: Dataset name (davis, kiba, metz)
- `--running_set`: Task setting (warm, novel-drug, novel-prot, novel-pair)
- `--fold`: Fold number for cross-validation (0-4)
- `--all_folds`: Train all 5 folds and aggregate results

## Implementation Details

### Forward Pass

1. Extract drug and protein embeddings
2. Apply bilinear attention
3. Compute combined representation: `h_combined = h_pre + h_post`
4. Gating network computes routing weights
5. Select top-k experts based on routing weights
6. Each selected expert makes a prediction
7. Final prediction = weighted sum of expert predictions

### Loss Function

```python
total_loss = prediction_loss + lb_weight * load_balancing_loss
```

- **Prediction Loss**: MSE between predicted and true binding affinity
- **Load Balancing Loss**: Encourages uniform expert usage

### Advantages of MoE

1. **Specialization**: Different experts learn to handle different types of drug-protein pairs
2. **Scalability**: Can add more experts without significantly increasing computation
3. **Interpretability**: Routing weights show which experts are activated for each input
4. **Performance**: Often improves over single model, especially in cold-start scenarios

## Code Structure

```
code/
├── LLMDTA_MoE.py          # Model architecture with MoE
├── train_moe.py           # Training script
├── hyperparameter.py      # Configuration (shared with original)
└── MyDataset.py           # Dataset loader (shared with original)
```

## Example Output

```
================================================================================
Training LLMDTA with Mixture of Experts - davis/warm/fold0
================================================================================
Device: cuda:0
Loading dataset...
Train size: 24000, Valid size: 3000, Test size: 3000
Model created with 4 experts, top-2 routing

Epoch 1/200
Train Loss: 0.523456 (Pred: 0.520000, LB: 0.003456)
Valid Loss: 0.412345, MSE: 0.412345, RMSE: 0.642138, R2: 0.725432
✓ Model saved: davis-warm-fold0-MoE-Nov27_10-30-45.pth

...

================================================================================
Test Results - davis/warm/fold0
================================================================================
MSE: 0.398765
RMSE: 0.631462
CI: 0.892341
R2: 0.742156
Pearson: 0.861234
Spearman: 0.856789
================================================================================
```

## Comparison with Original LLMDTA

| Aspect | Original LLMDTA | LLMDTA-MoE |
|--------|----------------|------------|
| Prediction Head | Single MLP | Multiple Expert MLPs |
| Routing | N/A | Gating Network |
| Capacity | Fixed | Scalable (add experts) |
| Specialization | Generalist | Expert specialization |
| Interpretability | Limited | Routing analysis possible |

## Analysis Tools

The model provides `forward_with_routing_info()` method for analysis:

```python
pred, routing_weights, top_k_indices, normalized_weights = model.forward_with_routing_info(...)
```

This allows you to:
- Analyze which experts are used for specific drug-protein pairs
- Identify expert specialization patterns
- Debug routing behavior

## Tips for Tuning

1. **Number of Experts**: Start with 4-8 experts
2. **Top-K**: Usually 2-3 works well (balance between capacity and efficiency)
3. **Load Balancing Weight**: 0.001-0.1 depending on how much you want to enforce uniform usage
4. **Learning Rate**: May need slight adjustment from original (1e-4 to 5e-5)

## Requirements

Same as original LLMDTA:
- torch>=1.8.2
- numpy, pandas, scikit-learn
- RDKit, ESM2
- See `requirements.txt`

## References

Based on:
1. Original LLMDTA paper and implementation
2. MoE routing mechanism from `train_moe_gnn (1).py`
