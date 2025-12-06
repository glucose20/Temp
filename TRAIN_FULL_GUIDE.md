# Training on Full Dataset (Without Fold Split)

This guide explains how to train models on the full dataset to create `All-davis`, `All-kiba`, or `All-metz` models.

## Files

- `hyperparameter_full.py` - Configuration for full dataset training
- `train_full.py` - Training script for full dataset

## Key Differences from Standard Training

| Standard Training | Full Dataset Training |
|------------------|----------------------|
| Uses `dta-5fold-dataset/` | Uses `dta-origin-dataset/` |
| 5-fold cross-validation | Single train/valid split (80/20) |
| Cold-start splits | Random split |
| Output: `{dataset}-{setting}-fold{i}.pth` | Output: `All-{dataset}.pth` |

## Prerequisites

1. **Dataset files** in `data/dta-origin-dataset/`:
   ```
   davis.txt
   kiba.txt
   metz.txt
   ```

2. **Pretrained features**:
   ```
   data/davis/davis_drug_pretrain.pkl
   data/davis/davis_esm_pretrain.pkl
   data/kiba/kiba_drug_pretrain.pkl
   data/kiba/kiba_esm_pretrain.pkl
   data/metz/metz_drug_pretrain.pkl
   data/metz/metz_esm_pretrain.pkl
   ```

## Usage

### Basic Training

Train on Davis dataset:
```bash
cd code
python train_full.py --dataset davis
```

Train on KIBA dataset:
```bash
python train_full.py --dataset kiba
```

Train on Metz dataset:
```bash
python train_full.py --dataset metz
```

### Advanced Options

Specify GPU:
```bash
python train_full.py --dataset davis --cuda 0
```

Adjust hyperparameters:
```bash
python train_full.py --dataset davis --epochs 200 --batch_size 128 --lr 1e-4
```

Full command with all options:
```bash
python train_full.py \
  --dataset davis \
  --cuda 0 \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-4
```

## Configuration

Edit `hyperparameter_full.py` to change default settings:

```python
# Training parameters
self.Learning_rate = 1e-4
self.Epoch = 200
self.Batch_size = 256
self.max_patience = 20

# Data split ratio
self.train_ratio = 0.8
self.valid_ratio = 0.2

# Model architecture
self.drug_max_len = 100
self.prot_max_len = 1022
self.mol2vec_dim = 300
self.protvec_dim = 1280
self.latent_dim = 512
self.com_dim = 2048
```

## Output Files

After training, you will get:

1. **Model file**: `savemodel/All-{dataset}-{timestamp}.pth`
   - Contains trained model weights
   - Can be used for inference with `pred.py`

2. **Training log**: `log/All-{dataset}-{timestamp}-train.csv`
   - Contains epoch-by-epoch metrics
   - Columns: epoch, mse, rmse, ci, r2, pearson, spearman

## Example Training Session

```bash
$ cd code
$ python train_full.py --dataset davis --cuda 0

================================================================================
Training LLMDTA on Full DAVIS Dataset
================================================================================
Dataset: davis
Data root: ./data/dta-origin-dataset
Running set: all
Device: cuda:0
Epochs: 200
Batch size: 256
Learning rate: 0.0001
================================================================================

Loaded 30056 samples from ./data/dta-origin-dataset/davis.txt
Train samples: 24044
Valid samples: 6012

Model will be saved to: ./savemodel/All-davis-Dec06_10-30-45.pth

================================================================================
Epoch 1/200
================================================================================
Training: 100%|████████████████| 94/94 [01:23<00:00,  1.13it/s]

Training Results:
  MSE: 0.523456
  RMSE: 0.723456
  R²: 0.654321
  Pearson: 0.812345
  Spearman: 0.798765

Validation Results:
  MSE: 0.489123
  RMSE: 0.699373
  R²: 0.678901
  Pearson: 0.824567
  Spearman: 0.810234

✓ Best model updated! MSE improved to 0.489123
  Model saved to: ./savemodel/All-davis-Dec06_10-30-45.pth

...

================================================================================
Training Completed!
================================================================================
Best validation MSE: 0.412345
Model saved to: ./savemodel/All-davis-Dec06_10-30-45.pth
Training log saved to: ./log/All-davis-Dec06_10-30-45-train.csv
================================================================================
```

## Using the Trained Model

After training, update `hyperparameter4pred.py` to use your model:

```python
self.model_fromTrain = './savemodel/All-davis-Dec06_10-30-45.pth'
```

Then run prediction:
```bash
python pred.py
```

## Tips

1. **GPU Memory**: If you get OOM errors, reduce `batch_size`:
   ```bash
   python train_full.py --dataset davis --batch_size 128
   ```

2. **Training Time**: Full dataset training takes longer than fold training:
   - Davis: ~30K samples → ~2-3 hours on V100
   - KIBA: ~118K samples → ~8-10 hours on V100
   - Metz: ~30K samples → ~2-3 hours on V100

3. **Early Stopping**: Training stops if validation MSE doesn't improve for 20 epochs (adjustable via `max_patience`)

4. **Reproducibility**: Fixed random seed (SEED=0) ensures reproducible results

## Comparison: Full vs Fold Training

**Use Full Training when:**
- Creating a general-purpose model for production
- Need best overall performance without cold-start evaluation
- Want to use all available data for training

**Use Fold Training when:**
- Evaluating cold-start performance (novel drugs/proteins/pairs)
- Need robust cross-validation results
- Comparing different model architectures or hyperparameters

## Troubleshooting

**FileNotFoundError: dataset not found**
- Make sure `davis.txt`/`kiba.txt`/`metz.txt` exists in `data/dta-origin-dataset/`
- Download from: https://www.kaggle.com/datasets/christang0002/llmdta/data

**FileNotFoundError: pretrain feature not found**
- Run `code_prepareEmb/_PreparePretrain.ipynb` to generate pretrained features
- Or download pre-computed features from Kaggle dataset

**CUDA out of memory**
- Reduce batch size: `--batch_size 128` or `--batch_size 64`
- Use smaller GPU or CPU (slower): remove `--cuda` flag

**Training too slow**
- Use GPU instead of CPU
- Increase batch size if GPU memory allows: `--batch_size 512`
- Reduce number of threads: Edit `torch.set_num_threads(2)` in `train_full.py`
