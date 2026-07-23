import os
import random
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from LLMDTA import LLMDTA as LLMDTA
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, batch2tensor, my_collate_fn

from sklearn.metrics import r2_score
from tqdm import tqdm
from math import sqrt
from scipy import stats
import csv
import wandb


# ----------------------------------------------------------------------------
# Bien the B (IMPROVE_FNet_Proposal.md, Tier 1-(3)): LearnableFourierMixing thay
# the CrossAttention -- giong het FourierCrossMixing cua train_fnet.py (2D DFT
# theo chieu hidden roi chieu sequence) nhung nhan them 1 FILTER PHUC HOC DUOC
# (complex_weight, shape seq_len x hidden_dim) vao pho truoc buoc bien doi
# nguoc (GFNet-style, arXiv:2107.00645). Khoi tao filter gan (re=1, im=0) de
# hanh vi luc bat dau train ~ giong het FourierCrossMixing khong tham so, roi
# hoc dan do lech -- day la "buoc ban tham so" ma REPORT_Fourier_Mixing_DTA.md
# S6 da tu de xuat nhung chua cai dat.
#
# Phan con lai cua LLMDTA (Encoder, MoE head) giu nguyen khong doi so voi
# train_fnet.py de dam bao so sanh cong bang -- chi doi 1 bien.
# ----------------------------------------------------------------------------

class LearnableFourierMixing(nn.Module):
    """GFNet-style: FFT 2 chieu (hidden roi seq) + filter phuc hoc duoc tren
    mien tan so, roi IFFT 2 chieu, lay phan thuc. Khac FourierCrossMixing (ban
    khong tham so trong train_fnet.py) o cho co them self.complex_weight
    duoc nhan element-wise vao pho truoc buoc bien doi nguoc."""

    def __init__(self, hidden_dim, seq_len, dropout=0.1):
        super().__init__()
        # Khoi tao gan (re=1, im=0) => filter ~ identity luc bat dau train
        # (on dinh hon random init, tranh pha vo tin hieu ngay tu epoch dau).
        weight = torch.zeros(seq_len, hidden_dim, 2)
        weight[..., 0] = 1.0
        weight = weight + torch.randn_like(weight) * 0.02
        self.complex_weight = nn.Parameter(weight)
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        seq_q = query.shape[1]
        combined = torch.cat([query, key_value], dim=1)  # (b, L, d)
        L = combined.shape[1]
        x = torch.fft.fft(combined, dim=-1)   # mix theo hidden dim (giong FNet goc)
        x = torch.fft.fft(x, dim=-2)          # mix theo sequence dim (giong FNet goc)
        weight = torch.view_as_complex(self.complex_weight[:L])   # (L, d) complex
        x = x * weight.unsqueeze(0)           # <-- diem khac biet duy nhat so voi FourierCrossMixing
        x = torch.fft.ifft(x, dim=-2)
        x = torch.fft.ifft(x, dim=-1)
        mixed = x.real
        mixed_query = mixed[:, :seq_q, :]
        mixed_query = self.dropout(mixed_query)
        return self.out_ln(mixed_query + query)


class LLMDTA_FNet_Learnable(LLMDTA):
    """LLMDTA (MoE) voi CrossAttention duoc thay bang LearnableFourierMixing."""

    def __init__(self, hp, device):
        super().__init__(hp, device)
        seq_len = hp.substructure_max_len + hp.prot_max_len
        self.drug_cross_attn = LearnableFourierMixing(self.hidden_dim, seq_len, self.cross_attention_dropout)
        self.prot_cross_attn = LearnableFourierMixing(self.hidden_dim, seq_len, self.cross_attention_dropout)


def cindex_score(y, p):
    sum_m = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    sum_m += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    if pair != 0:
        return sum_m / pair
    else:
        return 0

def regression_scores(label, pred, is_valid=True):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    mse = ((label - pred)**2).mean(axis=0)
    rmse = sqrt(mse)
    if is_valid:
        ci = -1
    else:
        ci = cindex_score(label, pred)
    r2 = r2_score(label, pred)
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(mse, 6), round(rmse, 6), round(ci, 6), round(r2, 6), round(pearson, 6), round(spearman, 6)

def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)

def test(model, dataloader, is_valid=True):
    model.eval()
    preds = []
    labels = []
    for batch_i, batch_data in enumerate(dataloader):
        mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask, affinity = batch_data
        with torch.no_grad():
            # Handle both DataParallel and regular model
            if hasattr(model, 'module'):
                pred = model.module.forward(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=False)
            else:
                pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist()

    preds = np.array(preds)
    labels = np.array(labels)
    mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(labels, preds, is_valid)
    return mse_value, rmse_value, ci, r2, pearson_value, spearman_value



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LLMDTA_FNet_Learnable (MoE + LearnableFourierMixing, GFNet-style) for a specific fold')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold index to train (0-4 for 5-fold CV)')
    parser.add_argument('--cuda', type=str, default=None,
                        help='CUDA device ID (e.g., "0", "1"). Overrides hyperparameter.py setting')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name to override hyperparameter.py setting')
    parser.add_argument('--running_set', type=str, default=None,
                        help='Running set to override hyperparameter.py setting')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides hyperparameter.py setting)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size to use (overrides hyperparameter.py setting)')
    parser.add_argument('--max_patience', type=int, default=None,
                        help='Max patience for early stopping (overrides hyperparameter.py setting)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate to use (overrides hyperparameter.py setting)')
    parser.add_argument('--wandb_project', type=str, default='LLMDTA',
                        help='Weights & Biases project name (default: LLMDTA)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/username (optional)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--use_esmc', type=lambda x: x.lower() == 'true', default=None,
                        help='Use ESM-C (True) or ESM2 (False). Overrides hyperparameter.py setting')
    parser.add_argument('--esmc_model', type=str, default=None, choices=['esmc_300m', 'esmc_600m', 'esmc_6b'],
                        help='ESM-C model variant (esmc_300m, esmc_600m, esmc_6b). Overrides hyperparameter.py setting')
    parser.add_argument('--encoder_dropout', type=float, default=None,
                        help='Dropout rate for encoder layers (overrides hyperparameter.py setting)')
    parser.add_argument('--cross_attention_dropout', type=float, default=None,
                        help='Dropout rate for the LearnableFourierMixing block (overrides hyperparameter.py setting)')
    parser.add_argument('--expert_dropout', type=float, default=None,
                        help='Dropout rate for expert networks (overrides hyperparameter.py setting)')
    # MoE parameters
    parser.add_argument('--num_experts', type=int, default=None,
                        help='Number of experts in MoE (default: 4)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Number of experts to select per sample (default: 2)')
    parser.add_argument('--moe_noise_std', type=float, default=None,
                        help='Noise std for MoE exploration (default: 0.1, 0 to disable)')
    parser.add_argument('--load_balance_weight', type=float, default=None,
                        help='Weight for load balancing loss (default: 0.01, 0 to disable)')
    # Mixed precision (khuyen nghi tren H100: bf16 autocast, khong can GradScaler)
    parser.add_argument('--amp', action='store_true',
                        help='Bat mixed precision autocast (khuyen nghi tren H100)')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help='bf16 khuyen nghi tren H100 (khong can GradScaler)')
    args = parser.parse_args()

    fold_i = args.fold

    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)

    hp = HyperParameter()

    # Override ESM settings BEFORE dataset (important for path resolution)
    if args.use_esmc is not None:
        hp.use_esmc = args.use_esmc
        # Update dimension when switching between ESM2 and ESM-C
        if not hp.use_esmc:
            hp.protvec_dim = 1280  # ESM2

    if args.esmc_model is not None:
        hp.esmc_model = args.esmc_model
        # Update dimension based on model
        if hp.esmc_model == "esmc_300m":
            hp.protvec_dim = 960
        elif hp.esmc_model == "esmc_600m":
            hp.protvec_dim = 1152
        elif hp.esmc_model == "esmc_6b":
            hp.protvec_dim = 2560

    # Override CUDA device if specified
    if args.cuda is not None:
        hp.cuda = args.cuda

    # Override dataset if specified (this will update paths based on use_esmc)
    if args.dataset is not None:
        hp.set_dataset(args.dataset)
    else:
        # If dataset not overridden but ESM settings changed, update paths for current dataset
        if args.use_esmc is not None or args.esmc_model is not None:
            hp.set_dataset(hp.dataset)

    if args.running_set is not None:
        # Convert underscores to hyphens (data directories use hyphens)
        hp.running_set = args.running_set.replace('_', '-')
    # Override epochs if specified
    if args.epochs is not None:
        hp.Epoch = args.epochs
    # Override batch size if specified
    if args.batch_size is not None:
        hp.Batch_size = args.batch_size
    # Override max patience if specified
    if args.max_patience is not None:
        hp.max_patience = args.max_patience
    # Override learning rate if specified
    if args.learning_rate is not None:
        hp.Learning_rate = args.learning_rate
    # Override MoE parameters if specified
    if args.num_experts is not None:
        hp.num_experts = args.num_experts
    if args.top_k is not None:
        hp.top_k = args.top_k
    if args.moe_noise_std is not None:
        hp.moe_noise_std = args.moe_noise_std
    if args.load_balance_weight is not None:
        hp.load_balance_weight = args.load_balance_weight
    # Override dropout rates if specified
    if args.encoder_dropout is not None:
        hp.encoder_dropout = args.encoder_dropout
    if args.cross_attention_dropout is not None:
        hp.cross_attention_dropout = args.cross_attention_dropout
    if args.expert_dropout is not None:
        hp.expert_dropout = args.expert_dropout

    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    use_amp = args.amp and device.type == 'cuda'
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print(f"=" * 60)
    print(f"Training Fold {fold_i}/{hp.kfold-1}  [model = FNet + Learnable Fourier Filter (MoE), GFNet-style]")
    print(f"Dataset: {hp.dataset}-{hp.running_set}")
    print(f"Training config: {hp.Learning_rate}-{hp.Batch_size}-{hp.Epoch}-patience{hp.max_patience}")
    print("Dropout rates: encoder-{:.2f}, fourier-mixing-{:.2f}, expert-{:.2f}".format(
        hp.encoder_dropout, hp.cross_attention_dropout, hp.expert_dropout))
    print(f"ESM Model: {'ESM-C-' + hp.esmc_model if hp.use_esmc else 'ESM2'} (dim={hp.protvec_dim})")
    print(f"MoE: num_experts={hp.num_experts}, top_k={hp.top_k}, noise={hp.moe_noise_std}, lb_weight={hp.load_balance_weight}")
    print(f"Device: {device} (CUDA_VISIBLE_DEVICES={hp.cuda})")
    print(f"AMP: {'on (' + args.amp_dtype + ')' if use_amp else 'off'}")
    print(f"Pretrain-{hp.mol2vec_dir}")
    print(f"Pretrain-{hp.protvec_dir}")
    print(f"=" * 60)

    # Initialize Weights & Biases
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_config = {
            'model': 'fnet_learnable',
            'dataset': hp.dataset,
            'running_set': hp.running_set,
            'fold': fold_i,
            'epochs': hp.Epoch,
            'batch_size': hp.Batch_size,
            'patience': hp.max_patience,
            'learning_rate': hp.Learning_rate,
            'max_patience': hp.max_patience,
            'cuda_device': hp.cuda,
            'use_esmc': hp.use_esmc,
            'esmc_model': hp.esmc_model if hp.use_esmc else None,
            'protvec_dim': hp.protvec_dim,
            'num_experts': hp.num_experts,
            'top_k': hp.top_k,
            'moe_noise_std': hp.moe_noise_std,
            'load_balance_weight': hp.load_balance_weight,
            'encoder_dropout': hp.encoder_dropout,
            'cross_attention_dropout': hp.cross_attention_dropout,
            'expert_dropout': hp.expert_dropout,
            'amp': use_amp,
            'amp_dtype': args.amp_dtype if use_amp else None,
        }

        esm_name = f"esmc-{hp.esmc_model}" if hp.use_esmc else "esm2"
        exp_name = f"{ hp.num_experts}exp.top{hp.top_k}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"fnet-learnable-{hp.dataset}-{hp.running_set}-fold{fold_i}-{exp_name}-b{hp.Batch_size}-lr{hp.Learning_rate}-lbw{hp.load_balance_weight}-noise{hp.moe_noise_std}",
            config=wandb_config,
            tags=[hp.dataset, hp.running_set, exp_name, f'fold{fold_i}', 'fnet_learnable'],
            reinit=True
        )
        print(f"Weights & Biases initialized: {args.wandb_project}")
    else:
        print("Weights & Biases logging disabled")

    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)

    # Validate fold index
    if fold_i < 0 or fold_i >= hp.kfold:
        raise ValueError(f"Fold index must be between 0 and {hp.kfold-1}, got {fold_i}")

    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)

    # Load data for the specified fold
    train_dir = os.path.join(dataset_root, f'fold_{fold_i}_train.csv')
    valid_dir = os.path.join(dataset_root, f'fold_{fold_i}_valid.csv')
    test_dir = os.path.join(dataset_root, f'fold_{fold_i}_test.csv')

    print(f"Loading fold {fold_i} data...")
    print(f"  Train: {train_dir}")
    print(f"  Valid: {valid_dir}")
    print(f"  Test:  {test_dir}")

    train_set = CustomDataSet(pd.read_csv(train_dir, sep=','), hp)
    valid_set = CustomDataSet(pd.read_csv(valid_dir, sep=','), hp)
    test_set = CustomDataSet(pd.read_csv(test_dir, sep=','), hp)
    train_dataset_load = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    valid_dataset_load = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    test_dataset_load = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict))
    print(f"Dataset loaded: {len(train_set)} train, {len(valid_set)} valid, {len(test_set)} test samples")

    model = nn.DataParallel(LLMDTA_FNet_Learnable(hp, device))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")

    # 1. Use AdamW for better Weight Decay performance
    # weight_decay usually ranges from 1e-4 to 1e-2
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp.Learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # 2. Initialize Cosine Annealing Scheduler
    # T_max is the total number of epochs. eta_min is the minimum LR it will drop to.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=hp.Epoch,
        eta_min=1e-6
    )

    criterion = F.mse_loss

    train_log = []
    best_valid_mse = float('inf')  # Initialize with infinity instead of 10
    patience = 0

    # Use consistent timestamp for all files
    timestamp = hp.current_time
    model_fromTrain = f'./savemodel/{hp.dataset}-{hp.running_set}-fnet_learnable-fold{fold_i}-{timestamp}.pth'

    # Create savemodel directory if not exists
    os.makedirs('./savemodel', exist_ok=True)

    print(f"Model will be saved to: {model_fromTrain}")

    for epoch in range(1, hp.Epoch + 1):
        # Reset MoE usage stats at start of each epoch
        if hasattr(model.module, 'reset_usage_stats'):
            model.module.reset_usage_stats()

        # trainning
        model.train()
        pred = []
        label = []
        total_load_balance_loss = 0.0
        num_batches = 0
        for batch_data in train_dataset_load:
            mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask, affinity = batch_data
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                predictions, gate_info = model.module.forward(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True)

                # Compute main loss
                loss = criterion(predictions.squeeze(), affinity)

                # Optional: Add load balancing loss to encourage diverse expert usage
                load_balance_loss = model.module.compute_load_balance_loss(gate_info['gate_weights'])

                # Combine losses (use hp.load_balance_weight, set to 0 to disable)
                total_loss = loss + hp.load_balance_weight * load_balance_loss

            pred = pred + predictions.detach().float().cpu().numpy().reshape(-1).tolist()
            label = label + affinity.detach().float().cpu().numpy().reshape(-1).tolist()
            total_load_balance_loss += load_balance_loss.item()
            num_batches += 1

            optimizer.zero_grad()
            if use_scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
        pred = np.array(pred)
        label= np.array(label)
        mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(pred, label)
        train_log.append([mse_value, rmse_value, ci, r2, pearson_value, spearman_value])

        # Get MoE expert usage statistics
        moe_stats = model.module.get_expert_usage_stats() if hasattr(model.module, 'get_expert_usage_stats') else None
        avg_load_balance_loss = total_load_balance_loss / num_batches if num_batches > 0 else 0

        print(f'Traing Log at fold-{fold_i} epoch-{epoch}: mse-{mse_value}, rmse-{rmse_value}, r2-{r2}')
        if moe_stats:
            print(f'  MoE Stats: usage_rate={moe_stats["expert_usage_rate"]}, entropy={moe_stats["usage_entropy"]:.4f}, dominant_expert={moe_stats["dominant_expert"]}, load_balance_loss={avg_load_balance_loss:.4f}')

        # Adaptive MoE parameter adjustment
        if hasattr(model.module, 'adaptive_update'):
            adjustments = model.module.adaptive_update(epoch, hp.Epoch, moe_stats)
            # Update hp.load_balance_weight for next epoch
            if 'load_balance_weight' in adjustments:
                hp.load_balance_weight = adjustments['load_balance_weight']
            print(f'  MoE Adaptive: noise={adjustments.get("noise_std", "N/A"):.4f}, lb_weight={adjustments.get("load_balance_weight", "N/A"):.4f}, lb_adj={adjustments.get("lb_adjustment", "N/A")}')

        # 3. Step the scheduler at the end of the epoch
        scheduler.step()

        # 4. Log the current learning rate to Weights & Biases
        current_lr = optimizer.param_groups[0]['lr']
        # Log training metrics to wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'learning_rate': current_lr,
                'train/mse': mse_value,
                'train/rmse': rmse_value,
                'train/ci': ci,
                'train/r2': r2,
                'train/pearson': pearson_value,
                'train/spearman': spearman_value,
            }
            # Add MoE statistics
            if moe_stats:
                log_dict['moe/load_balance_loss'] = avg_load_balance_loss
                log_dict['moe/usage_entropy'] = moe_stats['usage_entropy']
                log_dict['moe/usage_std'] = moe_stats['usage_std']
                log_dict['moe/dominant_expert'] = moe_stats['dominant_expert']
                for i, rate in enumerate(moe_stats['expert_usage_rate']):
                    log_dict[f'moe/expert_{i}_usage'] = rate

            # Add adaptive parameters
            if hasattr(model.module, 'adaptive_update'):
                log_dict['moe/adaptive_noise_std'] = adjustments.get('noise_std', 0)
                log_dict['moe/adaptive_lb_weight'] = adjustments.get('load_balance_weight', 0)
            wandb.log(log_dict)

        # valid
        mse, rmse, ci, r2, pearson, spearman = test(model, valid_dataset_load, is_valid=True)
        print(f'Valid at fold-{fold_i}: mse-{mse}')

        # Log validation metrics to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'valid/mse': mse,
                'valid/rmse': rmse,
                'valid/r2': r2,
                'valid/pearson': pearson,
                'valid/spearman': spearman,
            })

        # Early stop
        if mse < best_valid_mse :
            patience = 0
            best_valid_mse = mse
            # save model
            torch.save(model.state_dict(), model_fromTrain)
            print(f'Update best_mse, Valid at fold-{fold_i} epoch-{epoch}: mse-{mse}, rmse-{rmse}, ci-{ci}, r2-{r2}, pearson-{pearson}, spearman-{spearman}')

            # Log best validation metrics to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'best_valid/mse': mse,
                    'best_valid/rmse': rmse,
                    'best_valid/r2': r2,
                    'best_valid/pearson': pearson,
                    'best_valid/spearman': spearman,
                })
        else:
            patience += 1
            if patience > hp.max_patience:
                print(f'Traing stop at epoch-{epoch}, model save at-{model_fromTrain}')
                break

    log_dir = f"./log/{timestamp}-{hp.dataset}-{hp.running_set}-fnet_learnable-fold{fold_i}.csv"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    with open(log_dir, "w+")as f:
        writer = csv.writer(f)
        writer.writerow(["mse", "rmse",  "ci", "r2", 'pearson', 'spearman'])
        for r in train_log:
            writer.writerow(r)
    print(f'Save log over at {log_dir}')

    # Test
    print(f"\n{'='*60}")
    print(f"Testing fold {fold_i} with best model...")
    print(f"{'='*60}")
    predModel = nn.DataParallel(LLMDTA_FNet_Learnable(hp, device))
    predModel.load_state_dict(torch.load(model_fromTrain))
    predModel = predModel.to(device)
    mse, rmse, ci, r2, pearson, spearman = test(predModel, test_dataset_load, is_valid=False)
    print(f'Test at fold-{fold_i}, mse: {mse}, rmse: {rmse}, ci: {ci}, r2: {r2}, pearson: {pearson}, spearman: {spearman}\n')

    # Log test metrics to wandb
    if use_wandb:
        wandb.log({
            'test/mse': mse,
            'test/rmse': rmse,
            'test/ci': ci,
            'test/r2': r2,
            'test/pearson': pearson,
            'test/spearman': spearman,
        })
        wandb.summary['final_test_mse'] = mse
        wandb.summary['final_test_rmse'] = rmse
        wandb.summary['final_test_ci'] = ci
        wandb.summary['final_test_r2'] = r2
        wandb.summary['final_test_pearson'] = pearson
        wandb.summary['final_test_spearman'] = spearman

    # Save test results for this fold
    fold_result_file = f'./log/Test-{hp.dataset}-{hp.running_set}-fnet_learnable-fold{fold_i}-{timestamp}.csv'
    fold_result = pd.DataFrame({
        'fold': [fold_i],
        'mse': [mse],
        'rmse': [rmse],
        'ci': [ci],
        'r2': [r2],
        'pearson': [pearson],
        'spearman': [spearman]
    })
    fold_result.to_csv(fold_result_file, index=False)
    print(f"Fold {fold_i} results saved to: {fold_result_file}")
    print(f"{'='*60}")
    print(f"Training fold {fold_i} completed successfully!")
    print(f"{'='*60}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("Weights & Biases run finished")
