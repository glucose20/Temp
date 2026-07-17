"""
Train LLMDTA_FNet (MoE + FourierCrossMixing thay the CrossAttention) tren mot may co GPU rieng (vd. H100).
File nay tu chua toan bo (khong phu thuoc notebook/Colab/Kaggle) -- du lieu (fold csv + pretrain
embeddings mol2vec/ESM2) phai co san local truoc, dung cau truc thu muc cua repo (xem hyperparameter.py).

Vi du chay tren H100:
    python code/train_fnet.py --dataset davis --running_set warm --fold 0 --seed 0 \\
        --epochs 100 --batch_size 128 --amp --output_dir ./results

    # Chay het 5 fold, 3 seed (dung cho paired t-test so voi baseline sau nay)
    for fold in 0 1 2 3 4; do
      for seed in 0 1 2; do
        python code/train_fnet.py --fold $fold --seed $seed --amp --output_dir ./results
      done
    done

H100 goi y: --amp (bf16 autocast, H100 ho tro native, khong can GradScaler) + --batch_size 128-256 tuy VRAM.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyperparameter import HyperParameter
from LLMDTA import LLMDTA
from MyDataset import CustomDataSet, my_collate_fn
from train import regression_scores, load_pickle, set_seed, test as run_test


# ----------------------------------------------------------------------------
# Kien truc: FourierCrossMixing thay the CrossAttention (phan con lai cua
# LLMDTA -- Encoder, MoE head -- giu nguyen khong doi so voi baseline).
# ----------------------------------------------------------------------------

class FourierCrossMixing(nn.Module):
    """Parameter-free FNet-style Fourier mixing, dung thay the CrossAttention."""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        seq_q = query.shape[1]
        combined = torch.cat([query, key_value], dim=1)
        mixed = torch.fft.fft(torch.fft.fft(combined, dim=-1), dim=-2).real
        mixed_query = mixed[:, :seq_q, :]
        mixed_query = self.dropout(mixed_query)
        return self.out_ln(mixed_query + query)


class LLMDTA_FNet(LLMDTA):
    """LLMDTA (MoE) voi CrossAttention duoc thay bang FourierCrossMixing."""

    def __init__(self, hp, device):
        super().__init__(hp, device)
        self.drug_cross_attn = FourierCrossMixing(self.hidden_dim, self.cross_attention_dropout)
        self.prot_cross_attn = FourierCrossMixing(self.hidden_dim, self.cross_attention_dropout)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def build_hp(args):
    hp = HyperParameter()
    hp.set_dataset(args.dataset)
    hp.running_set = args.running_set
    hp.Epoch = args.epochs
    hp.Batch_size = args.batch_size
    hp.max_patience = args.max_patience
    hp.Learning_rate = args.lr
    if args.use_esmc:
        hp.use_esmc = True
        hp.esmc_model = args.esmc_model
        dims = {'esmc_300m': 960, 'esmc_600m': 1152, 'esmc_6b': 2560}
        hp.protvec_dim = dims[args.esmc_model]
        hp.set_dataset(args.dataset)  # re-derive protvec_dir voi cau hinh ESM-C moi
    return hp


def check_required_files(hp, fold):
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    required = [hp.mol2vec_dir, hp.protvec_dir, hp.drugs_dir, hp.prots_dir]
    for split in ['train', 'valid', 'test']:
        required.append(os.path.join(dataset_root, f'fold_{fold}_{split}.csv'))
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            'Thieu cac file sau (kiem tra data_root/pretrain embeddings):\n'
            + '\n'.join(f'  - {m}' for m in missing)
        )
    return dataset_root


def build_loaders(hp, dataset_root, fold, device, num_workers=0):
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)

    train_set = CustomDataSet(pd.read_csv(os.path.join(dataset_root, f'fold_{fold}_train.csv')), hp)
    valid_set = CustomDataSet(pd.read_csv(os.path.join(dataset_root, f'fold_{fold}_valid.csv')), hp)
    test_set = CustomDataSet(pd.read_csv(os.path.join(dataset_root, f'fold_{fold}_test.csv')), hp)

    collate = lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict)
    train_loader = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True,
                               num_workers=num_workers, collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=False,
                               num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=False,
                              num_workers=num_workers, collate_fn=collate)
    return train_loader, valid_loader, test_loader, len(train_set), len(valid_set), len(test_set)


# ----------------------------------------------------------------------------
# Train / eval loop
# ----------------------------------------------------------------------------

def run_training(hp, train_loader, valid_loader, test_loader, device,
                  seed=0, amp=False, amp_dtype=torch.bfloat16, verbose=True):
    set_seed(seed)
    model = LLMDTA_FNet(hp, device).to(device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.Epoch, eta_min=1e-6)
    criterion = F.mse_loss

    # bf16 khong can loss scaling; chi bat GradScaler khi dung fp16.
    use_scaler = amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    history = {'epoch': [], 'train_mse': [], 'train_ci': [], 'valid_mse': [], 'valid_rmse': [], 'valid_r2': []}
    best_valid_mse = float('inf')
    best_state = None
    patience = 0

    t0 = time.time()
    for epoch in range(1, hp.Epoch + 1):
        model.reset_usage_stats()

        model.train()
        preds, labels = [], []
        for batch_data in train_loader:
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp and device.type == 'cuda'):
                predictions, gate_info = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask,
                                                return_gate_info=True)
                loss = criterion(predictions.squeeze(), affinity)
                lb_loss = model.compute_load_balance_loss(gate_info['gate_weights'])
                total_loss = loss + hp.load_balance_weight * lb_loss

            if use_scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            preds += predictions.detach().float().cpu().numpy().reshape(-1).tolist()
            labels += affinity.detach().float().cpu().numpy().reshape(-1).tolist()

        train_mse, _, train_ci, _, _, _ = regression_scores(np.array(labels), np.array(preds), is_valid=False)
        valid_mse, valid_rmse, _, valid_r2, _, _ = run_test(model, valid_loader, is_valid=True)
        scheduler.step()

        history['epoch'].append(epoch)
        history['train_mse'].append(train_mse)
        history['train_ci'].append(train_ci)
        history['valid_mse'].append(valid_mse)
        history['valid_rmse'].append(valid_rmse)
        history['valid_r2'].append(valid_r2)

        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if verbose:
            print(f'[FNet] epoch {epoch:3d}/{hp.Epoch}  train_mse={train_mse:.4f} train_ci={train_ci:.4f}  '
                  f'valid_mse={valid_mse:.4f}  best_valid_mse={best_valid_mse:.4f}  patience={patience}',
                  flush=True)

        if patience > hp.max_patience:
            print(f'[FNet] early stopping at epoch {epoch}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse, test_rmse, test_ci, test_r2, test_pearson, test_spearman = run_test(model, test_loader, is_valid=False)
    elapsed = time.time() - t0

    result = {
        'name': 'FNet Fourier Mixing (MoE)',
        'n_params': n_params,
        'epochs_ran': history['epoch'][-1],
        'elapsed_sec': round(elapsed, 1),
        'test_mse': test_mse, 'test_rmse': test_rmse, 'test_ci': test_ci,
        'test_r2': test_r2, 'test_pearson': test_pearson, 'test_spearman': test_spearman,
    }
    print(f'[FNet] TEST  mse={test_mse:.4f}  rmse={test_rmse:.4f}  ci={test_ci:.4f}  '
          f'r2={test_r2:.4f}  pearson={test_pearson:.4f}  ({n_params:,} params, {elapsed:.1f}s)')
    return model, history, result


def main():
    parser = argparse.ArgumentParser(description='Train LLMDTA_FNet (MoE + Fourier mixing) tren 1 fold')
    parser.add_argument('--dataset', default='davis', choices=['davis', 'kiba', 'metz'])
    parser.add_argument('--running_set', default='warm', choices=['warm', 'novel-drug', 'novel-prot', 'novel-pair'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', default='0', help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--amp', action='store_true', help='Bat mixed precision (khuyen nghi tren H100)')
    parser.add_argument('--amp_dtype', choices=['bf16', 'fp16'], default='bf16',
                         help='bf16 khuyen nghi tren H100 (khong can GradScaler)')
    parser.add_argument('--num_workers', type=int, default=0,
                         help='Giu 0 vi my_collate_fn dua tensor len CUDA ben trong collate_fn')
    parser.add_argument('--use_esmc', action='store_true')
    parser.add_argument('--esmc_model', default='esmc_600m', choices=['esmc_300m', 'esmc_600m', 'esmc_6b'])
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--save_weights', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))

    hp = build_hp(args)
    dataset_root = check_required_files(hp, args.fold)
    train_loader, valid_loader, test_loader, n_tr, n_va, n_te = build_loaders(
        hp, dataset_root, args.fold, device, num_workers=args.num_workers
    )
    print(f'Dataset: {hp.dataset}/{hp.running_set} fold{args.fold}  Train={n_tr} Valid={n_va} Test={n_te}')

    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    model, history, result = run_training(
        hp, train_loader, valid_loader, test_loader, device,
        seed=args.seed, amp=args.amp, amp_dtype=amp_dtype,
    )

    tag = f'fnet_{args.dataset}_{args.running_set}_fold{args.fold}_seed{args.seed}'
    os.makedirs(args.output_dir, exist_ok=True)

    result_full = {**result, 'model_key': 'fnet', 'dataset': hp.dataset, 'running_set': hp.running_set,
                   'fold': args.fold, 'seed': args.seed}
    with open(os.path.join(args.output_dir, f'result_{tag}.json'), 'w') as f:
        json.dump(result_full, f, indent=2)
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, f'history_{tag}.csv'), index=False)
    if args.save_weights:
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'weights_{tag}.pth'))

    print(f'Da luu ket qua: {args.output_dir}/result_{tag}.json, history_{tag}.csv'
          + (f', weights_{tag}.pth' if args.save_weights else ''))


if __name__ == '__main__':
    main()
