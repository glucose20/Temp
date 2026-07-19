"""
Component Ablation Study for LLMDTA_FNet (model co FFT)
=========================================================
Danh gia dong gop cua TUNG THANH PHAN kien truc trong model LLMDTA_FNet
(model dung FourierMixing/FFT thay the CrossAttention). Khac voi ablation
chi xoay quanh rieng FFT, script nay ablate toan bo pipeline:

    Input (Mol2Vec/ESM pretrained)
        |
    1. Encoder Block (1D-CNN voi skip-connection)
        |
    2. FourierMixing (FFT 2D) -- co che "mixing" giua drug <-> protein
        |
    3. Self-Attention Pooling
        |
    4. Residual Feature Fusion (h_pre + h_post)
        |
    5. Mixture of Experts (MoE) Predictor

Cac bien the (variants):
    full               Full FNet Model - FFT lam co che mixing (MODEL GOC, la baseline)
    wo_encoder         Thay 1D-CNN Encoder bang Linear Encoder (giu FFT mixing)
    wo_fft_mixing      Bo han co che mixing (Identity passthrough) de co lap
                       dong gop rieng cua FFT trong toan bo pipeline
    attn_mixing        Thay FourierMixing bang Multi-head CrossAttention
                       (so sanh FFT voi co che mixing truyen thong)
    wo_self_attn_pool  Thay Self-Attention Pooling bang Mean Pooling (giu FFT)
    wo_residual        Bo residual fusion, chi dung h_post (giu FFT)
    wo_moe             Thay MoE bang mot predictor don (giu FFT)
    linear_only        Chi dung pretrained feature + Linear predictor
                       (khong Encoder/Mixing/MoE, kieu AI-Bind)

Usage:
    python ablation_fft.py --dataset davis --running_set warm --fold 0
    python ablation_fft.py --dataset davis --running_set warm --all_folds --epochs 100
    python ablation_fft.py --variants full,wo_fft_mixing,attn_mixing

Resume sau khi bi huy giua chung (vi du Kaggle timeout/idle-disconnect):
    Sau MOI bien the hoan tat, ket qua duoc ghi de vao
    <output_dir>/ablation_fnet_<dataset>_<running_set>_fold<F>_partial.csv
    (khong mang timestamp -> ton tai qua nhieu lan chay). Chay lai dung lenh cu,
    them --resume, script se bo qua cac bien the/fold da xong va chi chay tiep
    phan con thieu:
    python ablation_fft.py --dataset davis --running_set warm --all_folds \\
        --epochs 100 --patience 20 --output_dir ./ablation_fft_results --resume
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import math

from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn

from sklearn.metrics import r2_score
from scipy import stats
from math import sqrt
from tqdm import tqdm


# ============================================================
# COMPONENT MODULES (self-contained, hidden_dim co dinh = 128)
# ============================================================

class GatingNetwork(nn.Module):
    """Gating network chon expert theo dac trung dau vao, co Top-K selection."""
    def __init__(self, input_dim, num_experts, top_k=2, noise_std=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.initial_noise_std = noise_std
        self.noise_std = noise_std
        self.layer = nn.Linear(input_dim, num_experts)

    def set_noise_std(self, noise_std):
        self.noise_std = noise_std

    def forward(self, x, return_all_weights=False):
        logits = self.layer(x)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        raw_weights = F.softmax(logits, dim=1)

        if self.top_k < self.num_experts:
            top_k_weights, top_k_indices = torch.topk(raw_weights, self.top_k, dim=1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
            gate_weights = torch.zeros_like(raw_weights)
            gate_weights.scatter_(1, top_k_indices, top_k_weights)
        else:
            gate_weights = raw_weights

        if return_all_weights:
            return gate_weights, raw_weights
        return gate_weights


class SelfAttentionPooling(nn.Module):
    """Self-attention pooling: tong hop sequence thanh 1 vector."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled


class CrossAttention(nn.Module):
    """Multi-head cross attention - chi dung cho bien the 'attn_mixing' (so sanh voi FFT)."""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        batch_size = query.shape[0]

        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)

        return self.out_ln(attn_output + query)


class FourierMixing(nn.Module):
    """Parameter-free FNet-style 2D Fourier mixing - CO CHE FFT CUA MODEL GOC.
    fft theo hidden-dim roi fft theo sequence-dim, lay phan thuc (real)."""
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


class NoMixing_Identity(nn.Module):
    """Khong mixing (khong FFT, khong attention) - passthrough thuan tuy.
    Dung de co lap dong gop rieng cua FFT (bien the 'wo_fft_mixing')."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        return self.out_ln(self.dropout(query) + query)


# ============================================================
# ENCODER VARIANTS
# ============================================================

class Encoder(nn.Module):
    """Full Encoder: 1D-CNN voi skip-connection."""
    def __init__(self, max_len, input_dim, device, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = 7
        self.do = nn.Dropout(0.1)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([0.5])))

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.hidden_dim, self.hidden_dim * 2, self.kernel_size, padding=(self.kernel_size - 1) // 2),
            nn.Conv1d(self.hidden_dim, self.hidden_dim * 2, self.kernel_size, padding=(self.kernel_size - 1) // 2),
            nn.Conv1d(self.hidden_dim, self.hidden_dim * 2, self.kernel_size, padding=(self.kernel_size - 1) // 2)
        ])
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        feat_map = self.input_norm(feat_map)
        h_map = self.fc(feat_map)
        h_map = h_map.permute(0, 2, 1)

        for conv in self.convs:
            conved = conv(self.do(h_map))
            conved = F.glu(conved, dim=1)
            conved = (conved + h_map) * self.scale
            h_map = conved

        pool_map = self.max_pool(h_map).squeeze(-1)
        h_map = h_map.permute(0, 2, 1)
        h_map = self.ln(h_map)
        return h_map, pool_map


class LinearEncoder(nn.Module):
    """Linear Encoder don gian (w/o CNN) - dung cho ablation 'wo_encoder'."""
    def __init__(self, max_len, input_dim, device, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        feat_map = self.input_norm(feat_map)
        h_map = self.fc(feat_map)
        h_map = self.ln(h_map)

        h_map_t = h_map.permute(0, 2, 1)
        pool_map = self.max_pool(h_map_t).squeeze(-1)

        return h_map, pool_map


# ============================================================
# MODEL VARIANTS
# ============================================================

class LLMDTA_FNet_Full(nn.Module):
    """[full] Full FNet Model - FFT (FourierMixing) lam co che mixing. Day la
    model MUC TIEU ('model co FFT'), duoc dung lam baseline cho toan bo ablation."""
    def __init__(self, hp, device):
        super().__init__()
        self.variant_name = "Full FNet Model (with FFT)"

        self.hidden_dim = 128
        self.num_experts = getattr(hp, 'num_experts', 4)
        self.top_k = getattr(hp, 'top_k', 2)
        self.moe_noise_std = getattr(hp, 'moe_noise_std', 0.1)
        self.load_balance_weight = getattr(hp, 'load_balance_weight', 0.01)
        cross_dropout = getattr(hp, 'cross_attention_dropout', 0.1)

        self.drug_embed = Encoder(hp.drug_max_len, hp.mol2vec_dim, device)
        self.prot_embed = Encoder(hp.prot_max_len, hp.protvec_dim, device)

        self.drug_cross_attn = FourierMixing(self.hidden_dim, cross_dropout)
        self.prot_cross_attn = FourierMixing(self.hidden_dim, cross_dropout)

        self.drug_attn_pool = SelfAttentionPooling(self.hidden_dim)
        self.prot_attn_pool = SelfAttentionPooling(self.hidden_dim)

        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128 * 2, 1024), nn.ELU())
        self.linear_post = nn.Sequential(nn.Linear(128 * 2, 1024), nn.ELU())

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(1024, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, 1))
            for _ in range(self.num_experts)
        ])
        self.gating = GatingNetwork(1024, self.num_experts, self.top_k, self.moe_noise_std)

        self.register_buffer('expert_usage_count', torch.zeros(self.num_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, return_gate_info=False):
        drug_embed, drug_pool = self.drug_embed(drug_mat)
        prot_embed, prot_pool = self.prot_embed(prot_mat)

        new_drug_embed = self.drug_cross_attn(drug_embed, prot_embed)
        new_prot_embed = self.prot_cross_attn(prot_embed, drug_embed)

        drug_cross_pool = self.drug_attn_pool(new_drug_embed)
        prot_cross_pool = self.prot_attn_pool(new_prot_embed)

        h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))
        h_post = self.linear_post(torch.cat([drug_cross_pool, prot_cross_pool], dim=-1))
        h_combined = h_pre + h_post

        gate_weights, raw_weights = self.gating(h_combined, return_all_weights=True)

        if self.training:
            with torch.no_grad():
                self.expert_usage_count += (gate_weights > 0).float().sum(dim=0)
                self.total_samples += gate_weights.size(0)

        expert_outputs = torch.stack([exp(h_combined) for exp in self.experts], dim=1)
        pred = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        if return_gate_info:
            return pred, {'gate_weights': gate_weights, 'raw_weights': raw_weights}
        return pred

    def compute_load_balance_loss(self, gate_weights):
        return self.num_experts * torch.sum(gate_weights.mean(dim=0) ** 2)

    def reset_usage_stats(self):
        self.expert_usage_count.zero_()
        self.total_samples.zero_()

    def get_expert_usage_stats(self):
        if self.total_samples == 0:
            return None
        usage_rate = self.expert_usage_count / self.total_samples
        usage_rate = usage_rate / (usage_rate.sum() + 1e-8)
        return {
            'expert_usage_rate': usage_rate.cpu().numpy(),
            'usage_entropy': -torch.sum(usage_rate * torch.log(usage_rate + 1e-8)).item(),
            'dominant_expert': torch.argmax(usage_rate).item(),
            'usage_std': usage_rate.std().item(),
        }


class LLMDTA_FNet_woEncoder(LLMDTA_FNet_Full):
    """[wo_encoder] Thay 1D-CNN bang Linear Encoder, giu nguyen FFT mixing."""
    def __init__(self, hp, device):
        super().__init__(hp, device)
        self.variant_name = "w/o Encoder (Linear instead of CNN, keep FFT)"
        self.drug_embed = LinearEncoder(hp.drug_max_len, hp.mol2vec_dim, device)
        self.prot_embed = LinearEncoder(hp.prot_max_len, hp.protvec_dim, device)


class LLMDTA_FNet_woFFTMixing(LLMDTA_FNet_Full):
    """[wo_fft_mixing] Bo han co che mixing (Identity) de co lap dong gop
    rieng cua FFT trong toan bo pipeline."""
    def __init__(self, hp, device):
        super().__init__(hp, device)
        self.variant_name = "w/o FFT Mixing (Identity passthrough)"
        cross_dropout = getattr(hp, 'cross_attention_dropout', 0.1)
        self.drug_cross_attn = NoMixing_Identity(self.hidden_dim, cross_dropout)
        self.prot_cross_attn = NoMixing_Identity(self.hidden_dim, cross_dropout)


class LLMDTA_FNet_AttnMixing(LLMDTA_FNet_Full):
    """[attn_mixing] Thay FourierMixing bang Multi-head CrossAttention
    (so sanh truc tiep FFT voi co che mixing truyen thong)."""
    def __init__(self, hp, device):
        super().__init__(hp, device)
        self.variant_name = "FFT -> CrossAttention (mixing strategy swap)"
        cross_dropout = getattr(hp, 'cross_attention_dropout', 0.1)
        self.drug_cross_attn = CrossAttention(self.hidden_dim, num_heads=8, dropout=cross_dropout)
        self.prot_cross_attn = CrossAttention(self.hidden_dim, num_heads=8, dropout=cross_dropout)


class LLMDTA_FNet_woSelfAttnPool(LLMDTA_FNet_Full):
    """[wo_self_attn_pool] Thay Self-Attention Pooling bang Mean Pooling, giu FFT."""
    def __init__(self, hp, device):
        super().__init__(hp, device)
        self.variant_name = "w/o Self-Attention Pooling (Mean Pool, keep FFT)"

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, return_gate_info=False):
        drug_embed, drug_pool = self.drug_embed(drug_mat)
        prot_embed, prot_pool = self.prot_embed(prot_mat)

        new_drug_embed = self.drug_cross_attn(drug_embed, prot_embed)
        new_prot_embed = self.prot_cross_attn(prot_embed, drug_embed)

        # MEAN pooling thay vi self-attention pooling
        drug_cross_pool = new_drug_embed.mean(dim=1)
        prot_cross_pool = new_prot_embed.mean(dim=1)

        h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))
        h_post = self.linear_post(torch.cat([drug_cross_pool, prot_cross_pool], dim=-1))
        h_combined = h_pre + h_post

        gate_weights, raw_weights = self.gating(h_combined, return_all_weights=True)

        if self.training:
            with torch.no_grad():
                self.expert_usage_count += (gate_weights > 0).float().sum(dim=0)
                self.total_samples += gate_weights.size(0)

        expert_outputs = torch.stack([exp(h_combined) for exp in self.experts], dim=1)
        pred = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        if return_gate_info:
            return pred, {'gate_weights': gate_weights, 'raw_weights': raw_weights}
        return pred


class LLMDTA_FNet_woResidual(LLMDTA_FNet_Full):
    """[wo_residual] Bo residual fusion, chi dung h_post (giu FFT)."""
    def __init__(self, hp, device):
        super().__init__(hp, device)
        self.variant_name = "w/o Residual Fusion (only h_post, keep FFT)"
        # linear_pre / bn khong dung nua nhung giu lai de state_dict tuong thich khong bat buoc

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, return_gate_info=False):
        drug_embed, drug_pool = self.drug_embed(drug_mat)
        prot_embed, prot_pool = self.prot_embed(prot_mat)

        new_drug_embed = self.drug_cross_attn(drug_embed, prot_embed)
        new_prot_embed = self.prot_cross_attn(prot_embed, drug_embed)

        drug_cross_pool = self.drug_attn_pool(new_drug_embed)
        prot_cross_pool = self.prot_attn_pool(new_prot_embed)

        # KHONG residual - chi dung dac trung sau FFT mixing
        h_combined = self.bn(self.linear_post(torch.cat([drug_cross_pool, prot_cross_pool], dim=-1)))

        gate_weights, raw_weights = self.gating(h_combined, return_all_weights=True)

        if self.training:
            with torch.no_grad():
                self.expert_usage_count += (gate_weights > 0).float().sum(dim=0)
                self.total_samples += gate_weights.size(0)

        expert_outputs = torch.stack([exp(h_combined) for exp in self.experts], dim=1)
        pred = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        if return_gate_info:
            return pred, {'gate_weights': gate_weights, 'raw_weights': raw_weights}
        return pred


class LLMDTA_FNet_woMoE(nn.Module):
    """[wo_moe] Thay MoE bang mot predictor don, giu nguyen FFT mixing."""
    def __init__(self, hp, device):
        super().__init__()
        self.variant_name = "w/o MoE (Single Predictor, keep FFT)"

        self.hidden_dim = 128
        cross_dropout = getattr(hp, 'cross_attention_dropout', 0.1)

        self.drug_embed = Encoder(hp.drug_max_len, hp.mol2vec_dim, device)
        self.prot_embed = Encoder(hp.prot_max_len, hp.protvec_dim, device)

        self.drug_cross_attn = FourierMixing(self.hidden_dim, cross_dropout)
        self.prot_cross_attn = FourierMixing(self.hidden_dim, cross_dropout)

        self.drug_attn_pool = SelfAttentionPooling(self.hidden_dim)
        self.prot_attn_pool = SelfAttentionPooling(self.hidden_dim)

        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128 * 2, 1024), nn.ELU())
        self.linear_post = nn.Sequential(nn.Linear(128 * 2, 1024), nn.ELU())

        self.predictor = nn.Sequential(
            nn.Linear(1024, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, 1)
        )

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, return_gate_info=False):
        drug_embed, drug_pool = self.drug_embed(drug_mat)
        prot_embed, prot_pool = self.prot_embed(prot_mat)

        new_drug_embed = self.drug_cross_attn(drug_embed, prot_embed)
        new_prot_embed = self.prot_cross_attn(prot_embed, drug_embed)

        drug_cross_pool = self.drug_attn_pool(new_drug_embed)
        prot_cross_pool = self.prot_attn_pool(new_prot_embed)

        h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))
        h_post = self.linear_post(torch.cat([drug_cross_pool, prot_cross_pool], dim=-1))
        h_combined = h_pre + h_post

        pred = self.predictor(h_combined)

        if return_gate_info:
            batch_size = pred.size(0)
            return pred, {
                'gate_weights': torch.ones(batch_size, 1, device=pred.device),
                'raw_weights': torch.ones(batch_size, 1, device=pred.device)
            }
        return pred

    def compute_load_balance_loss(self, gate_weights):
        return torch.tensor(0.0, device=gate_weights.device)

    def reset_usage_stats(self):
        pass

    def get_expert_usage_stats(self):
        return {'usage_entropy': 0.0}


class LLMDTA_FNet_Linear(nn.Module):
    """[linear_only] Chi pretrained feature + Linear predictor (AI-Bind style).
    Khong Encoder/Mixing/MoE - baseline yeu nhat de doi chieu."""
    def __init__(self, hp, device):
        super().__init__()
        self.variant_name = "Linear-only (AI-Bind style, no FFT/encoder/MoE)"

        self.drug_proj = nn.Linear(hp.mol2vec_dim, 256)
        self.prot_proj = nn.Linear(hp.protvec_dim, 256)

        self.predictor = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, return_gate_info=False):
        drug_feat = self.drug_proj(drug_mat)
        prot_feat = self.prot_proj(prot_mat)

        drug_pool = drug_feat.mean(dim=1)
        prot_pool = prot_feat.mean(dim=1)

        combined = torch.cat([drug_pool, prot_pool], dim=-1)
        pred = self.predictor(combined)

        if return_gate_info:
            batch_size = pred.size(0)
            return pred, {
                'gate_weights': torch.ones(batch_size, 1, device=pred.device),
                'raw_weights': torch.ones(batch_size, 1, device=pred.device)
            }
        return pred

    def compute_load_balance_loss(self, gate_weights):
        return torch.tensor(0.0, device=gate_weights.device)

    def reset_usage_stats(self):
        pass

    def get_expert_usage_stats(self):
        return {'usage_entropy': 0.0}


# ============================================================
# MODEL FACTORY
# ============================================================

VARIANT_REGISTRY = {
    'full': (LLMDTA_FNet_Full, 'Full FNet Model (with FFT) - baseline'),
    'wo_encoder': (LLMDTA_FNet_woEncoder, 'w/o Encoder (Linear instead of CNN, keep FFT)'),
    'wo_fft_mixing': (LLMDTA_FNet_woFFTMixing, 'w/o FFT Mixing (Identity instead of FFT)'),
    'attn_mixing': (LLMDTA_FNet_AttnMixing, 'FFT -> CrossAttention (mixing strategy swap)'),
    'wo_self_attn_pool': (LLMDTA_FNet_woSelfAttnPool, 'w/o Self-Attention Pooling (Mean Pool, keep FFT)'),
    'wo_residual': (LLMDTA_FNet_woResidual, 'w/o Residual Fusion (only h_post, keep FFT)'),
    'wo_moe': (LLMDTA_FNet_woMoE, 'w/o MoE (Single Predictor, keep FFT)'),
    'linear_only': (LLMDTA_FNet_Linear, 'Linear-only (AI-Bind style, no FFT/encoder/MoE)'),
}


def get_ablation_model(variant_name, hp, device):
    if variant_name not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant: {variant_name}. Choose from {list(VARIANT_REGISTRY.keys())}")
    model_cls, _ = VARIANT_REGISTRY[variant_name]
    return model_cls(hp, device)


def get_all_variants():
    return [(name, desc) for name, (_, desc) in VARIANT_REGISTRY.items()]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)


def cindex_score(y, p):
    sum_m = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if y[i] > y[j]:
                pair += 1
                sum_m += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    return sum_m / pair if pair != 0 else 0


def regression_scores(label, pred, compute_ci=True):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    mse = ((label - pred) ** 2).mean(axis=0)
    rmse = sqrt(mse)
    ci = cindex_score(label, pred) if compute_ci else -1
    r2 = r2_score(label, pred)
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return {
        'mse': round(mse, 6),
        'rmse': round(rmse, 6),
        'ci': round(ci, 6),
        'r2': round(r2, 6),
        'pearson': round(pearson, 6),
        'spearman': round(spearman, 6),
    }


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_one_epoch(model, train_loader, optimizer, criterion, hp, max_grad_norm=1.0):
    """Train model for one epoch with gradient clipping."""
    model.train()
    preds, labels = [], []

    for batch_data in tqdm(train_loader, desc='Training', leave=False):
        mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data

        if hasattr(model, 'module'):
            predictions, gate_info = model.module.forward(
                mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True
            )
            lb_loss = model.module.compute_load_balance_loss(gate_info['gate_weights'])
        else:
            predictions, gate_info = model(
                mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=True
            )
            lb_loss = model.compute_load_balance_loss(gate_info['gate_weights'])

        preds.extend(predictions.cpu().detach().numpy().reshape(-1).tolist())
        labels.extend(affinity.cpu().detach().numpy().reshape(-1).tolist())

        loss = criterion(predictions.squeeze(), affinity)
        if lb_loss is not None and lb_loss.item() > 0:
            loss = loss + getattr(hp, 'load_balance_weight', 0.01) * lb_loss

        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

    return regression_scores(np.array(labels), np.array(preds), compute_ci=False)


def evaluate(model, dataloader, compute_ci=True):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='Evaluating', leave=False):
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, affinity = batch_data

            if hasattr(model, 'module'):
                pred = model.module.forward(
                    mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=False
                )
            else:
                pred = model.forward(
                    mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, return_gate_info=False
                )

            preds.extend(pred.cpu().numpy().reshape(-1).tolist())
            labels.extend(affinity.cpu().numpy().reshape(-1).tolist())

    return regression_scores(np.array(labels), np.array(preds), compute_ci=compute_ci)


def train_and_evaluate(variant_name, hp, device, train_loader, valid_loader, test_loader,
                        epochs=100, patience=20, verbose=True):
    """Full training with early stopping for one component-ablation variant."""

    model = get_ablation_model(variant_name, hp, device)
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp.Learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    criterion = F.mse_loss

    best_valid_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_test_metrics = None

    for epoch in range(1, epochs + 1):
        if hasattr(model.module, 'reset_usage_stats'):
            model.module.reset_usage_stats()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, hp)
        valid_metrics = evaluate(model, valid_loader, compute_ci=False)

        scheduler.step()

        if valid_metrics['mse'] < best_valid_mse:
            best_valid_mse = valid_metrics['mse']
            best_epoch = epoch
            patience_counter = 0

            test_metrics = evaluate(model, test_loader, compute_ci=True)
            best_test_metrics = test_metrics.copy()
            best_test_metrics['best_epoch'] = epoch
            best_test_metrics['best_valid_mse'] = best_valid_mse

            if hasattr(model.module, 'get_expert_usage_stats'):
                moe_stats = model.module.get_expert_usage_stats()
                if moe_stats:
                    best_test_metrics['entropy'] = moe_stats.get('usage_entropy', 0.0)
        else:
            patience_counter += 1

        if verbose and epoch % 20 == 0:
            print(f"    Epoch {epoch}: Valid MSE={valid_metrics['mse']:.4f}, Best={best_valid_mse:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopped at epoch {epoch}")
            break

    del model, optimizer
    torch.cuda.empty_cache()

    return best_test_metrics


# ============================================================
# ABLATION STUDY RUNNER
# ============================================================

def _load_partial_results(partial_csv_path):
    """Doc ket qua da hoan tat (status='success') tu 1 lan chay truoc bi ngat giua chung."""
    if not partial_csv_path or not os.path.exists(partial_csv_path):
        return {}
    try:
        prev_df = pd.read_csv(partial_csv_path)
    except Exception as e:
        print(f"[resume] Khong doc duoc {partial_csv_path} ({e}), bo qua resume.")
        return {}

    done = {}
    for _, row in prev_df.iterrows():
        if row.get('status') != 'success':
            continue
        # ep numpy scalar -> python native de json.dump khong loi sau nay
        record = {k: (v.item() if hasattr(v, 'item') else v) for k, v in row.to_dict().items()}
        done[record['variant']] = record
    return done


def run_component_ablation(hp, device, train_loader, valid_loader, test_loader,
                            variant_names, epochs=100, patience=20, verbose=True,
                            partial_csv_path=None, resume=False):
    """Run component ablation study on the FFT-based (FNet) model.

    Neu resume=True va partial_csv_path da co san ket qua tu lan chay truoc (vi du
    bi Kaggle huy giua chung), cac bien the da 'success' se duoc BO QUA (dung lai
    ket qua cu) thay vi train lai tu dau. Sau MOI bien the (du thanh cong hay loi),
    ket qua hien co duoc ghi de vao partial_csv_path -- neu session bi cat bat ky
    luc nao, lan chay tiep theo (voi --resume) se tiep tuc tu bien the con thieu.
    """

    print("\n" + "=" * 70)
    print("LLMDTA_FNet COMPONENT ABLATION STUDY (model co FFT)")
    print("=" * 70)
    print("Danh gia dong gop cua tung thanh phan kien truc...")
    print("=" * 70)

    results = []
    baseline_metrics = None  # 'full' la baseline (model FFT day du)

    done_variants = _load_partial_results(partial_csv_path) if resume else {}
    if done_variants:
        print(f"[resume] Tim thay {len(done_variants)} bien the da hoan tat truoc do: "
              f"{list(done_variants.keys())} -- se bo qua, chi chay phan con thieu.")

    for i, variant_name in enumerate(variant_names):
        description = VARIANT_REGISTRY[variant_name][1]

        if variant_name in done_variants:
            print(f"\n[{i + 1}/{len(variant_names)}] {variant_name}: SKIP (da co ket qua tu lan chay truoc)")
            result = done_variants[variant_name]
            results.append(result)
            if variant_name == 'full':
                baseline_metrics = {k: result[k] for k in
                                     ['mse', 'rmse', 'ci', 'r2', 'pearson', 'spearman']}
            if partial_csv_path:
                pd.DataFrame(results).to_csv(partial_csv_path, index=False)
            continue

        print(f"\n[{i + 1}/{len(variant_names)}] {variant_name}: {description}")
        print("-" * 50)

        set_seed(42)

        try:
            start_time = time.time()
            metrics = train_and_evaluate(
                variant_name, hp, device,
                train_loader, valid_loader, test_loader,
                epochs=epochs, patience=patience, verbose=verbose
            )
            elapsed = time.time() - start_time

            if variant_name == 'full':
                baseline_metrics = metrics.copy()

            result = {
                'variant': variant_name,
                'description': description,
                **metrics,
                'training_time': round(elapsed, 2),
                'status': 'success'
            }

            if baseline_metrics and variant_name != 'full':
                result['mse_diff'] = round(metrics['mse'] - baseline_metrics['mse'], 6)
                result['ci_diff'] = round(metrics['ci'] - baseline_metrics['ci'], 6)
                result['mse_diff_pct'] = round((metrics['mse'] - baseline_metrics['mse']) / baseline_metrics['mse'] * 100, 2)
                result['ci_diff_pct'] = round((metrics['ci'] - baseline_metrics['ci']) / baseline_metrics['ci'] * 100, 2)

            results.append(result)

            print(f"  CI={metrics['ci']:.4f}, MSE={metrics['mse']:.4f}, "
                  f"R2={metrics['r2']:.4f}, Epoch={metrics['best_epoch']}")

            if baseline_metrics and variant_name != 'full':
                print(f"  vs full (FFT): CI {result['ci_diff']:+.4f} ({result['ci_diff_pct']:+.2f}%), "
                      f"MSE {result['mse_diff']:+.4f} ({result['mse_diff_pct']:+.2f}%)")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                'variant': variant_name,
                'description': description,
                'status': 'failed',
                'error': str(e)
            })
            if partial_csv_path:
                pd.DataFrame(results).to_csv(partial_csv_path, index=False)
            continue

        if partial_csv_path:
            pd.DataFrame(results).to_csv(partial_csv_path, index=False)
            print(f"  [checkpoint] Da luu tien do vao {partial_csv_path}")

    return results, baseline_metrics


def print_ablation_summary(results, baseline_metrics):
    """Print formatted component ablation study summary."""

    print("\n" + "=" * 80)
    print("COMPONENT ABLATION STUDY RESULTS SUMMARY")
    print("=" * 80)

    successful_results = [r for r in results if r.get('status', 'success') == 'success']

    if not successful_results:
        print("No successful experiments to report.")
        return

    sorted_results = sorted(
        [r for r in successful_results if r['variant'] != 'full'],
        key=lambda x: x.get('ci_diff', 0)
    )

    print(f"\n{'Variant':<20} {'Description':<50} {'CI':>8} {'d CI':>10} {'MSE':>8} {'d MSE':>10}")
    print("-" * 110)

    baseline = next((r for r in successful_results if r['variant'] == 'full'), None)
    if baseline:
        print(f"{'full':<20} {'Full FNet Model (baseline, with FFT)':<50} {baseline['ci']:>8.4f} {'---':>10} "
              f"{baseline['mse']:>8.4f} {'---':>10}")
        print("-" * 110)

    for r in sorted_results:
        ci_diff_str = f"{r.get('ci_diff', 0):+.4f}" if 'ci_diff' in r else "N/A"
        mse_diff_str = f"{r.get('mse_diff', 0):+.4f}" if 'mse_diff' in r else "N/A"
        print(f"{r['variant']:<20} {r['description']:<50} {r['ci']:>8.4f} {ci_diff_str:>10} "
              f"{r['mse']:>8.4f} {mse_diff_str:>10}")

    print("\n" + "=" * 80)
    print("COMPONENT IMPORTANCE RANKING (theo muc CI giam khi bo/thay thanh phan)")
    print("=" * 80)

    importance = []
    for r in sorted_results:
        if 'ci_diff' in r:
            importance.append({
                'variant': r['variant'],
                'ci_drop': -r['ci_diff'],
                'mse_increase': r['mse_diff']
            })

    importance.sort(key=lambda x: x['ci_drop'], reverse=True)

    for i, item in enumerate(importance):
        bar_len = int(max(0, item['ci_drop'] * 500))
        bar = "#" * min(bar_len, 30)
        print(f"{i + 1}. {item['variant']:<25} CI drop: {item['ci_drop']:+.4f} {bar}")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("-" * 80)

    if importance:
        most_important = importance[0]
        least_important = importance[-1]
        print(f"- Thanh phan quan trong nhat: {most_important['variant']}")
        print(f"  -> Bo/thay no lam CI giam {most_important['ci_drop']:.4f}")
        print(f"- Thanh phan it quan trong nhat: {least_important['variant']}")
        print(f"  -> Bo/thay no chi lam CI thay doi {least_important['ci_drop']:.4f}")

        fft_row = next((r for r in sorted_results if r['variant'] == 'wo_fft_mixing'), None)
        if fft_row:
            print(f"- Dong gop rieng cua FFT (full vs wo_fft_mixing): d CI = {fft_row.get('ci_diff', 0):+.4f}")

        attn_row = next((r for r in sorted_results if r['variant'] == 'attn_mixing'), None)
        if attn_row:
            verdict = "FFT tot hon CrossAttention" if attn_row.get('ci_diff', 0) < 0 else "CrossAttention tot hon FFT"
            print(f"- So sanh co che mixing (FFT vs CrossAttention): {verdict} (d CI = {attn_row.get('ci_diff', 0):+.4f})")

    print("=" * 80)


def save_ablation_results(results, output_path, metadata=None):
    """Save results to CSV and JSON."""
    df = pd.DataFrame(results)
    df.to_csv(output_path + '.csv', index=False)
    print(f"\nResults saved to: {output_path}.csv")

    output_data = {'metadata': metadata or {}, 'results': results}
    with open(output_path + '.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_path}.json")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LLMDTA_FNet Component Ablation Study (model co FFT)')

    parser.add_argument('--dataset', type=str, default='davis',
                         choices=['davis', 'kiba', 'metz'])
    parser.add_argument('--running_set', type=str, default='warm',
                         choices=['warm', 'novel-drug', 'novel-prot', 'novel-pair'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--all_folds', action='store_true',
                         help='Run ablation on all 5 folds and average')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--num_experts', type=int, default=4,
                         help='Number of experts in MoE (default: 4)')
    parser.add_argument('--top_k', type=int, default=2,
                         help='Number of experts to select per sample (default: 2)')
    parser.add_argument('--moe_noise_std', type=float, default=0.1,
                         help='Noise std for MoE exploration (default: 0.1, 0 to disable)')
    parser.add_argument('--load_balance_weight', type=float, default=0.01,
                         help='Weight for load balancing loss (default: 0.01, 0 to disable)')

    parser.add_argument('--variants', type=str, default=None,
                         help='Comma-separated subset of variants to run '
                              f'(default: all of {list(VARIANT_REGISTRY.keys())})')

    parser.add_argument('--output_dir', type=str, default='./ablation_fft_results')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--resume', action='store_true',
                         help='Bo qua cac bien the/fold da hoan tat (doc tu file '
                              '*_partial.csv trong --output_dir) thay vi chay lai tu dau. '
                              'Dung khi phien chay truoc bi Kaggle/timeout huy giua chung.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.variants is not None:
        variant_names = [v.strip() for v in args.variants.split(',') if v.strip()]
        unknown = [v for v in variant_names if v not in VARIANT_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown variant(s): {unknown}. Choose from {list(VARIANT_REGISTRY.keys())}")
    else:
        variant_names = list(VARIANT_REGISTRY.keys())

    hp = HyperParameter()
    hp.set_dataset(args.dataset)
    hp.running_set = args.running_set
    hp.Batch_size = args.batch_size
    hp.Learning_rate = args.lr
    hp.cuda = args.cuda

    hp.num_experts = args.num_experts
    hp.top_k = args.top_k
    hp.moe_noise_std = args.moe_noise_std
    hp.load_balance_weight = args.load_balance_weight

    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"LLMDTA_FNet COMPONENT ABLATION STUDY")
    print(f"{'=' * 70}")
    print(f"Dataset: {args.dataset}-{args.running_set}")
    print(f"Folds: {'All (0-4)' if args.all_folds else args.fold}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}")
    print(f"Learning Rate: {args.lr}")
    print(f"Variants: {variant_names}")
    print(f"{'=' * 70}")

    print("\nLoading data...")
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)

    folds = range(5) if args.all_folds else [args.fold]
    all_fold_results = []

    for fold in folds:
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold}")
        print(f"{'=' * 70}")

        # File checkpoint KHONG mang timestamp -> ton tai xuyen suot nhieu lan chay,
        # de --resume co the tim lai duoc sau khi session bi Kaggle huy giua chung.
        partial_path = os.path.join(
            args.output_dir, f'ablation_fnet_{args.dataset}_{args.running_set}_fold{fold}_partial.csv')

        done_variants = _load_partial_results(partial_path) if args.resume else {}
        fold_fully_done = all(v in done_variants for v in variant_names)

        if args.resume and fold_fully_done:
            print(f"[resume] Fold {fold} da hoan tat du {len(variant_names)} bien the tu lan chay truoc "
                  f"({partial_path}) -- bo qua load data/training, dung lai ket qua cu.")
            results = [done_variants[v] for v in variant_names]
            baseline = next((r for r in results if r['variant'] == 'full'), None)
        else:
            dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)

            train_set = CustomDataSet(pd.read_csv(os.path.join(dataset_root, f'fold_{fold}_train.csv')), hp)
            valid_set = CustomDataSet(pd.read_csv(os.path.join(dataset_root, f'fold_{fold}_valid.csv')), hp)
            test_set = CustomDataSet(pd.read_csv(os.path.join(dataset_root, f'fold_{fold}_test.csv')), hp)

            def collate_fn(batch_data):
                return my_collate_fn(batch_data, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict)

            train_loader = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True,
                                       drop_last=True, num_workers=0, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False,
                                       drop_last=True, num_workers=0, collate_fn=collate_fn)
            test_loader = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False,
                                      drop_last=True, num_workers=0, collate_fn=collate_fn)

            print(f"Data: {len(train_set)} train, {len(valid_set)} valid, {len(test_set)} test")

            results, baseline = run_component_ablation(
                hp, device, train_loader, valid_loader, test_loader,
                variant_names, epochs=args.epochs, patience=args.patience, verbose=True,
                partial_csv_path=partial_path, resume=args.resume
            )

        for r in results:
            r['fold'] = fold

        all_fold_results.extend(results)

        print_ablation_summary(results, baseline)

        output_path = os.path.join(args.output_dir,
                                    f'ablation_fnet_{args.dataset}_{args.running_set}_fold{fold}_{timestamp}')

        metadata = {
            'dataset': args.dataset,
            'running_set': args.running_set,
            'fold': fold,
            'epochs': args.epochs,
            'patience': args.patience,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'timestamp': timestamp
        }
        save_ablation_results(results, output_path, metadata)

    if args.all_folds and len(folds) > 1:
        print("\n" + "=" * 80)
        print("AGGREGATED RESULTS (Mean +/- Std across all folds)")
        print("=" * 80)

        successful_fold_results = [r for r in all_fold_results if r.get('status', 'success') == 'success']

        if not successful_fold_results:
            print("No successful experiments to aggregate.")
        else:
            df = pd.DataFrame(successful_fold_results)

            agg_results = []
            for variant in df['variant'].unique():
                variant_df = df[df['variant'] == variant]
                agg = {
                    'variant': variant,
                    'description': variant_df['description'].iloc[0],
                    'ci_mean': variant_df['ci'].mean(),
                    'ci_std': variant_df['ci'].std(),
                    'mse_mean': variant_df['mse'].mean(),
                    'mse_std': variant_df['mse'].std(),
                    'r2_mean': variant_df['r2'].mean(),
                    'r2_std': variant_df['r2'].std(),
                }
                if 'ci_diff' in variant_df.columns:
                    agg['ci_diff_mean'] = variant_df['ci_diff'].mean()
                    agg['ci_diff_std'] = variant_df['ci_diff'].std()
                agg_results.append(agg)

            print(f"\n{'Variant':<20} {'CI (mean+/-std)':>18} {'MSE (mean+/-std)':>18}")
            print("-" * 80)

            for r in agg_results:
                ci_str = f"{r['ci_mean']:.4f}+/-{r['ci_std']:.4f}"
                mse_str = f"{r['mse_mean']:.4f}+/-{r['mse_std']:.4f}"
                print(f"{r['variant']:<20} {ci_str:>18} {mse_str:>18}")

            agg_df = pd.DataFrame(agg_results)
            agg_path = os.path.join(args.output_dir,
                                     f'ablation_fnet_{args.dataset}_{args.running_set}_AGGREGATED_{timestamp}')
            agg_df.to_csv(agg_path + '.csv', index=False)
            print(f"\nAggregated results saved to: {agg_path}.csv")

    print(f"\n{'=' * 70}")
    print(f"COMPONENT ABLATION STUDY COMPLETE!")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
