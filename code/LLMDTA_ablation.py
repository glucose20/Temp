"""
LLMDTA Model Variants for Ablation Study
=========================================
This file contains model variants to evaluate the contribution of each component:

Model Architecture (Full LLMDTA):
    Input: Drug (Mol2Vec pretrained) + Protein (ESM2/ESM-C pretrained)
           ↓
    Encoder Block: 1D-CNN with skip-connections
           ↓
    Cross-Attention: Drug ↔ Protein interaction modeling
           ↓
    Self-Attention Pooling: Aggregate to fixed-size vectors
           ↓
    Feature Fusion: h_pre (pooled) + h_post (cross-attended) with residual
           ↓
    MoE Predictor: Mixture of Experts for final prediction

Ablation Variants:
    1. w/o Pretrain: Random embedding instead of Mol2Vec/ESM2
    2. w/o Encoder: Linear layer instead of 1D-CNN 
    3. w/o CrossAttention: No cross-attention, direct pooling
    4. w/o SelfAttnPool: Mean pooling instead of self-attention pooling
    5. w/o Residual: Only use h_post (no h_pre + h_post fusion)
    6. w/o MoE: Single predictor instead of mixture of experts
    7. Linear-only: Only pretrained features + linear predictor (like AI-Bind)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# COMPONENT MODULES
# ============================================================

class GatingNetwork(nn.Module):
    """A gating network that selects experts based on input features with Top-K selection."""
    def __init__(self, input_dim, num_experts, top_k=2, noise_std=0.1):
        super(GatingNetwork, self).__init__()
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
    """Self-attention pooling layer to aggregate sequence into a single vector"""
    def __init__(self, hidden_dim):
        super(SelfAttentionPooling, self).__init__()
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
    """Multi-head cross attention module"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
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
        
        output = self.out_ln(attn_output + query)
        return output


# ============================================================
# ENCODER VARIANTS
# ============================================================

class Encoder(nn.Module):
    """Full Encoder with 1D-CNN and skip-connections"""
    def __init__(self, max_len, input_dim, device, hidden_dim=128):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = 7
        self.do = nn.Dropout(0.1)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([0.5])))
        
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
            nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
            nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)
        ])
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        feat_map = self.input_norm(feat_map)
        h_map = self.fc(feat_map)
        h_map = h_map.permute(0, 2, 1)
        
        for conv in self.convs:
            conved = conv(self.do(h_map))
            conved = F.glu(conved, dim=1)
            conved = (conved + h_map) * self.scale  # Skip connection
            h_map = conved
        
        pool_map = self.max_pool(h_map).squeeze(-1)
        h_map = h_map.permute(0, 2, 1)
        h_map = self.ln(h_map)
        return h_map, pool_map


class LinearEncoder(nn.Module):
    """Simple Linear Encoder (w/o CNN) - for ablation"""
    def __init__(self, max_len, input_dim, device, hidden_dim=128):
        super(LinearEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        feat_map = self.input_norm(feat_map)
        h_map = self.fc(feat_map)  # (b, seq, hidden_dim)
        h_map = self.ln(h_map)
        
        # Pooling
        h_map_t = h_map.permute(0, 2, 1)  # (b, hidden_dim, seq)
        pool_map = self.max_pool(h_map_t).squeeze(-1)  # (b, hidden_dim)
        
        return h_map, pool_map


class EmbeddingEncoder(nn.Module):
    """Embedding-based Encoder (w/o Pretrain) - learns embeddings from scratch"""
    def __init__(self, max_len, vocab_size, device, hidden_dim=128, embedding_dim=128):
        super(EmbeddingEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.do = nn.Dropout(0.1)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([0.5])))
        
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim*2, 7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim*2, 7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim*2, 7, padding=3)
        ])
        self.max_pool = nn.MaxPool1d(max_len)
    
    def forward(self, indices):
        # indices: (b, seq) - token indices
        x = self.embedding(indices)  # (b, seq, embedding_dim)
        h_map = self.fc(x)
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


# ============================================================
# ABLATION MODEL VARIANTS
# ============================================================

class LLMDTA_Full(nn.Module):
    """Full LLMDTA Model (Baseline for comparison)"""
    def __init__(self, hp, device):
        super(LLMDTA_Full, self).__init__()
        self.variant_name = "Full Model"
        
        self.hidden_dim = 128
        self.num_heads = 8
        self.num_experts = getattr(hp, 'num_experts', 4)
        self.top_k = getattr(hp, 'top_k', 2)
        self.moe_noise_std = getattr(hp, 'moe_noise_std', 0.1)
        self.load_balance_weight = getattr(hp, 'load_balance_weight', 0.01)
        
        # Encoders (Pretrained features)
        self.drug_embed = Encoder(hp.drug_max_len, hp.mol2vec_dim, device)
        self.prot_embed = Encoder(hp.prot_max_len, hp.protvec_dim, device)
        
        # Cross Attention
        self.drug_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        self.prot_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        
        # Self-attention pooling
        self.drug_attn_pool = SelfAttentionPooling(self.hidden_dim)
        self.prot_attn_pool = SelfAttentionPooling(self.hidden_dim)
        
        # Fusion
        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        self.linear_post = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        
        # MoE
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, 1)
            ) for _ in range(self.num_experts)
        ])
        self.gating = GatingNetwork(1024, self.num_experts, self.top_k, self.moe_noise_std)
        
        self.register_buffer('expert_usage_count', torch.zeros(self.num_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))
    
    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, return_gate_info=False):
        # Encoder
        drug_embed, drug_pool = self.drug_embed(drug_mat)
        prot_embed, prot_pool = self.prot_embed(prot_mat)
        
        # Cross Attention
        new_drug_embed = self.drug_cross_attn(drug_embed, prot_embed)
        new_prot_embed = self.prot_cross_attn(prot_embed, drug_embed)
        
        # Self-attention pooling
        drug_cross_pool = self.drug_attn_pool(new_drug_embed)
        prot_cross_pool = self.prot_attn_pool(new_prot_embed)
        
        # Fusion with residual
        h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))
        h_post = self.linear_post(torch.cat([drug_cross_pool, prot_cross_pool], dim=-1))
        h_combined = h_pre + h_post  # Residual
        
        # MoE prediction
        gate_weights, raw_weights = self.gating(h_combined, return_all_weights=True)
        
        if self.training:
            with torch.no_grad():
                selected = (gate_weights > 0).float().sum(dim=0)
                self.expert_usage_count += selected
                self.total_samples += gate_weights.size(0)
        
        expert_outputs = torch.stack([exp(h_combined) for exp in self.experts], dim=1)
        pred = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        
        if return_gate_info:
            return pred, {'gate_weights': gate_weights, 'raw_weights': raw_weights,
                         'expert_outputs': expert_outputs.squeeze(-1),
                         'selected_experts': (gate_weights > 0).int()}
        return pred
    
    def compute_load_balance_loss(self, gate_weights):
        expert_usage = gate_weights.mean(dim=0)
        return self.num_experts * torch.sum(expert_usage ** 2)
    
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
            'usage_std': usage_rate.std().item(),
        }


class LLMDTA_woEncoder(nn.Module):
    """w/o Encoder: Replace 1D-CNN with Linear layer"""
    def __init__(self, hp, device):
        super(LLMDTA_woEncoder, self).__init__()
        self.variant_name = "w/o Encoder (Linear only)"
        
        self.hidden_dim = 128
        self.num_heads = 8
        self.num_experts = getattr(hp, 'num_experts', 4)
        self.top_k = getattr(hp, 'top_k', 2)
        self.moe_noise_std = getattr(hp, 'moe_noise_std', 0.1)
        self.load_balance_weight = getattr(hp, 'load_balance_weight', 0.01)
        
        # LINEAR Encoders (no CNN)
        self.drug_embed = LinearEncoder(hp.drug_max_len, hp.mol2vec_dim, device)
        self.prot_embed = LinearEncoder(hp.prot_max_len, hp.protvec_dim, device)
        
        # Rest same as full model
        self.drug_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        self.prot_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        self.drug_attn_pool = SelfAttentionPooling(self.hidden_dim)
        self.prot_attn_pool = SelfAttentionPooling(self.hidden_dim)
        
        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        self.linear_post = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        
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
        if self.total_samples == 0: return None
        usage_rate = self.expert_usage_count / self.total_samples
        usage_rate = usage_rate / (usage_rate.sum() + 1e-8)
        # return {'usage_entropy': -torch.sum(usage_rate * torch.log(usage_rate + 1e-8)).item()}
        stats = {
            'expert_usage_rate': usage_rate.cpu().numpy(),  # How often each expert is selected
            'usage_entropy': -torch.sum(usage_rate * torch.log(usage_rate + 1e-8)).item(),  # Higher = more balanced
            'dominant_expert': torch.argmax(usage_rate).item(),
            'usage_std': usage_rate.std().item(),  # Lower = more balanced
        }
        return stats


class LLMDTA_woCrossAttention(nn.Module):
    """w/o Cross-Attention: No drug-protein interaction modeling"""
    def __init__(self, hp, device):
        super(LLMDTA_woCrossAttention, self).__init__()
        self.variant_name = "w/o Cross-Attention"
        
        self.hidden_dim = 128
        self.num_experts = getattr(hp, 'num_experts', 4)
        self.top_k = getattr(hp, 'top_k', 2)
        self.moe_noise_std = getattr(hp, 'moe_noise_std', 0.1)
        self.load_balance_weight = getattr(hp, 'load_balance_weight', 0.01)
        
        self.drug_embed = Encoder(hp.drug_max_len, hp.mol2vec_dim, device)
        self.prot_embed = Encoder(hp.prot_max_len, hp.protvec_dim, device)
        
        # NO Cross Attention - just use pooled features directly
        # Use mean pooling instead of self-attention on encoder output
        
        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        # No linear_post since no cross-attention
        
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
        
        # NO cross-attention, directly use pooled features
        h_combined = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))
        
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
        if self.total_samples == 0: return None
        usage_rate = self.expert_usage_count / self.total_samples
        usage_rate = usage_rate / (usage_rate.sum() + 1e-8)
        return {'usage_entropy': -torch.sum(usage_rate * torch.log(usage_rate + 1e-8)).item()}


class LLMDTA_woSelfAttnPool(nn.Module):
    """w/o Self-Attention Pooling: Use mean pooling instead"""
    def __init__(self, hp, device):
        super(LLMDTA_woSelfAttnPool, self).__init__()
        self.variant_name = "w/o Self-Attention Pooling (Mean Pool)"
        
        self.hidden_dim = 128
        self.num_heads = 8
        self.num_experts = getattr(hp, 'num_experts', 4)
        self.top_k = getattr(hp, 'top_k', 2)
        self.moe_noise_std = getattr(hp, 'moe_noise_std', 0.1)
        self.load_balance_weight = getattr(hp, 'load_balance_weight', 0.01)
        
        self.drug_embed = Encoder(hp.drug_max_len, hp.mol2vec_dim, device)
        self.prot_embed = Encoder(hp.prot_max_len, hp.protvec_dim, device)
        
        self.drug_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        self.prot_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        
        # NO Self-Attention Pooling - use mean pooling
        
        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        self.linear_post = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        
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
        
        # MEAN pooling instead of self-attention pooling
        drug_cross_pool = new_drug_embed.mean(dim=1)  # (b, hidden_dim)
        prot_cross_pool = new_prot_embed.mean(dim=1)  # (b, hidden_dim)
        
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
        if self.total_samples == 0: return None
        usage_rate = self.expert_usage_count / self.total_samples
        usage_rate = usage_rate / (usage_rate.sum() + 1e-8)
        return {'usage_entropy': -torch.sum(usage_rate * torch.log(usage_rate + 1e-8)).item()}


class LLMDTA_woResidual(nn.Module):
    """w/o Residual: Only use h_post, no h_pre + h_post fusion"""
    def __init__(self, hp, device):
        super(LLMDTA_woResidual, self).__init__()
        self.variant_name = "w/o Residual Fusion"
        
        self.hidden_dim = 128
        self.num_heads = 8
        self.num_experts = getattr(hp, 'num_experts', 4)
        self.top_k = getattr(hp, 'top_k', 2)
        self.moe_noise_std = getattr(hp, 'moe_noise_std', 0.1)
        self.load_balance_weight = getattr(hp, 'load_balance_weight', 0.01)
        
        self.drug_embed = Encoder(hp.drug_max_len, hp.mol2vec_dim, device)
        self.prot_embed = Encoder(hp.prot_max_len, hp.protvec_dim, device)
        
        self.drug_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        self.prot_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        self.drug_attn_pool = SelfAttentionPooling(self.hidden_dim)
        self.prot_attn_pool = SelfAttentionPooling(self.hidden_dim)
        
        # Only linear_post, no fusion with h_pre
        self.linear_post = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        self.bn = nn.BatchNorm1d(1024)
        
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
        
        # NO residual - only use cross-attended features
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
    
    def compute_load_balance_loss(self, gate_weights):
        return self.num_experts * torch.sum(gate_weights.mean(dim=0) ** 2)
    
    def reset_usage_stats(self):
        self.expert_usage_count.zero_()
        self.total_samples.zero_()
    
    def get_expert_usage_stats(self):
        if self.total_samples == 0: return None
        usage_rate = self.expert_usage_count / self.total_samples
        usage_rate = usage_rate / (usage_rate.sum() + 1e-8)
        return {'usage_entropy': -torch.sum(usage_rate * torch.log(usage_rate + 1e-8)).item()}


class LLMDTA_woMoE(nn.Module):
    """w/o MoE: Single predictor instead of Mixture of Experts"""
    def __init__(self, hp, device):
        super(LLMDTA_woMoE, self).__init__()
        self.variant_name = "w/o MoE (Single Predictor)"
        
        self.hidden_dim = 128
        self.num_heads = 8
        
        self.drug_embed = Encoder(hp.drug_max_len, hp.mol2vec_dim, device)
        self.prot_embed = Encoder(hp.prot_max_len, hp.protvec_dim, device)
        
        self.drug_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        self.prot_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)
        self.drug_attn_pool = SelfAttentionPooling(self.hidden_dim)
        self.prot_attn_pool = SelfAttentionPooling(self.hidden_dim)
        
        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        self.linear_post = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        
        # Single predictor (no MoE)
        self.predictor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
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
            # Return dummy gate info for compatibility
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


class LLMDTA_Linear(nn.Module):
    """Linear-only: Pretrained features + Linear predictor (like AI-Bind)"""
    def __init__(self, hp, device):
        super(LLMDTA_Linear, self).__init__()
        self.variant_name = "Linear-only (AI-Bind style)"
        
        # Direct projection from pretrained features
        self.drug_proj = nn.Linear(hp.mol2vec_dim, 256)
        self.prot_proj = nn.Linear(hp.protvec_dim, 256)
        
        # Simple pooling (mean)
        self.drug_max_len = hp.drug_max_len
        self.prot_max_len = hp.prot_max_len
        
        # Linear predictor
        self.predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, return_gate_info=False):
        # Project pretrained features
        drug_feat = self.drug_proj(drug_mat)  # (b, seq, 256)
        prot_feat = self.prot_proj(prot_mat)  # (b, seq, 256)
        
        # Mean pooling
        drug_pool = drug_feat.mean(dim=1)  # (b, 256)
        prot_pool = prot_feat.mean(dim=1)  # (b, 256)
        
        # Concatenate and predict
        combined = torch.cat([drug_pool, prot_pool], dim=-1)  # (b, 512)
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

def get_ablation_model(variant_name, hp, device):
    """
    Factory function to get model variant by name.
    
    Args:
        variant_name: One of ['full', 'wo_encoder', 'wo_cross_attn', 
                              'wo_self_attn_pool', 'wo_residual', 'wo_moe', 'linear_only']
        hp: HyperParameter object
        device: torch device
    
    Returns:
        Model instance
    """
    variants = {
        'full': LLMDTA_Full,
        'wo_encoder': LLMDTA_woEncoder,
        'wo_cross_attn': LLMDTA_woCrossAttention,
        'wo_self_attn_pool': LLMDTA_woSelfAttnPool,
        'wo_residual': LLMDTA_woResidual,
        'wo_moe': LLMDTA_woMoE,
        'linear_only': LLMDTA_Linear,
    }
    
    if variant_name not in variants:
        raise ValueError(f"Unknown variant: {variant_name}. Choose from {list(variants.keys())}")
    
    return variants[variant_name](hp, device)


def get_all_variants():
    """Return list of all variant names for ablation study."""
    return [
        ('full', 'Full Model (LLMDTA)'),
        ('wo_encoder', 'w/o Encoder (Linear instead of CNN)'),
        ('wo_cross_attn', 'w/o Cross-Attention'),
        ('wo_self_attn_pool', 'w/o Self-Attention Pooling'),
        ('wo_residual', 'w/o Residual Fusion'),
        ('wo_moe', 'w/o MoE (Single Predictor)'),
        ('linear_only', 'Linear-only (AI-Bind style)'),
    ]
