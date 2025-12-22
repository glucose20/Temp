import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from TryAttentionBlock import *

'''
    LLMDTA
    Two encoders on pretrained vector/matrix respectively
    The pre-combine vec = plus two poolling vecs
    The post-combine vec = Flash Attention Cross Attention(drug_mat, prot_mat)
    Then, use a residual connection on the above two combine-vecs
    Lastly, we use a two-layers mlp to predict the bindding affinity
'''


class GatingNetwork(nn.Module):
    """A gating network that selects experts based on input features with Top-K selection."""
    def __init__(self, input_dim, num_experts, top_k=2, noise_std=0.1):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.initial_noise_std = noise_std
        self.noise_std = noise_std  # Noise for load balancing during training
        self.layer = nn.Linear(input_dim, num_experts)
    
    def set_noise_std(self, noise_std):
        """Update noise std (for adaptive scheduling)"""
        self.noise_std = noise_std
    
    def forward(self, x, return_all_weights=False):
        """
        Args:
            x: input features (batch, input_dim)
            return_all_weights: if True, return raw softmax weights for analysis
        Returns:
            gate_weights: sparse top-k weights (batch, num_experts)
            raw_weights: original softmax weights (only if return_all_weights=True)
        """
        logits = self.layer(x)
        
        # Add noise during training for better exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        raw_weights = F.softmax(logits, dim=1)  # (batch, num_experts)
        
        # Top-K selection: keep only top-k experts
        if self.top_k < self.num_experts:
            top_k_weights, top_k_indices = torch.topk(raw_weights, self.top_k, dim=1)
            # Renormalize top-k weights
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
            
            # Create sparse gate weights
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
        # x: (batch, seq_len, hidden_dim)
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch, hidden_dim)
        return pooled


class CrossAttention(nn.Module):
    """Multi-head cross attention module"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        Args:
            query: (batch, seq_q, hidden_dim) - the sequence that attends
            key_value: (batch, seq_kv, hidden_dim) - the sequence being attended to
        Returns:
            output: (batch, seq_q, hidden_dim) - attended output with residual connection
        """
        batch_size = query.shape[0]
        
        # Project Q, K, V
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Reshape for attention: (b, seq, h, d) -> (b, h, seq, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (b, h, seq_q, d)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (b, seq_q, h, d)
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)  # (b, seq_q, hidden_dim)
        
        # Residual connection and layer norm
        output = self.out_ln(attn_output + query)
        return output

class Encoder(nn.Module):
    def __init__(self, max_len, input_dim, device, hidden_dim=128):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = 7
        self.do = nn.Dropout(0.1)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([0.5])))
        
        # Add normalization layer before FC to stabilize embeddings
        self.input_norm = nn.LayerNorm(self.input_dim)
        
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList([nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)])
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        # Normalize input embeddings first
        feat_map = self.input_norm(feat_map)
        
        h_map = self.fc(feat_map)
        h_map = h_map.permute(0,2,1)  
              
        for i, conv in enumerate(self.convs):
            conved = conv(self.do(h_map))
            conved = F.glu(conved, dim=1)
            conved = (conved+h_map)* self.scale
            h_map = conved
        
        pool_map = self.max_pool(h_map).squeeze(-1)  # b,d
        h_map = h_map.permute(0,2,1)
        h_map = self.ln(h_map)    # b, len, d
        return h_map, pool_map


class LLMDTA(nn.Module):
    def __init__(self, hp, device):
        super(LLMDTA, self).__init__()

        self.com_dim = hp.com_dim
        self.mlp_dim = hp.mlp_dim
        self.mol2vec_dim = hp.mol2vec_dim
        self.protvec_dim = hp.protvec_dim                
        self.hidden_dim = 128
        self.num_heads = 8  # Number of attention heads for flash attention
        
        # MoE hyperparameters (from hp config)
        self.num_experts = hp.num_experts  # Number of experts in MoE
        self.top_k = hp.top_k  # Number of experts to select per sample
        self.moe_noise_std = getattr(hp, 'moe_noise_std', 0.1)  # Noise for exploration
        self.load_balance_weight = getattr(hp, 'load_balance_weight', 0.01)  # Load balance loss weight

        self.dropout = nn.Dropout(0.1)  # 0.5      
        
        self.drug_embed = Encoder(hp.drug_max_len, self.mol2vec_dim, device)  # b, 100, 128
        self.prot_embed = Encoder(hp.prot_max_len, self.protvec_dim, device)  # b, 1022, 128

        # Cross Attention modules
        self.drug_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)  # drug attending to protein
        self.prot_cross_attn = CrossAttention(self.hidden_dim, self.num_heads)  # protein attending to drug
        
        # Self-attention pooling layers
        self.drug_attn_pool = SelfAttentionPooling(self.hidden_dim)
        self.prot_attn_pool = SelfAttentionPooling(self.hidden_dim)
        
        # MLP
        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())     
        self.linear_post = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        
        # Mixture of Experts: each expert views drug/protein combinations differently
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.ELU(),
                nn.Dropout(0.1),  # Add dropout for regularization
                nn.Linear(512, 1)
            ) for _ in range(self.num_experts)
        ])
        
        # Gating network with Top-K selection
        self.gating = GatingNetwork(1024, self.num_experts, top_k=self.top_k, noise_std=self.moe_noise_std)
        
        # For tracking expert usage statistics
        self.register_buffer('expert_usage_count', torch.zeros(self.num_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))
        

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, return_gate_info=False):
        # Pretrain
        drug_embed, drug_pool = self.drug_embed(drug_mat)  # 300 -> 128
        prot_embed, prot_pool = self.prot_embed(prot_mat)  # 100 -> 128      
        
        # Cross Attention
        new_drug_embed = self.drug_cross_attn(drug_embed, prot_embed)  # drug attending to protein
        new_prot_embed = self.prot_cross_attn(prot_embed, drug_embed)  # protein attending to drug
        
        # Self-attention pooling
        drug_cross_pool = self.drug_attn_pool(new_drug_embed)  # (b, hidden_dim)
        prot_cross_pool = self.prot_attn_pool(new_prot_embed)  # (b, hidden_dim)
        
        # Fusion
        h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))  # 128*2 -> 1024
        h_post = self.linear_post(torch.cat([drug_cross_pool, prot_cross_pool], dim=-1))  # 128*2 -> 1024
        
        # Combined representation for MoE
        h_combined = h_pre + h_post  # (batch, 1024)
        
        # Get gating weights (sparse top-k) and raw weights for analysis
        gate_weights, raw_weights = self.gating(h_combined, return_all_weights=True)  # (batch, num_experts)
        
        # Update expert usage statistics (for monitoring)
        if self.training:
            with torch.no_grad():
                # Count which experts are selected (gate_weight > 0)
                selected = (gate_weights > 0).float().sum(dim=0)  # (num_experts,)
                self.expert_usage_count += selected
                self.total_samples += gate_weights.size(0)
        
        # Get predictions from each expert
        expert_outputs = []
        for i in range(self.num_experts):
            expert_pred = self.experts[i](h_combined)  # (batch, 1)
            expert_outputs.append(expert_pred)
        
        # Stack expert outputs: (batch, num_experts, 1)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Weighted sum of expert predictions (only top-k contribute)
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (batch, num_experts, 1)
        pred = torch.sum(gate_weights_expanded * expert_outputs, dim=1)  # (batch, 1)
        
        if return_gate_info:
            return pred, {
                'gate_weights': gate_weights,  # Sparse top-k weights
                'raw_weights': raw_weights,    # Original softmax weights
                'expert_outputs': expert_outputs.squeeze(-1),  # (batch, num_experts)
                'selected_experts': (gate_weights > 0).int()   # Which experts were selected
            }
        
        return pred
    
    def compute_load_balance_loss(self, gate_weights):
        """
        Compute load balancing loss to encourage uniform expert usage.
        This helps prevent collapse to using only a few experts.
        
        Args:
            gate_weights: (batch, num_experts) gating weights
        Returns:
            load_balance_loss: scalar loss value
        """
        # Average gate weight per expert across batch
        expert_usage = gate_weights.mean(dim=0)  # (num_experts,)
        
        # Ideal uniform distribution
        ideal = 1.0 / self.num_experts
        
        # Loss is variance from uniform distribution
        load_balance_loss = self.num_experts * torch.sum(expert_usage ** 2)
        
        return load_balance_loss
    
    def get_expert_usage_stats(self):
        """Get statistics about expert usage for monitoring."""
        if self.total_samples == 0:
            return None
        
        usage_rate = self.expert_usage_count / self.total_samples
        # Normalize to ensure it sums to 1 (for proper entropy calculation)
        usage_rate = usage_rate / (usage_rate.sum() + 1e-8)
        
        stats = {
            'expert_usage_rate': usage_rate.cpu().numpy(),  # How often each expert is selected
            'usage_entropy': -torch.sum(usage_rate * torch.log(usage_rate + 1e-8)).item(),  # Higher = more balanced
            'dominant_expert': torch.argmax(usage_rate).item(),
            'usage_std': usage_rate.std().item(),  # Lower = more balanced
        }
        return stats
    
    def reset_usage_stats(self):
        """Reset expert usage statistics (call at start of each epoch)."""
        self.expert_usage_count.zero_()
        self.total_samples.zero_()
    
    def adaptive_update(self, epoch, total_epochs, moe_stats=None):
        """
        Automatically adjust MoE parameters during training.
        Call this at the end of each epoch.
        
        Args:
            epoch: current epoch (1-indexed)
            total_epochs: total number of epochs
            moe_stats: expert usage stats from get_expert_usage_stats()
        Returns:
            dict of adjusted parameters
        """
        progress = epoch / total_epochs  # 0 -> 1
        adjustments = {}
        
        # 1. Noise annealing: high exploration early, low later (like temperature)
        # Decay from initial_noise_std to 0.01 * initial_noise_std
        initial_noise = self.gating.initial_noise_std
        new_noise = initial_noise * (1 - 0.9 * progress)  # Decay to 10% at end
        self.gating.set_noise_std(new_noise)
        adjustments['noise_std'] = new_noise
        
        # 2. Adaptive load balance weight based on expert usage balance
        if moe_stats is not None:
            usage_std = moe_stats['usage_std']
            # If experts are very imbalanced (high std), increase load balance weight
            # Target std for 4 experts with top-2: ~0.15-0.2
            target_std = 0.15
            if usage_std > target_std * 1.5:  # Too imbalanced
                self.load_balance_weight = min(self.load_balance_weight * 1.2, 0.1)
                adjustments['lb_adjustment'] = 'increased'
            elif usage_std < target_std * 0.5 and progress > 0.3:  # Well balanced, can reduce
                self.load_balance_weight = max(self.load_balance_weight * 0.9, 0.001)
                adjustments['lb_adjustment'] = 'decreased'
            else:
                adjustments['lb_adjustment'] = 'stable'
            adjustments['load_balance_weight'] = self.load_balance_weight
        
        return adjustments
