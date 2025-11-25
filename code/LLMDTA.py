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
        self.mlp_pred =  nn.Sequential(nn.Linear(1024, 512),
                                        nn.ELU(),
                                        nn.Linear(512, 1))
        

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask):
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
        
        pred = self.mlp_pred(h_pre + h_post)
        return pred
