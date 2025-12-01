import torch
import torch.nn as nn
import torch.nn.functional as F
from TryAttentionBlock import *

'''
    LLMDTA with Mixture of Experts
    Two encoders on pretrained vector/matrix respectively
    The pre-combine vec = plus two poolling vecs
    The post-combine vec = BilinearAtt(drug_mat, prot_mat)
    Then, use a residual connection on the above two combine-vecs
    
    MoE: Multiple expert MLPs predict the binding affinity, 
    each viewing the drug-protein interaction from different angles.
    A gating network routes the combined representation (h_pre + h_post) to top-k experts.
'''

class Encoder(nn.Module):
    def __init__(self, max_len, input_dim, device, hidden_dim=128):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = 7
        self.do = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList([nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)])
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
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


class GatingNetwork(nn.Module):
    """A gating network that selects experts based on combined drug-protein representation."""
    def __init__(self, input_dim, num_experts, hidden_dim=512):
        super(GatingNetwork, self).__init__()
        # Deeper gating network for better routing decisions
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )

    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=1)


class Expert(nn.Module):
    """An expert MLP that predicts binding affinity from a specific perspective."""
    def __init__(self, mlp_dim, dropout=0.1):
        super(Expert, self).__init__()
        # Each expert has the same architecture as the original mlp_pred
        # mlp_dim is expected to be [1024, 512, 1]
        # Use LayerNorm instead of BatchNorm to handle batch_size=1
        self.fc1 = nn.Linear(mlp_dim[0], mlp_dim[1])
        self.ln = nn.LayerNorm(mlp_dim[1])
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim[1], mlp_dim[2])

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LLMDTA_MoE(nn.Module):
    def __init__(self, hp, device, num_experts=4, top_k=2):
        super(LLMDTA_MoE, self).__init__()

        self.com_dim = hp.com_dim
        self.mlp_dim = hp.mlp_dim
        self.mol2vec_dim = hp.mol2vec_dim
        self.protvec_dim = hp.protvec_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.dropout = nn.Dropout(0.1)
        
        self.drug_embed = Encoder(hp.drug_max_len, self.mol2vec_dim, device)  # b, 100, 128
        self.prot_embed = Encoder(hp.prot_max_len, self.protvec_dim, device)  # b, 1022, 128

        # FeatureMat Encoder for attention         
        self.bilinear_att = weight_norm(BANLayer(128, 128, 256, h_out=2), name='h_mat', dim=None)
        
        # MLP for fusion
        self.bn = nn.BatchNorm1d(1024)
        self.linear_pre = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())     
        self.linear_post = nn.Sequential(nn.Linear(128*2, 1024), nn.ELU())
        
        # Mixture of Experts
        self.gating_network = GatingNetwork(1024, num_experts, hidden_dim=512)
        self.experts = nn.ModuleList([Expert(self.mlp_dim, dropout=0.1) for _ in range(num_experts)])
    
    def load_pretrained_base(self, pretrained_model_path):
        """Load pretrained weights from original LLMDTA model for shared layers."""
        print(f"Loading pretrained weights from {pretrained_model_path}")
        pretrained_state = torch.load(pretrained_model_path, map_location='cpu')
        
        # Load shared encoder and attention weights
        own_state = self.state_dict()
        for name, param in pretrained_state.items():
            # Remove 'module.' prefix if present (from DataParallel)
            if name.startswith('module.'):
                name = name[7:]
            
            # Load weights for shared components
            if name in own_state and own_state[name].shape == param.shape:
                if not name.startswith('mlp_pred'):  # Don't load old MLP weights
                    own_state[name].copy_(param)
                    print(f"  Loaded: {name}")
        
        print("Pretrained weights loaded successfully")

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask):
        # Pretrain embeddings
        drug_embed, drug_pool = self.drug_embed(drug_mat)  # 300 -> 128
        prot_embed, prot_pool = self.prot_embed(prot_mat)  # 100 -> 128      
           
        # Bilinear attention
        h, att = self.bilinear_att(drug_embed, prot_embed)  # 256
        
        # Fusion: combine pre and post representations
        h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))  # 128*2 -> 1024
        h_post = self.linear_post(h)  # 256 -> 1024
        
        # Combined representation for routing and prediction
        combined_repr = h_pre + h_post  # [batch_size, 1024]
        
        # Get routing weights from gating network
        routing_weights = self.gating_network(combined_repr)  # [batch_size, num_experts]
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=1)
        
        # Normalize the top-k weights
        norm_top_k_weights = F.softmax(top_k_weights, dim=1)
        
        # Collect expert predictions
        batch_size = combined_repr.size(0)
        expert_outputs = torch.zeros(batch_size, self.top_k, 1, device=combined_repr.device)
        
        # For each position in top_k, get the expert's prediction
        for k_idx in range(self.top_k):
            for batch_idx in range(batch_size):
                expert_idx = top_k_indices[batch_idx, k_idx]
                expert_outputs[batch_idx, k_idx] = self.experts[expert_idx](combined_repr[batch_idx:batch_idx+1])
        
        # Weighted sum of expert predictions
        weighted_outputs = expert_outputs.squeeze(-1) * norm_top_k_weights  # [batch_size, top_k]
        final_pred = torch.sum(weighted_outputs, dim=1, keepdim=True)  # [batch_size, 1]
        
        return final_pred, routing_weights

    def forward_with_routing_info(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask):
        """Forward pass that returns prediction, routing weights, and expert indices for analysis."""
        # Pretrain embeddings
        drug_embed, drug_pool = self.drug_embed(drug_mat)
        prot_embed, prot_pool = self.prot_embed(prot_mat)
           
        # Bilinear attention
        h, att = self.bilinear_att(drug_embed, prot_embed)
        
        # Fusion
        h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))
        h_post = self.linear_post(h)
        
        # Combined representation
        combined_repr = h_pre + h_post
        
        # Get routing weights
        routing_weights = self.gating_network(combined_repr)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=1)
        norm_top_k_weights = F.softmax(top_k_weights, dim=1)
        
        # Collect expert predictions
        batch_size = combined_repr.size(0)
        expert_outputs = torch.zeros(batch_size, self.top_k, 1, device=combined_repr.device)
        
        for k_idx in range(self.top_k):
            for batch_idx in range(batch_size):
                expert_idx = top_k_indices[batch_idx, k_idx]
                expert_outputs[batch_idx, k_idx] = self.experts[expert_idx](combined_repr[batch_idx:batch_idx+1])
        
        # Weighted sum
        weighted_outputs = expert_outputs.squeeze(-1) * norm_top_k_weights
        final_pred = torch.sum(weighted_outputs, dim=1, keepdim=True)
        
        return final_pred, routing_weights, top_k_indices, norm_top_k_weights


def load_balancing_loss(routing_weights, num_experts):
    """
    Computes the load balancing loss for the MoE model.
    This loss encourages the gating network to distribute inputs evenly across all experts.
    
    Args:
        routing_weights: [batch_size, num_experts] - softmax probabilities from gating network
        num_experts: number of experts in the model
    
    Returns:
        load_balance_loss: scalar tensor
    """
    # Average routing probability for each expert across the batch
    expert_usage = torch.mean(routing_weights, dim=0)  # [num_experts]
    
    # Ideal uniform distribution
    uniform_distribution = torch.ones_like(expert_usage) / num_experts
    
    # KL divergence or simply MSE between actual and uniform distribution
    # Using coefficient of variation to penalize imbalance
    cv_loss = torch.std(expert_usage) / (torch.mean(expert_usage) + 1e-10)
    
    return cv_loss
