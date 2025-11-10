import torch
import torch.nn as nn
import torch.nn.functional as F
from TryAttentionBlock import *

'''
    LLMDTA
    Two encoders on pretrained vector/matrix respectively
    The pre-combine vec = plus two poolling vecs
    The post-combine vec = BilinearAtt(drug_mat, prot_mat)
    Then, use a residual connection on the above two combine-vecs
    Lastly, we use a two-layers mlp to predict the bindding affinity
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


class LLMDTA(nn.Module):
    def __init__(self, hp, device):
        super(LLMDTA, self).__init__()

        self.com_dim = hp.com_dim
        self.mlp_dim = hp.mlp_dim
        self.mol2vec_dim = hp.mol2vec_dim
        self.protvec_dim = hp.protvec_dim                

        self.dropout = nn.Dropout(0.1)  # 0.5      
        
        self.drug_embed = Encoder(hp.drug_max_len, self.mol2vec_dim, device)  # b, 100, 128
        self.prot_embed = Encoder(hp.prot_max_len, self.protvec_dim, device)  # b, 1022, 128

        # FeatureMat Encoder for attention         
        self.bilinear_att = weight_norm(BANLayer(128, 128, 256, h_out=2), name='h_mat', dim=None)
        
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
           
        # bilinear-att
        h, att = self.bilinear_att(drug_embed, prot_embed)  # 256
        
        # Fusion
        h_pre = self.bn(self.linear_pre(torch.cat([drug_pool, prot_pool], dim=-1)))  # 128*2 -> 1024
        h_post = self.linear_post(h)  # 256 -> 1024
        
        pred = self.mlp_pred(h_pre + h_post)
        return pred
