import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math


class SimpleCrossAttBlock(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.hidden_dim = latent_dim
        self.w = nn.Parameter(torch.rand(self.hidden_dim, self.hidden_dim)) # b,d,d        
        self.t_drug = nn.Sequential(nn.Linear(300, self.hidden_dim), 
                                    nn.LeakyReLU())
        self.t_prot = nn.Sequential(nn.Linear(100, self.hidden_dim), 
                                    nn.LeakyReLU())
           
        self.tv_drug = nn.Sequential(nn.Linear(300, self.hidden_dim), 
                                    nn.LeakyReLU())
        self.tv_prot = nn.Sequential(nn.Linear(100, self.hidden_dim), 
                                    nn.LeakyReLU())  
        
    def forward(self, drug_mat, prot_mat):
        drug_qk_mat = self.t_drug(drug_mat)  # b,200,512
        prot_qk_mat = self.t_prot(prot_mat)  # b,1000,512
        drug_v_mat = self.tv_drug(drug_mat)
        prot_v_mat = self.tv_prot(prot_mat)
        
        # att_score =  torch.einsum('bmd,dd,bnd->bmn', drug_qk_mat, self.w, prot_qk_mat) / self.hidden_dim  # b,m,n
        att_score =  torch.einsum('bmd,dd,bnd->bmn', drug_qk_mat, self.w, prot_qk_mat) / math.sqrt(self.hidden_dim)  # b,m,n
        drug_attention = torch.sum(att_score, 2)  # b,m
        prot_attention = torch.sum(att_score, 1)  # b,n
        
        drug_hidden = drug_v_mat*drug_attention.unsqueeze(2)  # b,m,d
        prot_hidden = prot_v_mat*prot_attention.unsqueeze(2)  # b,n,d
             
        drug_inter = F.max_pool1d(drug_hidden.permute(0, 2, 1), kernel_size=drug_hidden.shape[1], stride=1,
                                     padding=0, dilation=1, ceil_mode=False, return_indices=False).squeeze(2)  # b,d
        prot_inter = F.max_pool1d(prot_hidden.permute(0, 2, 1), kernel_size=prot_hidden.shape[1], stride=1,
                                     padding=0, dilation=1, ceil_mode=False, return_indices=False).squeeze(2)  # b,d
        return drug_inter, prot_inter, att_score


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
    
class MLPNet(nn.Module):    
    def __init__(self, dims, device, repete_n=1, act='LeakyReLU', dropout=0):
        super(MLPNet, self).__init__()        
        self.fc = nn.Linear(dims[0], dims[1])
        self.layers = []        
        self.repete_n = repete_n
        for i in range(self.repete_n):        
            self.layers.append(nn.Linear(dims[1], dims[1]))
        self.linears = nn.ModuleList(self.layers)
        self.activator = getattr(nn, act)()
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.ln = nn.LayerNorm(dims[-1])

    def forward(self, x):        
        x = self.activator(self.fc(self.do(x)))
        for i, ln_layer in enumerate(self.linears):
            h = self.activator(ln_layer(x))
            h = (h + x) * self.scale
            x = h        
        x = self.ln(x)
        return x


# Bi-linear Att in DrugBAN 
class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class BANLayer_MLP(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, device, act='ReLU', dropout=0.2, k=3):
        super(BANLayer_MLP, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = MLPNet((v_dim, h_dim * self.k), device, act="LeakyReLU", dropout=dropout)
        self.q_net = MLPNet((q_dim, h_dim * self.k), device, act="LeakyReLU", dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class TransformerDecoder(nn.ModuleList):
    def __init__(self, device, n_layers=3, n_heads=8):
        super(TransformerDecoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hid_dim = 128
        self.device = device
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hid_dim, nhead=self.n_heads, dim_feedforward=self.hid_dim * 4,dropout=0.1)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        
        
    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, hid_dim]
        trg = trg.permute(1,0,2).contiguous()
        src = src.permute(1,0,2).contiguous()
        # trg = [compound len, batch, hid_dim]
        trg = self.decoder(trg, src)        
        trg = trg.permute(1,0,2).contiguous()  # b, m, d

        w_trg =  F.softmax(torch.norm(trg, dim=2), dim=1)
        sum_trg = torch.einsum('bmd,bm->bd', trg, w_trg)
        return sum_trg


class CrossTransformerDecoder(nn.ModuleList):
    def __init__(self, n_layers=3, n_heads=8):
        super(CrossTransformerDecoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hid_dim = 128
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hid_dim, nhead=self.n_heads, dim_feedforward=self.hid_dim * 4,dropout=0.1)
        self.decoder_l = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)        
        self.decoder_r = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        
        
    def forward(self, trg, src):        
        trg = trg.permute(1,0,2).contiguous()  # m, b, d
        src = src.permute(1,0,2).contiguous()        
        n_trg = self.decoder_l(trg, src) 
        n_src = self.decoder_r(src, trg)         
        n_trg = n_trg.permute(1,0,2).contiguous()  # b, m, d
        n_src = n_src.permute(1,0,2).contiguous()  # b, m, d
        
        # weight sum
        w_trg = F.softmax(torch.norm(n_trg, dim=2), dim=1)  # b, m
        w_src = F.softmax(torch.norm(n_src, dim=2), dim=1)
        sum_trg = torch.einsum('bmd,bm->bd', n_trg, w_trg)
        sum_src = torch.einsum('bnd,bn->bd', n_src, w_src)
        return sum_trg, sum_src
        

class AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, hid_dim, n_heads=1, dropout=0.1):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        """ 
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Query and Value should always come from the same source (Aiming to forcus on), Key comes from the other source
        Self-Att : Both three Query, Key, Value come form the same source (For refining purpose)
        """

        batch_size = query.shape[0]

        Q = self.f_q(query)     # m,dim
        K = self.f_k(key)       # n,dim     
        V = self.f_v(value)     # n,dim

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix

class BidirectionAttBlock(nn.Module):
    def __init__(self, latent_dim=100):
        super(BidirectionAttBlock, self).__init__()
        self.bidat_num = 4      # 4
        self.latent_dim = latent_dim
        self.U = nn.ParameterList([nn.Parameter(torch.empty(size=(latent_dim, latent_dim))) for _ in range(self.bidat_num)])
        self.transform_c2p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.transform_p2c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        
        self.bihidden_c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.biatt_c = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.biatt_p = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])

        # 融合多头的输出
        self.comb_c = nn.Linear(latent_dim * self.bidat_num, latent_dim)
        self.comb_p = nn.Linear(latent_dim * self.bidat_num, latent_dim)
    
    def normalization(self, vector_present, threshold=0.1): 
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim = True)[0]
        de = vector_present_clone.max(1,keepdim = True)[0] - vector_present_clone.min(1,keepdim = True)[0]
        return num / de
    
    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax
    
    def forward(self, drug_mat, drug_mask, prot_mat, prot_mask):
        b = drug_mat.shape[0]
        for i in range(self.bidat_num):
            A = torch.tanh(torch.matmul(torch.matmul(drug_mat, self.U[i]), prot_mat.transpose(1, 2)))
            A = A * torch.matmul(drug_mask.view(b, -1, 1), prot_mask.view(b, 1, -1))  # b,m,n

            atoms_trans = torch.matmul(A, torch.tanh(self.transform_p2c[i](prot_mat)))  # b,m,d
            amino_trans = torch.matmul(A.transpose(1, 2), torch.tanh(self.transform_c2p[i](drug_mat)))

            atoms_tmp = torch.cat([torch.tanh(self.bihidden_c[i](drug_mat)), atoms_trans], dim=2)
            amino_tmp = torch.cat([torch.tanh(self.bihidden_p[i](prot_mat)), amino_trans], dim=2)

            atoms_att = self.mask_softmax(self.biatt_c[i](atoms_tmp).view(b, -1), drug_mask.view(b, -1))  # b,m
            amino_att = self.mask_softmax(self.biatt_p[i](amino_tmp).view(b, -1), prot_mask.view(b, -1))

            cf = torch.sum(drug_mat * atoms_att.view(b, -1, 1), dim=1)  # b,d
            pf = torch.sum(prot_mat * amino_att.view(b, -1, 1), dim=1)

            if i == 0:
                cat_cf = cf
                cat_pf = pf
            else:
                cat_cf = torch.cat([cat_cf.view(b, -1), cf.view(b, -1)], dim=1)
                cat_pf = torch.cat([cat_pf.view(b, -1), pf.view(b, -1)], dim=1)

        cf_final = self.comb_c(cat_cf)
        pf_final = self.comb_p(cat_pf)
        cf_pf = F.leaky_relu(torch.matmul(cf_final.view(b, -1, 1), pf_final.view(b, 1, -1)).view(b, -1), 0.1)
        return cf_pf  

