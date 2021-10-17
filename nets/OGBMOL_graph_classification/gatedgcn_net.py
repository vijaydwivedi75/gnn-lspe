import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm 

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

"""
    GatedGCN and GatedGCN-LSPE
    
"""

from layers.gatedgcn_layer import GatedGCNLayer
from layers.gatedgcn_lspe_layer import GatedGCNLSPELayer
from layers.mlp_readout_layer import MLPReadout


class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.pe_init = net_params['pe_init']
        self.n_classes = net_params['n_classes']
        
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        
        self.pos_enc_dim = net_params['pos_enc_dim']
        
        if self.pe_init in ['rand_walk', 'lap_pe']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)
        
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        
        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList([ GatedGCNLSPELayer(hidden_dim, hidden_dim, dropout, self.batch_norm, self.residual) 
                    for _ in range(n_layers-1) ]) 
            self.layers.append(GatedGCNLSPELayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        else: 
            # NoPE or LapPE
            self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout, self.batch_norm, self.residual) 
                    for _ in range(n_layers-1) ]) 
            self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(out_dim+self.pos_enc_dim, out_dim)
        
        self.g = None              # For util; To be accessed in loss() function
        
    def forward(self, g, h, p, e, snorm_n):
        
        h = self.atom_encoder(h)
        e = self.bond_encoder(e)

        if self.pe_init in ['rand_walk', 'lap_pe']:
            p = self.embedding_p(p)
            
        if self.pe_init == 'lap_pe':
            h = h + p
            p = None
        
        for conv in self.layers:
            h, p, e = conv(g, h, p, e, snorm_n)
            
        g.ndata['h'] = h
        
        if self.pe_init == 'rand_walk':
            p = self.p_out(p)
            g.ndata['p'] = p
            
        if self.use_lapeig_loss:
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p']**2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms+1e-6)            
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p
        
        if self.pe_init == 'rand_walk':
            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'],g.ndata['p']),dim=-1))
            g.ndata['h'] = hp
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        self.g = g # For util; To be accessed in loss() function
        
        if self.n_classes == 128:
            return_g = None # not passing PCBA graphs due to memory
        else:
            return_g = g
            
        return self.MLP_layer(hg), return_g
        
    def loss(self, pred, labels):
        
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = torch.nn.BCEWithLogitsLoss()(pred, labels)
        
        if self.use_lapeig_loss:
            raise NotImplementedError
        else:
            loss = loss_a
            
        return loss