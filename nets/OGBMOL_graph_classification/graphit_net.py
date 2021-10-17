import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm 

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

"""
    GraphiT-GT and GraphiT-GT-LSPE
    
"""

from layers.graphit_gt_layer import GraphiT_GT_Layer
from layers.graphit_gt_lspe_layer import GraphiT_GT_LSPE_Layer
from layers.mlp_readout_layer import MLPReadout

class GraphiTNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        full_graph = net_params['full_graph']
        gamma = net_params['gamma']
        self.adaptive_edge_PE = net_params['adaptive_edge_PE']
        
        GT_layers = net_params['L']
        GT_hidden_dim = net_params['hidden_dim']
        GT_out_dim = net_params['out_dim']
        GT_n_heads = net_params['n_heads']
        
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']

        n_classes = net_params['n_classes']
        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.pe_init = net_params['pe_init']
        
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        
        self.pos_enc_dim = net_params['pos_enc_dim']
        
        if self.pe_init in ['rand_walk']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, GT_hidden_dim)
        
        self.embedding_h = AtomEncoder(GT_hidden_dim)
        self.embedding_e = BondEncoder(GT_hidden_dim)
        
        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList([ GraphiT_GT_LSPE_Layer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph,
                                                                dropout, self.layer_norm, self.batch_norm, self.residual, self.adaptive_edge_PE) for _ in range(GT_layers-1) ])
            self.layers.append(GraphiT_GT_LSPE_Layer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph,
                                                     dropout, self.layer_norm, self.batch_norm, self.residual, self.adaptive_edge_PE))
        else: 
            # NoPE
            self.layers = nn.ModuleList([ GraphiT_GT_Layer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph,
                                                                dropout, self.layer_norm, self.batch_norm, self.residual, self.adaptive_edge_PE) for _ in range(GT_layers-1) ])
            self.layers.append(GraphiT_GT_Layer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph,
                                                     dropout, self.layer_norm, self.batch_norm, self.residual, self.adaptive_edge_PE))
        
        self.MLP_layer = MLPReadout(GT_out_dim, n_classes)   
 
        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(GT_out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(GT_out_dim+self.pos_enc_dim, GT_out_dim)
        
        self.g = None              # For util; To be accessed in loss() function
        
    def forward(self, g, h, p, e, snorm_n):
        
        h = self.embedding_h(h)
        e = self.embedding_e(e)

        if self.pe_init in ['rand_walk']:
            p = self.embedding_p(p)
        
        for conv in self.layers:
            h, p = conv(g, h, p, e, snorm_n)
            
        g.ndata['h'] = h
        
        if self.pe_init == 'rand_walk':
            p = self.p_out(p)
            g.ndata['p'] = p
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
        
        return self.MLP_layer(hg), g
        
    def loss(self, pred, labels):
        
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = torch.nn.BCEWithLogitsLoss()(pred, labels)
        
        if self.use_lapeig_loss:
            raise NotImplementedError
        else:
            loss = loss_a
            
        return loss