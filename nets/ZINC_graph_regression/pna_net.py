import torch
import torch.nn as nn
import dgl
from scipy import sparse as sp
from scipy.sparse.linalg import norm


"""
    PNA and PNA-LSPE
    
"""

from layers.pna_layer import PNALayer
from layers.pna_lspe_layer import PNALSPELayer
from layers.pna_utils import GRU
from layers.mlp_readout_layer import MLPReadout

class PNANet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.towers = net_params['towers']
        self.divide_input_first = net_params['divide_input_first']
        self.divide_input_last = net_params['divide_input_last']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.gru_enable = net_params['gru']
        device = net_params['device']
        self.device = device
        self.pe_init = net_params['pe_init']
        
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        
        self.pos_enc_dim = net_params['pos_enc_dim']
        
        if self.pe_init in ['rand_walk']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)        

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, edge_dim)
            
        
        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList([PNALSPELayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout,
                                                  graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                                  residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                                  avg_d=self.avg_d, towers=self.towers, edge_features=self.edge_feat,
                                                  edge_dim=edge_dim, divide_input=self.divide_input_first,
                                                  pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers) for _
                                         in range(n_layers - 1)])
            self.layers.append(PNALSPELayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                        graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                        residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                        avg_d=self.avg_d, towers=self.towers, divide_input=self.divide_input_last,
                                        edge_features=self.edge_feat, edge_dim=edge_dim,
                                        pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers))
        else: 
            # NoPE
            self.layers = nn.ModuleList([PNALayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout,
                                                  graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                                  residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                                  avg_d=self.avg_d, towers=self.towers, edge_features=self.edge_feat,
                                                  edge_dim=edge_dim, divide_input=self.divide_input_first,
                                                  pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers) for _
                                         in range(n_layers - 1)])
            self.layers.append(PNALayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                        graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                        residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                        avg_d=self.avg_d, towers=self.towers, divide_input=self.divide_input_last,
                                        edge_features=self.edge_feat, edge_dim=edge_dim,
                                        pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers))

        if self.gru_enable:
            self.gru = GRU(hidden_dim, hidden_dim, device)

        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem
        
        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(out_dim+self.pos_enc_dim, out_dim)
        
        self.g = None              # For util; To be accessed in loss() function

    def forward(self, g, h, p, e, snorm_n):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        
        if self.pe_init in ['rand_walk']:
            p = self.embedding_p(p) 
        
        if self.edge_feat:
            e = self.embedding_e(e)

        for i, conv in enumerate(self.layers):
            h_t, p_t = conv(g, h, p, e, snorm_n)
            if self.gru_enable and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h, p = h_t, p_t

        g.ndata['h'] = h
        
        if self.pe_init == 'rand_walk':
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            p = self.p_out(p)
            g.ndata['p'] = p
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p']**2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms)            
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

    def loss(self, scores, targets):
        
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = nn.L1Loss()(scores, targets)
        
        if self.use_lapeig_loss:
            raise NotImplementedError
        else:
            loss = loss_a
        
        return loss