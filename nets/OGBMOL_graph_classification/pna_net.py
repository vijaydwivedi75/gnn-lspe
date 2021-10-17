import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm 

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

"""
    PNA-LSPE
    
"""

from layers.pna_layer import PNANoTowersLayer as PNALayer
from layers.pna_lspe_layer import PNANoTowersLSPELayer as PNALSPELayer
from layers.mlp_readout_layer import MLPReadout2 as MLPReadout


class PNANet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.dropout_2 = net_params['dropout_2']
        
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.gru_enable = net_params['gru']
        device = net_params['device']
        self.device = device
        self.pe_init = net_params['pe_init']
        self.n_classes = net_params['n_classes']
    
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        
        self.pos_enc_dim = net_params['pos_enc_dim']
        
        if self.pe_init in ['rand_walk']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)
        
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        
        if self.edge_feat:
            self.embedding_e = BondEncoder(emb_dim=edge_dim)

        
        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList(
                [PNALSPELayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                          batch_norm=self.batch_norm, aggregators=self.aggregators, scalers=self.scalers, avg_d=self.avg_d,
                          pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers, residual=self.residual,
                          edge_features=self.edge_feat, edge_dim=edge_dim, use_lapeig_loss=self.use_lapeig_loss)
                 for _ in range(n_layers - 1)])
            self.layers.append(PNALSPELayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout, graph_norm=self.graph_norm,
                          batch_norm=self.batch_norm, aggregators=self.aggregators, scalers=self.scalers, avg_d=self.avg_d,
                                        pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers, 
                                        residual=self.residual, edge_features=self.edge_feat, edge_dim=edge_dim, use_lapeig_loss=self.use_lapeig_loss))

        else:
            # NoPE
            self.layers = nn.ModuleList(
                [PNALayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                          batch_norm=self.batch_norm, aggregators=self.aggregators, scalers=self.scalers, avg_d=self.avg_d,
                          pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers, residual=self.residual,
                          edge_features=self.edge_feat, edge_dim=edge_dim, use_lapeig_loss=self.use_lapeig_loss)
                 for _ in range(n_layers - 1)])
            self.layers.append(PNALayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout, graph_norm=self.graph_norm,
                          batch_norm=self.batch_norm, aggregators=self.aggregators, scalers=self.scalers, avg_d=self.avg_d,
                                        pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers, 
                                        residual=self.residual, edge_features=self.edge_feat, edge_dim=edge_dim, use_lapeig_loss=self.use_lapeig_loss))
            
        if self.gru_enable:
            self.gru = GRU(hidden_dim, hidden_dim, device)
        
        self.MLP_layer = MLPReadout(out_dim, n_classes, self.dropout_2)

        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(out_dim+self.pos_enc_dim, out_dim)
        
        self.g = None              # For util; To be accessed in loss() function
        
    def forward(self, g, h, p, e, snorm_n):
        
        h = self.embedding_h(h)
        
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
            p = F.dropout(p, self.dropout_2, training=self.training)
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
            hp = torch.cat((g.ndata['h'],g.ndata['p']),dim=-1)
            hp = F.dropout(hp, self.dropout_2, training=self.training)
            hp = self.Whp(hp)
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
            # Loss B: Laplacian Eigenvector Loss --------------------------------------------
            g = self.g
            n = g.number_of_nodes()

            # Laplacian 
            A = g.adjacency_matrix(scipy_fmt="csr")
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(n) - N * A * N

            p = g.ndata['p']
            pT = torch.transpose(p, 1, 0)
            loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(self.device)), p))

            # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
            bg = dgl.unbatch(g)
            batch_size = len(bg)
            P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
            PTP_In = P.T * P - sp.eye(P.shape[1])
            loss_b_2 = torch.tensor(norm(PTP_In, 'fro')**2).float().to(self.device)

            loss_b = ( loss_b_1 + self.lambda_loss * loss_b_2 ) / ( self.pos_enc_dim * batch_size * n) 

            del bg, P, PTP_In, loss_b_1, loss_b_2

            loss = loss_a + self.alpha_loss * loss_b
        else:
            loss = loss_a
            
        return loss