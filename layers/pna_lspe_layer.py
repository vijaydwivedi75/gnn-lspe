import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl

from .pna_utils import AGGREGATORS, SCALERS, MLP, FCLayer

"""
    PNALSPE: PNA with LSPE
    
    PNA: Principal Neighbourhood Aggregation 
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""


class PNATower(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d,
                 pretrans_layers, posttrans_layers, edge_features, edge_dim):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.aggregators = aggregators
        self.scalers = scalers
        self.pretrans_h = MLP(in_size=2 * 2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim,
                            out_size=in_dim, layers=pretrans_layers, mid_activation='relu', last_activation='none')
        self.pretrans_p = MLP(in_size=2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim,
                            out_size=in_dim, layers=pretrans_layers, mid_activation='tanh', last_activation='none')
        
        self.posttrans_h = MLP(in_size=(len(aggregators) * len(scalers) + 2) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.posttrans_p = MLP(in_size=(len(aggregators) * len(scalers) + 1) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='tanh', last_activation='none')
        
        self.avg_d = avg_d

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2_for_h = torch.cat([edges.src['h'], edges.dst['h'], edges.data['ef']], dim=1)
            z2_for_p = torch.cat([edges.src['p'], edges.dst['p'], edges.data['ef']], dim=1)
        else:
            z2_for_h = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            z2_for_p = torch.cat([edges.src['p'], edges.dst['p']], dim=1)

        return {'e_for_h': self.pretrans_h(z2_for_h), 'e_for_p': self.pretrans_p(z2_for_p)}

    # Message func for h
    def message_func_for_h(self, edges):
        return {'e_for_h': edges.data['e_for_h']}

    # Reduce func for h
    def reduce_func_for_h(self, nodes):
        h = nodes.mailbox['e_for_h']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'h': h}
    
    # Message func for p
    def message_func_for_p(self, edges):
        return {'e_for_p': edges.data['e_for_p']}
    
    # Reduce func for p
    def reduce_func_for_p(self, nodes):
        p = nodes.mailbox['e_for_p']
        D = p.shape[-2]
        p = torch.cat([aggregate(p) for aggregate in self.aggregators], dim=1)
        p = torch.cat([scale(p, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'p': p}

    def forward(self, g, h, p, e, snorm_n):
        g.ndata['h'] = h
        g.ndata['p'] = p
        
        if self.edge_features: # add the edges information only if edge_features = True
            g.edata['ef'] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)
        
        # aggregation for h
        g.update_all(self.message_func_for_h, self.reduce_func_for_h)
        h = torch.cat([h, g.ndata['h']], dim=1)
        
        # aggregation for p
        g.update_all(self.message_func_for_p, self.reduce_func_for_p)
        p = torch.cat([p, g.ndata['p']], dim=1)
        
        # posttransformation
        h = self.posttrans_h(h)
        p = self.posttrans_p(p)

        # graph and batch normalization
        if self.graph_norm:
            h = h * snorm_n
            
        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        h = F.dropout(h, self.dropout, training=self.training)
        p = F.dropout(p, self.dropout, training=self.training)
        
        return h, p


class PNALSPELayer(nn.Module):

    def __init__(self, in_dim, out_dim, aggregators, scalers, avg_d, dropout, graph_norm, batch_norm, towers=1,
                 pretrans_layers=1, posttrans_layers=1, divide_input=True, residual=False, edge_features=False,
                 edge_dim=0):
        """
        :param in_dim:              size of the input per node
        :param out_dim:             size of the output per node
        :param aggregators:         set of aggregation function identifiers
        :param scalers:             set of scaling functions identifiers
        :param avg_d:               average degree of nodes in the training set, used by scalers to normalize
        :param dropout:             dropout used
        :param graph_norm:          whether to use graph normalisation
        :param batch_norm:          whether to use batch normalisation
        :param towers:              number of towers to use
        :param pretrans_layers:     number of layers in the transformation before the aggregation
        :param posttrans_layers:    number of layers in the transformation after the aggregation
        :param divide_input:        whether the input features should be split between towers or not
        :param residual:            whether to add a residual connection
        :param edge_features:       whether to use the edge features
        :param edge_dim:            size of the edge features
        """
        super().__init__()
        assert ((not divide_input) or in_dim % towers == 0), "if divide_input is set the number of towers has to divide in_dim"
        assert (out_dim % towers == 0), "the number of towers has to divide the out_dim"
        assert avg_d is not None

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]

        self.divide_input = divide_input
        self.input_tower = in_dim // towers if divide_input else in_dim
        self.output_tower = out_dim // towers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False
            
        # convolution
        self.towers = nn.ModuleList()
        for _ in range(towers):
            self.towers.append(PNATower(in_dim=self.input_tower, out_dim=self.output_tower, aggregators=aggregators,
                                        scalers=scalers, avg_d=avg_d, pretrans_layers=pretrans_layers,
                                        posttrans_layers=posttrans_layers, batch_norm=batch_norm, dropout=dropout,
                                        graph_norm=graph_norm, edge_features=edge_features, edge_dim=edge_dim))
        # mixing network
        self.mixing_network_h = FCLayer(out_dim, out_dim, activation='LeakyReLU')
        self.mixing_network_p = FCLayer(out_dim, out_dim, activation='tanh')
    
    def forward(self, g, h, p, e, snorm_n):
        h_in = h  # for residual connection
        p_in = p  # for residual connection

        # Concating p to h, as in PEGNN
        h = torch.cat((h, p), -1)
        
        if self.divide_input:
            tower_outs = [tower(g, 
                                h[:, n_tower * 2 * self.input_tower: (n_tower + 1) * 2 * self.input_tower],
                                p[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower],
                                e,
                                snorm_n) for n_tower, tower in enumerate(self.towers)]
            h_tower_outs, p_tower_outs = map(list,zip(*tower_outs))
            h_cat = torch.cat(h_tower_outs, dim=1)
            p_cat = torch.cat(p_tower_outs, dim=1)
        else:
            tower_outs = [tower(g, h, p, e, snorm_n) for tower in self.towers]
            h_tower_outs, p_tower_outs = map(list,zip(*tower_outs))
            h_cat = torch.cat(h_tower_outs, dim=1)
            p_cat = torch.cat(p_tower_outs, dim=1)
                
        h_out = self.mixing_network_h(h_cat)
        p_out = self.mixing_network_p(p_cat)
        
        
        if self.residual:
            h_out = h_in + h_out  # residual connection
            p_out = p_in + p_out  # residual connection
        
        return h_out, p_out

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)


    
    
    
# This layer file below has no towers
# and is similar to DGNLayerComplex used for best PNA score on MOLPCBA
# implemented here https://github.com/Saro00/DGN/blob/master/models/dgl/dgn_layer.py

class PNANoTowersLSPELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d,
                 pretrans_layers, posttrans_layers, residual, edge_features, edge_dim=0, use_lapeig_loss=False):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False
            
        self.use_lapeig_loss = use_lapeig_loss

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        
        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]
        
        self.aggregators = aggregators
        self.scalers = scalers
        
        if self.edge_features:
            self.pretrans_h = MLP(in_size=2 * 2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim,
                                out_size=in_dim, layers=pretrans_layers, mid_activation='relu', last_activation='none')
            self.pretrans_p = MLP(in_size=2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim,
                                out_size=in_dim, layers=pretrans_layers, mid_activation='tanh', last_activation='none')

            self.posttrans_h = MLP(in_size=(len(aggregators) * len(scalers) + 2) * in_dim, hidden_size=out_dim,
                                 out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
            self.posttrans_p = MLP(in_size=(len(aggregators) * len(scalers) + 1) * in_dim, hidden_size=out_dim,
                                 out_size=out_dim, layers=posttrans_layers, mid_activation='tanh', last_activation='none')
        else:
            self.posttrans_h = MLP(in_size=(len(aggregators) * len(scalers)) * 2 * in_dim, hidden_size=out_dim,
                                 out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
            self.posttrans_p = MLP(in_size=(len(aggregators) * len(scalers)) * in_dim, hidden_size=out_dim,
                                 out_size=out_dim, layers=posttrans_layers, mid_activation='tanh', last_activation='none')
        
        self.avg_d = avg_d

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2_for_h = torch.cat([edges.src['h'], edges.dst['h'], edges.data['ef']], dim=1)
            z2_for_p = torch.cat([edges.src['p'], edges.dst['p'], edges.data['ef']], dim=1)
        else:
            z2_for_h = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            z2_for_p = torch.cat([edges.src['p'], edges.dst['p']], dim=1)

        return {'e_for_h': self.pretrans_h(z2_for_h), 'e_for_p': self.pretrans_p(z2_for_p)}

    # Message func for h
    def message_func_for_h(self, edges):
        return {'e_for_h': edges.data['e_for_h']}

    # Reduce func for h
    def reduce_func_for_h(self, nodes):
        if self.edge_features:
            h = nodes.mailbox['e_for_h']
        else:
            h = nodes.mailbox['m_h']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'h': h}
    
    # Message func for p
    def message_func_for_p(self, edges):
        return {'e_for_p': edges.data['e_for_p']}
    
    # Reduce func for p
    def reduce_func_for_p(self, nodes):
        if self.edge_features:
            p = nodes.mailbox['e_for_p']
        else:
            p = nodes.mailbox['m_p']
        D = p.shape[-2]
        p = torch.cat([aggregate(p) for aggregate in self.aggregators], dim=1)
        if len(self.scalers) > 1:
            p = torch.cat([scale(p, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'p': p}

    def forward(self, g, h, p, e, snorm_n):
        
        h = F.dropout(h, self.dropout, training=self.training)
        p = F.dropout(p, self.dropout, training=self.training)
        
        h_in = h  # for residual connection
        p_in = p  # for residual connection
        
        # Concating p to h, as in PEGNN
        h = torch.cat((h, p), -1)
        
        g.ndata['h'] = h
        g.ndata['p'] = p
        
        if self.edge_features: # add the edges information only if edge_features = True
            g.edata['ef'] = e

        if self.edge_features:
            # pretransformation
            g.apply_edges(self.pretrans_edges)
        
        if self.edge_features:
            # aggregation for h
            g.update_all(self.message_func_for_h, self.reduce_func_for_h)
            h = torch.cat([h, g.ndata['h']], dim=1)

            # aggregation for p
            g.update_all(self.message_func_for_p, self.reduce_func_for_p)
            p = torch.cat([p, g.ndata['p']], dim=1)
        else:
            # aggregation for h
            g.update_all(fn.copy_u('h', 'm_h'), self.reduce_func_for_h)
            h = g.ndata['h']
            
            # aggregation for p
            g.update_all(fn.copy_u('p', 'm_p'), self.reduce_func_for_p)
            p = g.ndata['p']
        
        # posttransformation
        h = self.posttrans_h(h)
        p = self.posttrans_p(p)

        # graph and batch normalization
        if self.graph_norm and self.edge_features:
            h = h * snorm_n
            
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.relu(h)
        p = torch.tanh(h)
        
        if self.residual:
            h = h_in + h  # residual connection
            p = p_in + p  # residual connection
        
        return h, p

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)
