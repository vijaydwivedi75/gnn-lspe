import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    GraphiT-GT-LSPE: GraphiT-GT with LSPE
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func

def adaptive_edge_PE(field, adaptive_weight):
    def func(edges):
        # initial shape was: adaptive_weight: [edges,1]; data: [edges, num_heads, 1]
        # repeating adaptive_weight to have: [edges, num_heads, 1]
        edges.data['tmp'] = edges.data[adaptive_weight].repeat(1, edges.data[field].shape[1]).unsqueeze(-1)
        return {'score_soft': edges.data['tmp'] * edges.data[field]}
    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, use_bias, adaptive_edge_PE, attention_for):
        super().__init__()
        
       
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph=full_graph
        self.attention_for = attention_for
        self.adaptive_edge_PE = adaptive_edge_PE
        
        if self.attention_for == "h": # attention module for h has input h = [h,p], so 2*in_dim for Q,K,V
            if use_bias:
                self.Q = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)
                self.K = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)
                    self.K_2 = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)
                    self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                self.V = nn.Linear(in_dim*2, out_dim * num_heads, bias=True)

            else:
                self.Q = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
                self.K = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
                    self.K_2 = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
                    self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                self.V = nn.Linear(in_dim*2, out_dim * num_heads, bias=False)
        
        elif self.attention_for == "p": # attention module for p
            if use_bias:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                    self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                    self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

            else:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                    self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                    self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    
    def propagate_attention(self, g):

        
        if self.full_graph:
            real_ids = torch.nonzero(g.edata['real']).squeeze()
            fake_ids = torch.nonzero(g.edata['real']==0).squeeze()

        else:
            real_ids = g.edges(form='eid')
            
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)
        
        if self.full_graph:
            g.apply_edges(src_dot_dst('K_2h', 'Q_2h', 'score'), edges=fake_ids)
        

        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores for edges
        g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)
        
        if self.full_graph:
            g.apply_edges(imp_exp_attn('score', 'E_2'), edges=fake_ids)
    
        g.apply_edges(exp('score'))
        
        # Adaptive weighting with k_RW_eij
        # Only applicable to full graph, For NOW
        if self.adaptive_edge_PE and self.full_graph:
            g.apply_edges(adaptive_edge_PE('score_soft', 'k_RW'))
        del g.edata['tmp']
        
        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))
    
    
    def forward(self, g, h, p, e):
        if self.attention_for == "h":
            h = torch.cat((h, p), -1)
        elif self.attention_for == "p":
            h = p
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(e)
        
        if self.full_graph:
            Q_2h = self.Q_2(h)
            K_2h = self.K_2(h)
            E_2 = self.E_2(e)
            
        V_h = self.V(h)

        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.edata['E'] = E.view(-1, self.num_heads, self.out_dim)
        
        
        if self.full_graph:
            g.ndata['Q_2h'] = Q_2h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_2h'] = K_2h.view(-1, self.num_heads, self.out_dim)
            g.edata['E_2'] = E_2.view(-1, self.num_heads, self.out_dim)
        
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        
        del g.ndata['wV']
        del g.ndata['z']
        del g.ndata['Q_h']
        del g.ndata['K_h']
        del g.edata['E']
        
        if self.full_graph:
            del g.ndata['Q_2h']
            del g.ndata['K_2h']
            del g.edata['E_2']
        
        return h_out
    

class GraphiT_GT_LSPE_Layer(nn.Module):
    """
        Param: 
    """
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, dropout=0.0,
                 layer_norm=False, batch_norm=True, residual=True, adaptive_edge_PE=False, use_bias=False):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention_h = MultiHeadAttentionLayer(gamma, in_dim, out_dim//num_heads, num_heads,
                                                   full_graph, use_bias, adaptive_edge_PE, attention_for="h")
        self.attention_p = MultiHeadAttentionLayer(gamma, in_dim, out_dim//num_heads, num_heads,
                                                   full_graph, use_bias, adaptive_edge_PE, attention_for="p")
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_p = nn.Linear(out_dim, out_dim)
        
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            
        
    def forward(self, g, h, p, e, snorm_n):
        h_in1 = h # for first residual connection
        p_in1 = p # for first residual connection
        
        # [START] For calculation of h -----------------------------------------------------------------
        
        # multi-head attention out
        h_attn_out = self.attention_h(g, h, p, e)
        
        #Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)
       
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h # residual connection
            
        # # GN from benchmarking-gnns-v1
        # h = h * snorm_n

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection       

        # # GN from benchmarking-gnns-v1
        # h = h * snorm_n
        
        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)         
        
        # [END] For calculation of h -----------------------------------------------------------------
        
        
        # [START] For calculation of p -----------------------------------------------------------------
        
        # multi-head attention out
        p_attn_out = self.attention_p(g, None, p, e)
        
        #Concat multi-head outputs
        p = p_attn_out.view(-1, self.out_channels)
       
        p = F.dropout(p, self.dropout, training=self.training)

        p = self.O_p(p)
        
        p = torch.tanh(p)
        
        if self.residual:
            p = p_in1 + p # residual connection

        # [END] For calculation of p -----------------------------------------------------------------

        return h, p
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)