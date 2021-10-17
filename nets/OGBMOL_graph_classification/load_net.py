"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.OGBMOL_graph_classification.gatedgcn_net import GatedGCNNet
from nets.OGBMOL_graph_classification.pna_net import PNANet
from nets.OGBMOL_graph_classification.san_net import SANNet
from nets.OGBMOL_graph_classification.graphit_net import GraphiTNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def PNA(net_params):
    return PNANet(net_params)

def SAN(net_params):
    return SANNet(net_params)

def GraphiT(net_params):
    return GraphiTNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'PNA': PNA,
        'SAN': SAN,
        'GraphiT': GraphiT
    }
        
    return models[MODEL_NAME](net_params)