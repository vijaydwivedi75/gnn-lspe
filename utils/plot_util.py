""" 
    Util function to plot graph with eigenvectors
        x-axis: first dim
        y-axis: second dim
"""

import networkx as nx

def plot_graph_eigvec(plt, g_id, g_dgl, feature_key, actual_eigvecs=False, predicted_eigvecs=False):
    
    if actual_eigvecs:
        plt.set_xlabel('first eigenvec')
        plt.set_ylabel('second eigenvec')
    else:
        plt.set_xlabel('first predicted pe')
        plt.set_ylabel('second predicted pe')
    
    g_dgl = g_dgl.cpu()
    g_dgl.ndata['feats'] = g_dgl.ndata[feature_key][:,:2]
    g_nx = g_dgl.to_networkx(node_attrs=['feats'])

    labels = {} 
    for idx, node in enumerate(g_nx.nodes()): 
        labels[node] = str(idx)
    
    num_nodes = g_dgl.num_nodes()
    num_edges = g_dgl.num_edges()

    edge_list = []
    srcs, dsts = g_dgl.edges()
    for edge_i in range(num_edges):
        edge_list.append((srcs[edge_i].item(), dsts[edge_i].item()))

    # fig, ax = plt.subplots()
    # first 2-dim of eigenvecs are x,y coordinates, and the 3rd dim of eigenvec is plotted as node intensity
    # intensities = g_dgl.ndata['feats'][:,2]
    nx.draw_networkx_nodes(g_nx, g_dgl.ndata['feats'][:,:2].numpy(), node_color='r', node_size=180, label=list(range(g_dgl.number_of_nodes())))
    nx.draw_networkx_edges(g_nx, g_dgl.ndata['feats'][:,:2].numpy(), edge_list, alpha=0.3)
    nx.draw_networkx_labels(g_nx, g_dgl.ndata['feats'][:,:2].numpy(), labels, font_size=16)
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    title = "Graph ID: " + str(g_id)
    
    title += " | Actual eigvecs" if actual_eigvecs else " | Predicted PEs"
    plt.title.set_text(title)
    