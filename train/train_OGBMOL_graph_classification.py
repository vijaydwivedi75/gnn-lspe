"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm

import dgl


def train_epoch_sparse(model, optimizer, device, data_loader, epoch, evaluator):
    model.train()
    
    epoch_loss = 0
    nb_data = 0
    
    y_true = []
    y_pred = []

    for iter, (batch_graphs, batch_labels, batch_snorm_n) in enumerate(data_loader):
        optimizer.zero_grad()

        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
        except KeyError:
            batch_pos_enc = None
        
        if model.pe_init == 'lap_pe':
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            
        batch_pred, __ = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
        del __
        
        # ignore nan labels (unlabeled) when computing training loss
        is_labeled = batch_labels == batch_labels
        loss = model.loss(batch_pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
        
        loss.backward()
        optimizer.step()
        
        y_true.append(batch_labels.view(batch_pred.shape).detach().cpu())
        y_pred.append(batch_pred.detach().cpu())
        
        epoch_loss += loss.detach().item()
        nb_data += batch_labels.size(0)
    
    epoch_loss /= (iter + 1)
    
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    # compute performance metric using OGB evaluator
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    perf = evaluator.eval(input_dict)
        
    if batch_labels.size(1) == 128: # MOLPCBA
        return_perf = perf['ap']
    elif batch_labels.size(1) == 12: # MOLTOX21
        return_perf = perf['rocauc']
    
    return epoch_loss, return_perf, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch, evaluator):
    model.eval()

    epoch_loss = 0
    nb_data = 0

    y_true = []
    y_pred = []
    
    out_graphs_for_lapeig_viz = []
    
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            
            
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            except KeyError:
                batch_pos_enc = None
            
            batch_pred, batch_g = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)

            
            # ignore nan labels (unlabeled) when computing loss
            is_labeled = batch_labels == batch_labels
            loss = model.loss(batch_pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
            
            y_true.append(batch_labels.view(batch_pred.shape).detach().cpu())
            y_pred.append(batch_pred.detach().cpu())

            epoch_loss += loss.detach().item()
            nb_data += batch_labels.size(0)
            
            if batch_g is not None:
                out_graphs_for_lapeig_viz += dgl.unbatch(batch_g)
            else:
                out_graphs_for_lapeig_viz = None
            
        epoch_loss /= (iter + 1)

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    # compute performance metric using OGB evaluator
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    perf = evaluator.eval(input_dict)
        
    if batch_labels.size(1) == 128: # MOLPCBA
        return_perf = perf['ap']
    elif batch_labels.size(1) == 12: # MOLTOX21
        return_perf = perf['rocauc']
        
    return epoch_loss, return_perf, out_graphs_for_lapeig_viz