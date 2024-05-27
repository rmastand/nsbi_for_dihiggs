import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# alternative architectures
import torchbnn as bnn

from tqdm import tqdm
from livelossplot import PlotLosses
from helpers.utils import EarlyStopping, LRScheduler
import matplotlib.pyplot as plt


"""
NETWORK ARCHITECTURES
"""

class NeuralNet(nn.Module):
    def __init__(self, n_inputs, layers = [32, 32]):
        super(NeuralNet, self).__init__()
            
        # assemble the network
        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            n_inputs = nodes
            
        self.layers.append(nn.Linear(n_inputs, 1))
        self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_stack(x)
    
    
class BNN(nn.Module):
    def __init__(self, n_inputs, layers = [32, 32], prior_mu = 0, prior_sigma = 0.1, activation = "relu"):
        super(BNN, self).__init__()
        
        if activation == "relu":
            activation_func = nn.ReLU()
        elif activation == "tanh":
            activation_func = nn.Tanh()
            
        # assemble the network
        self.layers = []
        for nodes in layers:
            self.layers.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=n_inputs, out_features=nodes))
            self.layers.append(activation_func)
            n_inputs = nodes
        
        self.layers.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=n_inputs, out_features=1))
        self.layers.append(nn.Sigmoid())
        #self.layers.append(nn.ReLU())
        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_stack(x)
    
    
    

    
""" 
TRAINING FUNCTIONS
"""

eps=1e-10
       
    
def compute_loss_1(network, batch_data, batch_labels, train_bnn, loss_type):
    
    if loss_type == "BCE":
        loss_bce = F.binary_cross_entropy(network(batch_data), batch_labels)
    else:
        print("SOMETHING VERY WRONG")
        return None
    
    if train_bnn:
        kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
        loss_kl = kl_loss(network)[0]
    else: 
        loss_kl = 0
        
    return loss_bce, loss_kl
        

    
    
def train_network(X_train, Y_train, X_val, Y_val, network, optimizer, n_epochs, batch_size, device, seed=0, use_lr_scheduler=True, patience_lr = 5, use_early_stop=True, patience_ES = 5, train_bnn=False, kl_weight=0.01, network_id="", min_delta=0, loss_type = "BCE"):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_set = torch.utils.data.TensorDataset(X_train, Y_train)
    val_set = torch.utils.data.TensorDataset(X_val, Y_val)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False)

    network.to(device)
    
    epochs, losses, losses_val = [], [], []
    # save the best model
    val_loss_to_beat = 10000
    best_epoch = -1
    
    if use_lr_scheduler: 
        lr_scheduler = LRScheduler(optimizer, patience=patience_lr)
    if use_early_stop:
        early_stopping = EarlyStopping(patience=patience_ES, min_delta=min_delta)
        
    if train_bnn:
        print(f"Using KL weight of {kl_weight}.")
    print(f"Using loss type {loss_type}.")
        
    #liveloss = PlotLosses()

    for epoch in tqdm(range(n_epochs)):
        logs = {}
        losses_batch_per_e = []
        # batching    
        for batch_index, (batch_data, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            loss_bce, loss_kl = compute_loss_1(network, batch_data, batch_labels, train_bnn, loss_type)
            loss_total = loss_bce + kl_weight*loss_kl
   
            loss_total.backward()
            optimizer.step()
            losses_batch_per_e.append(loss_total.detach().cpu().numpy())
            
        epochs.append(epoch)
        losses.append(np.mean(losses_batch_per_e))
        logs["loss"] = np.mean(losses_batch_per_e)
        

        # validation
        with torch.no_grad():
            val_losses_batch_per_e = []
            for batch_index, (batch_data, batch_labels) in enumerate(val_loader):
                
                val_loss_bce, val_loss_kl = compute_loss_1(network, batch_data, batch_labels, train_bnn, loss_type)
                val_loss_total = val_loss_bce + kl_weight*val_loss_kl
            
                val_losses_batch_per_e.append(val_loss_total.detach().cpu().numpy())
                val_epoch_loss = np.mean(val_losses_batch_per_e)
            losses_val.append(val_epoch_loss)
            logs["val_loss"] = val_epoch_loss
            
            if val_epoch_loss < val_loss_to_beat:
                val_loss_to_beat = val_epoch_loss
                # save the model
                model_path = f"{network_id}_best_model.pt"
                torch.save({
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    }, model_path)
                best_epoch = epoch
                
            if use_lr_scheduler:
                lr_scheduler(val_epoch_loss)
                
            if use_early_stop:
                early_stopping(val_epoch_loss)
                
        #liveloss.update(logs)
        #liveloss.send()
        
        if use_early_stop:
            if early_stopping.early_stop:
                print("Stopping network training")
                break
                
    print("Done training!")
    print(f"Best epoch: {best_epoch}")
    
    """
    plt.figure()
    plt.plot(epochs, losses, label = "train")
    plt.plot(epochs, losses_val, label = "val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(network_id)
    plt.close()
    """
    return epochs, losses, losses_val



