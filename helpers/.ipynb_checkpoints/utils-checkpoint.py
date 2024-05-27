"""
NOT MY CODE -- minimally adapted from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
"""

import torch
import numpy as np


def np_to_torch(array, device):
    return torch.tensor(array.astype(np.float32)).to(device)


def crop_feature(array, feature_index, low_crop, high_crop, arrays_to_crop):
    
    selected_ind = (array[:, feature_index] >= low_crop) & (array[:, feature_index] < high_crop)
    cropped_arrays = []
    
    for arr in arrays_to_crop:
        cropped_arrays.append(arr[selected_ind])

    return cropped_arrays



class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
     
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, verbose = False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
    def __call__(self, val_loss):
        
        if self.best_loss == None:
            self.best_loss = val_loss
        
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            #print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print('INFO: Early stopping')
                self.early_stop = True