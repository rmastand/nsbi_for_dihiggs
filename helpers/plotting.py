import numpy as np
import matplotlib.pyplot as plt

def plot_features(data_sets, data_labels, axis_labels, nb, weights_list = None, kwargs_list = None):
    
    n_features = data_sets[0].shape[1]
    
    bins = [nb for i in range(n_features)]
    if weights_list is None:
        weights_list = [np.ones((d.shape[0], 1)) for d in data_sets]
    if kwargs_list is None:
        kwargs_list = [{"histtype": "step", "lw": 2} for d in data_sets]
        
    if n_features > 1:

        fig, ax = plt.subplots(1, n_features, figsize = (4*n_features, 4), squeeze = True)

        for i in range(n_features):

            for loc_id, loc_data in enumerate(data_sets):

                ax[i].hist(loc_data[:,i], bins = bins[i], density = True, weights = weights_list[loc_id], label = data_labels[loc_id], **kwargs_list[loc_id])

            ax[i].set_xlabel(axis_labels[i], fontsize = 18)
            ax[i].set_yticks([])

        ax[0].set_ylabel("Density", fontsize = 18)

        ax[0].legend()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        
    else:
        
        fig, ax = plt.subplots(1, 1, figsize = (4, 4), squeeze = True)

        for loc_id, loc_data in enumerate(data_sets):

            ax.hist(loc_data, bins = bins[0], density = True, weights = weights_list[loc_id], label = data_labels[loc_id], **kwargs_list[loc_id])

            ax.set_xlabel(axis_labels[0], fontsize = 18)
            ax.set_yticks([])

        ax.set_ylabel("Density", fontsize = 18)

        ax.legend()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
