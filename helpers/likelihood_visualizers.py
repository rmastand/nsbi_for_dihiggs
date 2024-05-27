import numpy as np
import matplotlib.pyplot as plt
import matplotlib


from helpers.test_statistics import *


def make_c_grid(num_points, c_ranges_lower, c_ranges_upper, inds_string):
    
    # inds_string is a string like "1" or "012"

    c_scans = {i: np.linspace(c_ranges_lower[i], c_ranges_upper[i], num_points) for i in range(3)}
    
    c0_to_scan = c_scans[0] if "0" in inds_string else 0
    c1_to_scan = c_scans[1] if "1" in inds_string else 0
    c2_to_scan = c_scans[2] if "2" in inds_string else 0
    
    c0, c1, c2 = np.meshgrid(c0_to_scan, c1_to_scan, c2_to_scan)
    c_grid =  np.vstack((c0.flatten(), c1.flatten(), c2.flatten())).T
    bin_sizes = [c_scans[i][1] - c_scans[i][0] for i in range(3)]
    
    edges = {i: np.linspace(c_scans[i][0] - bin_sizes[i] / 2, c_scans[i][-1] + bin_sizes[i] / 2, num_points + 1,) for i in range(3) }
    
    return c_grid, c_scans, edges


def get_coords_in_3d(target_loc, c_scans):
    
    indices = []
    for i in range(3):
        indices.append(np.where(c_scans[i] == target_loc[i])[0][0])
    return indices



def c_scan_1d(trained_network_dict, device, test_features, N_sig_c_scan, N_sig_SM_target, N_bkg_SM_target, c_grid, c_scans, index, seeds_to_ensemble, q_rate=None, network_type="dense"):
    
    # evaluate the shape test statistic
    #q_shape = eval_test_statistic_shape(test_features, c_grid, trained_network, device)
    #q_shape = eval_test_statistic_shape_mixture_model_04_05(test_features, c_grid, trained_network_dict["Ssm_Salt"], trained_network_dict["B_Salt"], trained_network_dict["Ssm_B"], N_sig_c_scan, N_sig_SM_target, N_bkg_SM_target, device, network_type=network_type)
    q_shape = eval_test_statistic_shape_mixture_model_4_comp_ensemble(test_features, c_grid, trained_network_dict, seeds_to_ensemble, N_sig_c_scan, N_sig_SM_target, N_bkg_SM_target, device, network_type = network_type)
    # get the maximum
    max_index_shape = np.argmax(q_shape)
    best_fit_shape = c_grid[max_index_shape]
    
    if q_rate is not None:
        q_total = q_rate + q_shape
        max_index_total = np.argmax(q_total)
        best_fit_total = c_grid[max_index_total]
    
    # plot results
    fig, ax = plt.subplots(1, 1, figsize = (4,4))
    
    ax.scatter(c_scans[int(index)], q_shape, color = "red", label = "shape only")
    ax.axvline(best_fit_shape[int(index)], color = "red")
    if q_rate is not None:
        ax.scatter(c_scans[int(index)], q_rate, color = "green", label = "rate only")
        
        ax.scatter(c_scans[int(index)], q_total, color = "blue", label = "shape+rate")
        ax.axvline(best_fit_total[int(index)], color = "blue")
        
    ax.set_xlabel(f"$c_{index}$")
    ax.set_ylabel("$q$")
    
    plt.legend()
    
    plt.show()
    
    if q_rate is not None:
        return q_total
    else:
        return q_shape
    
    
def c_scan_2d(trained_network_dict, device, test_features, N_sig_c_scan, N_sig_SM_target, N_bkg_SM_target, c_grid, c_scans, edges, indices, seeds_to_ensemble, target=[0,0,0], q_rate=None, network_type="dense"):
        
    # evaluate the shape test statistic
    #q_shape = eval_test_statistic_shape_mixture_model_04_05(test_features, c_grid, trained_network_dict["Ssm_Salt"], trained_network_dict["B_Salt"], trained_network_dict["Ssm_B"], N_sig_c_scan, N_sig_SM_target, N_bkg_SM_target, device)
    q_shape = eval_test_statistic_shape_mixture_model_4_comp_ensemble(test_features, c_grid, trained_network_dict, seeds_to_ensemble, N_sig_c_scan, N_sig_SM_target, N_bkg_SM_target, device, network_type = network_type)
    
    
    if q_rate is not None:
        q_total = q_rate + q_shape
        max_index_total = np.argmax(q_total)
        best_fit_total = c_grid[max_index_total]
        q_to_plot = q_total
    else:
         q_to_plot = q_shape
                     
    # get the maximum
    max_index = np.argmax(q_to_plot)
    best_fit = c_grid[max_index]
    
    max_val = np.max(q_to_plot)
    min_val = np.min(q_to_plot)

    #print(f"Max llr of {np.max(llr_eval_grid)} at theta = {best_fit}")

    # plot results

    # reshape the llr
    num_points = len(c_scans[0])
    q_to_plot = np.array(q_to_plot).reshape((num_points,num_points))
    
    if indices == "01":
        index0 = np.where(c_scans[0] == best_fit[0])[0][0]
        index1 = np.where(c_scans[1] == best_fit[1])[0][0]
        marginal0 = q_to_plot[index1,:]
        marginal1 = q_to_plot[:,index0]
        plot_2d_slice(q_to_plot, 0, 1, edges, c_scans, best_fit, marginal0, marginal1, cmap="gnuplot", target=target)
    elif indices == "02":
        index0 = np.where(c_scans[0] == best_fit[0])[0][0]
        index2 = np.where(c_scans[2] == best_fit[2])[0][0]
        marginal0 = q_to_plot[index0,:]
        marginal1 = q_to_plot[:,index2]
        plot_2d_slice(q_to_plot, 2, 0, edges, c_scans, best_fit, marginal0, marginal1, cmap="gnuplot", target=target)
    elif indices == "12":
        index1 = np.where(c_scans[1] == best_fit[1])[0][0]
        index2 = np.where(c_scans[2] == best_fit[2])[0][0]
        marginal0 = q_to_plot[:,index2]
        marginal1 = q_to_plot[index1,:]
        plot_2d_slice(q_to_plot.T, 1, 2, edges, c_scans, best_fit, marginal0, marginal1, cmap="gnuplot", target=target)
    
    return q_to_plot
    

def c_scan_1d_old(trained_network_dict, device, test_features, N_sig_c_scan, N_bkg_SM_target, c_grid, c_scans, index, q_rate=None ):
    
    # evaluate the shape test statistic
    #q_shape = eval_test_statistic_shape(test_features, c_grid, trained_network, device)
    q_shape = eval_test_statistic_shape_mixture_model(test_features, c_grid, trained_network_dict["SM_Salt"], trained_network_dict["SM_B"], N_sig_c_scan, N_bkg_SM_target, device)
    
    # get the maximum
    max_index_shape = np.argmax(q_shape)
    best_fit_shape = c_grid[max_index_shape]
    
    if q_rate is not None:
        q_total = q_rate + q_shape
        max_index_total = np.argmax(q_total)
        best_fit_total = c_grid[max_index_total]
    
    # plot results
    fig, ax = plt.subplots(1, 1, figsize = (4,4))
    
    ax.scatter(c_scans[int(index)], q_shape, color = "red", label = "shape only")
    ax.axvline(best_fit_shape[int(index)], color = "red")
    if q_rate is not None:
        ax.scatter(c_scans[int(index)], q_rate, color = "green", label = "rate only")
        
        ax.scatter(c_scans[int(index)], q_total, color = "blue", label = "shape+rate")
        ax.axvline(best_fit_total[int(index)], color = "blue")
        
    ax.set_xlabel(f"$c_{index}$")
    ax.set_ylabel("$q$")
    
    plt.legend()
    
    plt.show()
    
    if q_rate is not None:
        return q_total
    else:
        return q_shape
    
    
    
    
def c_scan_2d_old(trained_network, device, test_features, c_grid, c_scans, edges, indices, target=[0,0,0], q_rate=None):
        
    # evaluate the shape test statistic
    q_shape = eval_test_statistic_shape(test_features, c_grid, trained_network, device)
    
    if q_rate is not None:
        q_total = q_rate + q_shape
        max_index_total = np.argmax(q_total)
        best_fit_total = c_grid[max_index_total]
        q_to_plot = q_total
    else:
         q_to_plot = q_shape
                     
    # get the maximum
    max_index = np.argmax(q_to_plot)
    best_fit = c_grid[max_index]
    
    max_val = np.max(q_to_plot)
    min_val = np.min(q_to_plot)

    #print(f"Max llr of {np.max(llr_eval_grid)} at theta = {best_fit}")

    # plot results

    # reshape the llr
    num_points = len(c_scans[0])
    q_to_plot = np.array(q_to_plot).reshape((num_points,num_points))
    
    if indices == "01":
        index0 = np.where(c_scans[0] == best_fit[0])[0][0]
        index1 = np.where(c_scans[1] == best_fit[1])[0][0]
        marginal0 = q_to_plot[index1,:]
        marginal1 = q_to_plot[:,index0]
        plot_2d_slice(q_to_plot, 0, 1, edges, c_scans, best_fit, marginal0, marginal1, cmap="gnuplot", target=target)
    elif indices == "02":
        index0 = np.where(c_scans[0] == best_fit[0])[0][0]
        index2 = np.where(c_scans[2] == best_fit[2])[0][0]
        marginal0 = q_to_plot[index0,:]
        marginal1 = q_to_plot[:,index2]
        plot_2d_slice(q_to_plot, 2, 0, edges, c_scans, best_fit, marginal0, marginal1, cmap="gnuplot", target=target)
    elif indices == "12":
        index1 = np.where(c_scans[1] == best_fit[1])[0][0]
        index2 = np.where(c_scans[2] == best_fit[2])[0][0]
        marginal0 = q_to_plot[:,index2]
        marginal1 = q_to_plot[index1,:]
        plot_2d_slice(q_to_plot.T, 1, 2, edges, c_scans, best_fit, marginal0, marginal1, cmap="gnuplot", target=target)
    
    return q_to_plot



def c_scan_3d_with_confusion(trained_network, device, test_features, c_grid, c_scans, edges, target=[0,0,0], q_rate=None):
        
    # evaluate the shape test statistic
    q_shape = eval_test_statistic_shape(test_features, c_grid, trained_network, device)
    
    if q_rate is not None:
        q_total = q_rate + q_shape
        max_index_total = np.argmax(q_total)
        best_fit_total = c_grid[max_index_total]
        q_to_plot = q_total
    else:
         q_to_plot = q_shape
                     
    # get the maximum
    max_index = np.argmax(q_to_plot)
    best_fit = c_grid[max_index]
    
    max_val = np.max(q_to_plot)
    min_val = np.min(q_to_plot)
    
    # also get the theta corresponding to the target
    extended_target = np.repeat(np.array(target).reshape(-1, 1), num_points**3, axis = 1).T
    distances = np.sum((extended_target - c_grid)**2, axis = 1)
    target_index = np.argmin(distances)
    target_fit = c_grid[target_index]

    print(f"Max llr of {np.round(np.max(llr_eval_grid), 4)} at theta = {best_fit}")
    print("llr = {val} at theta = {tt}".format(val = np.round(llr_eval_grid[target_index], 4), tt = target_fit))

    index0, index1, index2 = get_coords_in_3d(best_fit, c_scans)
    index0_target, index1_target, index2_target = get_coords_in_3d(target_fit, c_scans)

    # reshape the q
    num_points = len(c_scans[0])
    q_to_plot = np.array(q_to_plot).reshape((num_points,num_points,num_points))
    
    # plotting for the recovered theta
    print("2D slices for network estimation of theta")
    plot_2d_slices(q_to_plot, edges, best_fit, index0, index1, index2, cmap="gnuplot", target=target)
    
    # Plot 1-D slices
    marginal0 = q_to_plot[index1,:,index2]
    marginal1 = q_to_plot[:,index0,index2]
    marginal2 = q_to_plot[index1,index0,:]

    fig, ax = plt.subplots(1, 3, figsize = (12,3))

    ax[0].scatter(c_scans[0], marginal0)
    ax[0].set_xlabel("c0")
    ax[0].axvline(best_fit[0], color = "red")
    ax[0].set_ylim(min_val, max_val)

    ax[1].scatter(c_scans[1], marginal1)
    ax[1].set_xlabel("c1")
    ax[1].axvline(best_fit[1], color = "red")
    ax[1].set_ylim(min_val, max_val)

    ax[2].scatter(c_scans[2], marginal2)
    ax[2].set_xlabel("c2")
    ax[2].axvline(best_fit[2], color = "red")
    ax[2].set_ylim(min_val, max_val)

    plt.show()
    
    # plotting for the true target theta
    print(3*"\n")
    print("2D slices at the target theta")
    plot_2d_slices(q_to_plot, edges, target_fit, index0_target, index1_target, index2_target, cmap="gnuplot", target=target)
    # Plot 1-D slices
    marginal0 = q_to_plot[index1_target,:,index2_target]
    marginal1 = q_to_plot[:,index0_target,index2_target]
    marginal2 = q_to_plot[index1_target,index0_target,:]

    fig, ax = plt.subplots(1, 3, figsize = (12,3))

    ax[0].scatter(c_scans[0], marginal0)
    ax[0].set_xlabel("c0")
    ax[0].axvline(target_fit[0], color = "red")
    ax[0].set_ylim(min_val, max_val)

    ax[1].scatter(c_scans[1], marginal1)
    ax[1].set_xlabel("c1")
    ax[1].axvline(target_fit[1], color = "red")
    ax[1].set_ylim(min_val, max_val)

    ax[2].scatter(c_scans[2], marginal2)
    ax[2].set_xlabel("c2")
    ax[2].axvline(target_fit[2], color = "red")
    ax[2].set_ylim(min_val, max_val)
    
    plt.show()
    
    print(5*"\n")
    
    return q_to_plot

    
    
    
def c_scan_3d(trained_network, device, test_features, num_points, upper_limit, loss_type, target=[0,0,0]):
        
    c_grid, c_scans, edges = make_c_grid(num_points, upper_limit)
    
    
    # evaluate the llr
    llr_eval_grid = eval_loglikelihood_ratios(test_features, c_grid, trained_network, device, loss_type)
    
    # get the maximum
    max_index = np.argmax(llr_eval_grid)
    best_fit = c_grid[max_index]
    
    max_val = np.max(llr_eval_grid)
    min_val = np.min(llr_eval_grid)

    sm_index = int((num_points**3)/2)

    print(f"Max llr of {np.max(llr_eval_grid)} at theta = {best_fit}")
    print("llr = {val} at theta = {tt}".format(val = llr_eval_grid[sm_index], tt = c_grid[sm_index]))
    
    index0, index1, index2 = get_coords_in_3d(best_fit, c_scans)


    # reshape the llr
    llr = np.array(llr_eval_grid).reshape((num_points,num_points,num_points))
    plot_2d_slices(llr, edges, best_fit, index0, index1, index2, cmap="gnuplot", target=target)
    
    # Plot 1-D slices
    marginal0 = llr[index1,:,index2]
    marginal1 = llr[:,index0,index2]
    marginal2 = llr[index1,index0,:]

    fig, ax = plt.subplots(1, 3, figsize = (12,3))

    ax[0].scatter(c_scan, marginal0)
    ax[0].set_xlabel("c1")
    ax[0].axvline(best_fit[0], color = "red")
    ax[0].set_ylim(min_val, max_val)

    ax[1].scatter(c_scan, marginal1)
    ax[1].set_xlabel("c1")
    ax[1].axvline(best_fit[1], color = "red")
    ax[1].set_ylim(min_val, max_val)

    ax[2].scatter(c_scan, marginal2)
    ax[2].set_xlabel("c2")
    ax[2].axvline(best_fit[2], color = "red")
    ax[2].set_ylim(min_val, max_val)

    plt.show()



def plot_2d_slice(quantity_to_min_flat, ind0, ind1, edges, c_scans, best_fit, marginal0, marginal1, cmin=None, cmax=None, cmap="viridis_r", target=[0, 0]):
    
    if cmin is None:
        cmin = np.min(quantity_to_min_flat)
    if cmax is None:
        cmax = np.max(quantity_to_min_flat)
        
    
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4]})
    ax[0, 1].axis("off")

    # 2d colormesh
    pcm = ax[1, 0].pcolormesh(edges[ind0], edges[ind1],quantity_to_min_flat, norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),cmap=cmap,)
    cbar = plt.colorbar(pcm, ax=ax[1,1], extend="both")
    # plot best fit
    ax[1, 0].scatter(best_fit[ind0], best_fit[ind1], s=100.0, color="black", marker="*",)
    # plot target
    ax[1, 0].scatter(target[ind0], target[ind1], s=100, facecolors='none', edgecolors='black', linewidth = 2)
    
    # marginals
    ax[0, 0].scatter(c_scans[ind0], marginal0)
    ax[0, 0].axvline(best_fit[ind0], color = "red")
    ax[0, 0].set_ylim(cmin, cmax)
    ax[0, 0].set_xticks([])
    
    ax[1, 1].scatter(marginal1, c_scans[ind1])
    ax[1, 1].axhline(best_fit[ind1], color = "red")
    ax[1, 1].set_xlim(cmin, cmax)
    ax[1, 1].set_yticks([])

    ax[1, 0].set_xlabel(f"$c_{ind0}$", fontsize = 14)
    ax[1, 0].set_ylabel(f"$c_{ind1}$", fontsize = 14)
    cbar.set_label(r"$\mathbb{E}_x [\log \,\hat{r}(x | \theta, \theta_{SM}) ]$")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    


def plot_2d_slices(quantity_to_min_cube, edges, best_fit, index0, index1, index2, cmap = "viridis_r", target=[0, 0, 0]):
    
    cmin = np.min(quantity_to_min_cube)
    cmax = np.max(quantity_to_min_cube)

    
    fig = plt.figure(figsize = (15, 4))
    
    ax0 = fig.add_subplot(1,3,1, aspect = "equal")
    ax1 = fig.add_subplot(1,3,2, aspect = "equal")
    ax2 = fig.add_subplot(1,3,3, aspect = "equal")
    
    # 0-1
    quantity_to_min_flat = quantity_to_min_cube[:,:,index2]
    ax0.pcolormesh(edges[0],edges[1],quantity_to_min_flat,norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),cmap=cmap,)
    ax0.scatter(best_fit[0],best_fit[1],s=100.0,color="black",marker="*",)
    ax0.scatter(target[0], target[1], s=100, facecolors='none', edgecolors='black', linewidth = 2)
    ax0.set_xlabel(f"$c_0$", fontsize = 14)
    ax0.set_ylabel(f"$c_1$", fontsize = 14)

    # 2-1
    quantity_to_min_flat = quantity_to_min_cube[:,index0,:]
    ax1.pcolormesh(edges[2],edges[1],quantity_to_min_flat,norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),cmap=cmap,)
    ax1.scatter(best_fit[2],best_fit[1],s=100.0,color="black",marker="*",)
    ax1.scatter(target[2], target[1], s=100, facecolors='none', edgecolors='black', linewidth = 2)
    ax1.set_xlabel(f"$c_2$", fontsize = 14)
    ax1.set_ylabel(f"$c_1$", fontsize = 14)

    # 2-0
    quantity_to_min_flat = quantity_to_min_cube[index1,:,:]
    pcm = ax2.pcolormesh(edges[2],edges[0], quantity_to_min_flat,norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),cmap=cmap,)
    ax2.scatter(best_fit[2],best_fit[0],s=100.0,color="black",marker="*",)
    ax2.scatter(target[2], target[0], s=100, facecolors='none', edgecolors='black', linewidth = 2)
    ax2.set_xlabel(f"$c_2$", fontsize = 14)
    ax2.set_ylabel(f"$c_0$", fontsize = 14)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(pcm, ax = cbar_ax)

    cbar.set_label(r"$\mathbb{E}_x [\log \,\hat{r}(x | \theta, \theta_{SM}) ]$")

    plt.show()
