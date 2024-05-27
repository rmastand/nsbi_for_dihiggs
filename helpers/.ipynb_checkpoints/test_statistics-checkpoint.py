import numpy as np
import torch

from helpers.utils import np_to_torch
from madminer import SampleAugmenter, sampling


eps = 1e-10

def eval_test_statistic_rate(N_obs, N_A, N_SM_target):
    """
    N_obs: number of observed signal events, S and B
    
    N_A: number of events (S+B) predicted by the alternative hypothesis
    
    N_SM_target: number of events (S+B) predicted by the SM hypothesis
    """
    q = (-N_A + N_SM_target + N_obs*np.log(N_A/N_SM_target))
    return 2.0*q

def get_N_sig_obs_at_c_point(data_input_dir, input_precode, c_point, N_sig_SM_target):
    
    """
    
    data_input_dir: path to where the data is stored
    
    input_precode: {lhe or delphes}_{s or sb}
    
    N_SM_target: FIXED. Read from the paper.
    
    """
    
    # load in the sampler
    sampler = SampleAugmenter(f'{data_input_dir}/{input_precode}_shuffled.h5')
    
    # get the scale factor to the SM
    _, xsecs_morphing, _ = sampler.cross_sections(theta=sampling.morphing_point((0,0,0)))
    scale_factor = N_sig_SM_target/xsecs_morphing[0]
    
    # get the unscaled cross-sections at the arbitrary c_grid points
    thetas_morphing, xsecs_morphing, xsec_errors_morphing = sampler.cross_sections(
        theta=sampling.morphing_point(c_point)
    )
    
    # convert cross sections to N_events
    N_sig_events_c_point = scale_factor*xsecs_morphing[0]

    return N_sig_events_c_point


def get_test_statistic_rate_at_c_points(data_input_dir, input_precode, c_grid, N_sig_SM_target, N_bkg_SM_target, N_obs):
    
    """
    ALL INPUTS AND OUTPUTS ARE WITHOUT BACKGROUND
    
    data_input_dir: path to where the data is stored
    
    input_precode: {lhe or delphes}_{s or sb}
    
    N_sig_SM_target: FIXED. Read from the paper. todo: should be ~5000 for 100 TeV
    
    N_obs: test set size. THIS IS SIGNAL AND BACKGROUND

    """
    
    # load in the sampler
    sampler = SampleAugmenter(f'{data_input_dir}/{input_precode}_shuffled.h5')
    
    # get the scale factor to the SM
    _, xsecs_morphing, _ = sampler.cross_sections(theta=sampling.morphing_point((0,0,0)))
    scale_factor = N_sig_SM_target/xsecs_morphing[0]
    
    # get the unscaled cross-sections at the arbitrary c_grid points
    thetas_morphing, xsecs_morphing, xsec_errors_morphing = sampler.cross_sections(
        theta=sampling.morphing_points(c_grid)
    )
    
    # convert cross sections to get N signal events at arbitrary C points
    N_sig_c_scan = scale_factor*xsecs_morphing
    
    # get the test statistics
    # the morping is only for signal events. We only add bkg effects at this stage
    q_rate = eval_test_statistic_rate(N_obs, N_sig_c_scan + N_bkg_SM_target, N_sig_SM_target + N_bkg_SM_target)

    return q_rate, N_sig_c_scan
    

def eval_test_statistic_shape(samples_test, c_grid, trained_network, device):
    

    Evaluate the bnn a different time for every point in the test set (all thetas at once)
    llr_vals = []
    with torch.no_grad():
        for x_loc_index in range(samples_test.shape[0]):
            # "tensor product" between ONE x value (observables) and coeffs to test
            loc_x_point_expand = np.repeat(samples_test[x_loc_index].reshape(-1,1), c_grid.shape[0], axis = 1).T
            loc_x_test = np.c_[loc_x_point_expand, c_grid]
            
            if isinstance(trained_network, RepulsiveNeuralNet):
                # expand the channels out
                loc_x_test = np_to_torch(loc_x_test, device)[None,:,:].expand(trained_network.n_channels,-1,-1) # (n_channels, batch_size, n_features)
                preds = trained_network(loc_x_test, device).detach().cpu().numpy().mean(axis=0)
            else:
                preds = trained_network(np_to_torch(loc_x_test, device)).detach().cpu().numpy()    
            
            llr_vals.append(np.log(preds/(1.0 - preds + eps)))
        
        llr_vals = np.hstack(llr_vals)
        llr_eval_grid = np.sum(llr_vals, axis = 1) # sum over all the x values (and also over all the network calls)
        
    return 2*llr_eval_grid # to get the test statistic



def eval_test_statistic_shape_mixture_model_4_comp(samples_test, c_grid, trained_network_c1, 
                              trained_network_c2, trained_network_c3, N_sig_c_scan, N_sig_SM, N_bkg_SM, device, network_type = "dense"):
    
    """
    Evaluates the full shape-based test statistic for a given tets set
    
    Also returns N_sig, which is necessary for the rate-level test statistic
    """
    
    N_test_set = samples_test.shape[0]
    N_c_points = c_grid.shape[0]
    
    # Calculate the signal and background fractions
    f_sig_SM = N_sig_SM / (N_sig_SM + N_bkg_SM)
    f_sig_alt = N_sig_c_scan / (N_sig_c_scan + N_bkg_SM) # shape: (# c points, )
        
    # Calculate the likelihood ratios
    
    # for B to S_SM
    if network_type == "dense":
        with torch.no_grad():
            preds_c3 = trained_network_c3(np_to_torch(samples_test, device)).detach().cpu().numpy()
    elif network_type == "bdt":
        preds_c3 = trained_network_c3.predict_proba(samples_test)[:,1].reshape(-1,1)
    lr_B_to_S_SM = preds_c3/(1.0 - preds_c3 + eps)   # shape: (# test points, 1)
    
    
    with torch.no_grad():
        lr_vals_S = [] # for S_alt to S_SM
        lr_vals_B = [] # for S_alt to B
        # Method is to interate through every point in the test set and construct the c-grid
        # This is computationally expensive, especially if the c-grid is fine
        for x_loc_index in range(N_test_set):
            # "tensor product" between ONE x value (observables) and coeffs to test
            loc_x_point_expand = np.repeat(samples_test[x_loc_index].reshape(-1,1), N_c_points, axis = 1).T
            loc_x_test = np.c_[loc_x_point_expand, c_grid]
            if network_type == "dense":
                preds_c1 = trained_network_c1(np_to_torch(loc_x_test, device)).detach().cpu().numpy()    
                preds_c2 = trained_network_c2(np_to_torch(loc_x_test, device)).detach().cpu().numpy() 
            elif network_type == "bdt":
                preds_c1 = trained_network_c1.predict_proba(loc_x_test)[:,1].reshape(-1,1)
                preds_c2 = trained_network_c2.predict_proba(loc_x_test)[:,1].reshape(-1,1)
            lr_vals_S.append(preds_c1/(1.0 - preds_c1 + eps))
            lr_vals_B.append(preds_c2/(1.0 - preds_c2 + eps))
        
    lr_S_alt_to_S_SM = np.hstack(lr_vals_S) # shape: (# c points, # test points)
    lr_S_alt_to_B = np.hstack(lr_vals_B) # shape: (# c points, # test points)
    
    # calculate the lr
    comp_1 = np.array([(f_sig_SM/f_sig_alt[i])*(1.0/lr_S_alt_to_S_SM[i,:]) for i in range(N_c_points)]) # shape: (# c points, # test points)
    
    comp_2 = np.array([((1.0 - f_sig_SM) / f_sig_alt[i])*(1.0/lr_S_alt_to_B[i,:]) for i in range(N_c_points)])  # shape: (# c points, # test points)
    
    comp_3  = np.array([(f_sig_SM / (1.0 - f_sig_alt[i]))*(1.0/lr_B_to_S_SM[:]) for i in range(N_c_points)])  # shape: (# c points, # test points, 1)
    
    comp_4 = np.array([((1.0 - f_sig_SM) / (1.0 - f_sig_alt[i])) for i in range(N_c_points)])  # shape: (# c points,)
    
    lr_0 = np.expand_dims(1.0/(comp_1 + comp_2), axis = 2) 
    lr_1 = np.array([1.0/(comp_3[i,:,:] + comp_4[i]) for i in range(N_c_points)])
    
    
    llr_eval_grid = np.sum(np.log(lr_0 + lr_1), axis = 1)

    return 2*llr_eval_grid.reshape(-1,) # to get the test stat

def eval_test_statistic_shape_mixture_model_4_comp_ensemble(samples_test, c_grid, trained_network_dict_seeds, seeds_to_ensemble, N_sig_c_scan, N_sig_SM, N_bkg_SM, device, network_type = "dense"):
    
    """
    Evaluates the full shape-based test statistic for a given tets set
    
    Also returns N_sig, which is necessary for the rate-level test statistic
    """
    
    N_test_set = samples_test.shape[0]
    N_c_points = c_grid.shape[0]
    
    # Calculate the signal and background fractions
    f_sig_SM = N_sig_SM / (N_sig_SM + N_bkg_SM)
    f_sig_alt = N_sig_c_scan / (N_sig_c_scan + N_bkg_SM) # shape: (# c points, )
        
    # Calculate the likelihood ratios
    
    # for B to S_SM
    lr_B_to_S_SM = []
    for s in seeds_to_ensemble:
        if network_type == "dense":
            with torch.no_grad():
                preds_c3 = trained_network_dict_seeds[s]["Ssm_B"](np_to_torch(samples_test, device)).detach().cpu().numpy()
        elif network_type == "bdt":
            preds_c3 = trained_network_dict_seeds[s]["Ssm_B"].predict_proba(samples_test)[:,1].reshape(-1,1)
        lr_B_to_S_SM.append(preds_c3/(1.0 - preds_c3 + eps))   # shape: (# test points, 1)
    lr_B_to_S_SM = np.mean(np.hstack(lr_B_to_S_SM), axis = 1).reshape(-1,1)

    with torch.no_grad():
        lr_vals_S = [] # for S_alt to S_SM
        lr_vals_B = [] # for S_alt to B
        # Method is to interate through every point in the test set and construct the c-grid
        # This is computationally expensive, especially if the c-grid is fine
        for x_loc_index in range(N_test_set):
            # "tensor product" between ONE x value (observables) and coeffs to test
            loc_x_point_expand = np.repeat(samples_test[x_loc_index].reshape(-1,1), N_c_points, axis = 1).T
            loc_x_test = np.c_[loc_x_point_expand, c_grid]
            
            loc_scores_c1, loc_scores_c2 = [], []
            for s in seeds_to_ensemble:
            
                if network_type == "dense":
                    preds_c1 = trained_network_dict_seeds[s]["Ssm_Salt"](np_to_torch(loc_x_test, device)).detach().cpu().numpy() 
                    preds_c2 = trained_network_dict_seeds[s]["B_Salt"](np_to_torch(loc_x_test, device)).detach().cpu().numpy() 
                elif network_type == "bdt":
                    preds_c1 = trained_network_dict_seeds[s]["Ssm_Salt"].predict_proba(loc_x_test)[:,1].reshape(-1,1)
                    preds_c2 = trained_network_dict_seeds[s]["B_Salt"].predict_proba(loc_x_test)[:,1].reshape(-1,1)
                    
                loc_scores_c1.append(preds_c1/(1.0 - preds_c1 + eps))
                loc_scores_c2.append(preds_c2/(1.0 - preds_c2 + eps))
                
            lr_vals_S.append(np.mean(np.hstack(loc_scores_c1), axis = 1).reshape(-1, 1))
            lr_vals_B.append(np.mean(np.hstack(loc_scores_c2), axis = 1).reshape(-1, 1))
            
        
    lr_S_alt_to_S_SM = np.hstack(lr_vals_S) # shape: (# c points, # test points)
    lr_S_alt_to_B = np.hstack(lr_vals_B) # shape: (# c points, # test points)
    
    # calculate the lr
    comp_1 = np.array([(f_sig_SM/f_sig_alt[i])*(1.0/lr_S_alt_to_S_SM[i,:]) for i in range(N_c_points)]) # shape: (# c points, # test points)
    
    comp_2 = np.array([((1.0 - f_sig_SM) / f_sig_alt[i])*(1.0/lr_S_alt_to_B[i,:]) for i in range(N_c_points)])  # shape: (# c points, # test points)
    
    comp_3  = np.array([(f_sig_SM / (1.0 - f_sig_alt[i]))*(1.0/lr_B_to_S_SM[:]) for i in range(N_c_points)])  # shape: (# c points, # test points, 1)
    
    comp_4 = np.array([((1.0 - f_sig_SM) / (1.0 - f_sig_alt[i])) for i in range(N_c_points)])  # shape: (# c points,)
    
    lr_0 = np.expand_dims(1.0/(comp_1 + comp_2), axis = 2) 
    lr_1 = np.array([1.0/(comp_3[i,:,:] + comp_4[i]) for i in range(N_c_points)])
    
    
    llr_eval_grid = np.sum(np.log(lr_0 + lr_1), axis = 1)
    
    # normalize to zero

    llr_eval_grid = llr_eval_grid - np.max(llr_eval_grid)

    return 2*llr_eval_grid.reshape(-1,) # to get the test stat


def get_errorbands(q_vals, sigma, n_dof=1):
    
    median_index = np.argmax(q_vals)
    
    chi2_vals = {1:{1:0.94890, # {dof:{sigma:}}
                    2:3.84146,
                    3:6.63490},
                 2:{1:2.21733,
                    2:5.99146,
                    3:9.21034}
                }
    
    lower_ordinate = np.max(q_vals) - chi2_vals[n_dof][sigma]
        
    low_to_high, high_to_low = [], []
    
    for i in range(len(q_vals)-1):
        if (q_vals[i] <=  lower_ordinate) and (q_vals[i+1] >  lower_ordinate):
            low_to_high.append(i)
        elif (q_vals[i] >=  lower_ordinate) and (q_vals[i+1] < lower_ordinate):
            high_to_low.append(i)
        
    return low_to_high, high_to_low


    
def get_errorbands_local_min(q_vals, sigma, n_dof=1):
    
    chi2_vals = {1:{1:0.94890, # {dof:{sigma:}}
                    2:3.84146,
                    3:6.63490}
                }
    
    # get local maxima
    local_max_x = []
    local_max_y = []
    
    for i in range(1, len(q_vals)-1):
        if (q_vals[i] >  q_vals[i-1]) and (q_vals[i] >  q_vals[i+1]):
            local_max_x.append(i)
            local_max_y.append(q_vals[i])
            
    try:
        median_index = local_max_x[np.argmax(local_max_y)]
    except ValueError:
        return [0], [len(q_vals)-1]
        
    lower_ordinate = np.max(local_max_y) - chi2_vals[n_dof][sigma]

    # determine the CI around the local max    
    # get the upper index
    low_to_high, high_to_low = [], []
    
    for i in range(len(q_vals)-1):
        if (q_vals[i] <=  lower_ordinate) and (q_vals[i+1] >  lower_ordinate):
            low_to_high.append(i)
        elif (q_vals[i] >=  lower_ordinate) and (q_vals[i+1] < lower_ordinate):
            high_to_low.append(i)

    return low_to_high, high_to_low

    
 