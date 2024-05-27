import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import torch
from numba import cuda 
import argparse
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from helpers.network_training import *
from helpers.utils import np_to_torch, crop_feature

parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-p", "--parameter_code", help = "Which Wilson coefficient(s) you're scanning over.")
parser.add_argument("-dtype", "--dtype", help = "Data type (Delphes)", default = "delphes_s")
parser.add_argument("-rid", "--run_id", help = "run_id")
parser.add_argument("-f", "--num_features", help = "Number of features to use while training")
parser.add_argument("-n", "--network", help = "Network architecture", default = "dense")
parser.add_argument("-c1",action='store_true',help="Train classifier 1")
parser.add_argument("-c2",action='store_true',help="Train classifier 2")
parser.add_argument("-c3",action='store_true',help="Train classifier 3")
parser.add_argument("-s", "--seed", help = "Random seed", default = 0)


# Read arguments from command line
args = parser.parse_args()


# computing
#os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# set the number of threads that pytorch will use
torch.set_num_threads(2)

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)


# workflow
import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
run_id = args.run_id
run_configs = {}
run_configs["input_precode"] = args.dtype
run_configs["parameter_code"] = args.parameter_code
run_configs["network_id"] = run_id
run_configs["seed"] = args.seed
seed = int(args.seed)

# features to train on

# in order: m_hh, pt_bb, pt_aa, deltaR_aa, deltaR_bb, a0_pt, a1_pt, b0_pt, b1_pt
# These correspond to the features (in order) in the function add_observables of 03_a_read_delphes.py
run_configs["features"] = [15, 16, 17, 10, 9, 0, 2, 5, 7][:int(args.num_features)]


run_configs["bkg.N_train"] = 10000000 # To reduce training time, if necessary

# network architecture
run_configs["network.type"] = args.network
run_configs["network.layers"] = [32, 32]

# training hyperparameters
run_configs["hyperparam.batch_size"] = 1024
run_configs["hyperparam.lr"] = 0.001
run_configs["hyperparam.n_epochs"] = 150
run_configs["hyperparam.patience_ES"] = 20
run_configs["hyperparam.patience_lr"] = 5

with open(f"run_configs/{run_id}.yml", "w") as outfile:
    yaml.dump(run_configs, outfile, default_flow_style=False)


n_features = len(run_configs["features"])

print("Loading in samples from with {parameter_code}.".format(parameter_code=run_configs["parameter_code"]))
print("Running analysis on {input_precode}.".format(input_precode=run_configs["input_precode"]))
print("Analysis will use features {features}".format(features=run_configs["features"]))
print()


N_train = int(run_configs["bkg.N_train"])
samples_dir = workflow["sampling"]["output_dir"]
identity_code = run_configs["input_precode"]
features = run_configs["features"]
parameter_code = run_configs["parameter_code"]

# load in the samples
samples_SM = np.load(f'{samples_dir}/plain_real/{identity_code}/x_sm.npy')[:,features]
samples_alt = np.load(f'{samples_dir}/plain_real/{identity_code}/x_alt_{parameter_code}.npy')[:,features]
samples_bkg = np.load(f'{samples_dir}/plain_real/delphes_b0/x_bkg.npy')[:,features]
# load in the theta values
theta_alt = np.load(f'{samples_dir}/plain_real/{identity_code}/theta_alt_{parameter_code}.npy')
theta_alt_sm = np.load(f'{samples_dir}/plain_real/{identity_code}/theta_alt_{parameter_code}.npy')

# shuffle the samples, since they are grouped in chunks of generating theta out of the box
#samples_alt, theta_alt, samples_SM, theta_alt_sm = shuffle(samples_alt, theta_alt, samples_SM, theta_alt_sm, random_state = 42) 

# crop to the number of desired signal events
samples_SM = samples_SM[:N_train]
samples_alt = samples_alt[:N_train]
samples_bkg = samples_bkg[:N_train]
theta_alt = theta_alt[:N_train]
theta_alt_sm = theta_alt_sm[:N_train]

print("Preprocessing data...")
print()

all_data = np.vstack((samples_SM, samples_bkg))

scaler = StandardScaler()
scaler.fit(all_data)

# transform
samples_SM = scaler.transform(samples_SM)
samples_alt = scaler.transform(samples_alt)
samples_bkg = scaler.transform(samples_bkg)

with open(f"models/scaler_{run_id}", "wb") as ofile:
    pickle.dump(scaler, ofile)


def train_classifier(train_set_0, train_set_1, loc_id):
    
    x_train = np.vstack([train_set_0, train_set_1])
    all_labels = np.vstack([np.zeros((train_set_0.shape[0], 1)), np.ones((train_set_1.shape[0], 1))])
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, all_labels, test_size=0.2, random_state = seed)
    print(f"X_train: {X_train.shape}.\nX_val: {X_val.shape}.\nY_train: {Y_train.shape}.\nY_val: {Y_val.shape}.")
    
    X_train = np_to_torch(X_train, device)
    X_val = np_to_torch(X_val, device)
    Y_train = np_to_torch(Y_train, device)
    Y_val = np_to_torch(Y_val, device)
    kl_weight = run_configs["hyperparam.batch_size"]/X_train.shape[0]
    
    if args.network == "bnn":
        dense_net = BNN(n_inputs = train_set_0.shape[1], layers = run_configs["network.layers"], prior_sigma = 0.1)
        optimizer = torch.optim.AdamW(dense_net.parameters(), lr = run_configs["hyperparam.lr"], weight_decay = 0)
    elif args.network == "dense":
        dense_net = NeuralNet(n_inputs = train_set_0.shape[1], layers = run_configs["network.layers"])
        optimizer = torch.optim.AdamW(dense_net.parameters(), lr = run_configs["hyperparam.lr"], weight_decay = kl_weight)
    
    
    print("Starting network training...")
    epochs, losses, losses_val = train_network(X_train, Y_train, X_val, Y_val, dense_net, optimizer, run_configs["hyperparam.n_epochs"], run_configs["hyperparam.batch_size"], device, seed = seed, train_bnn = False, kl_weight = kl_weight, network_id = f"models/{run_id}_{loc_id}", use_early_stop = True, min_delta = 0, patience_ES = run_configs["hyperparam.patience_ES"], patience_lr = run_configs["hyperparam.patience_lr"], loss_type = "BCE")

    

print(f"Using {args.network} type networks.")
    

"""
TRAIN CLASSIFIER 1
    - learn LR of alternative S (class 1) to SM S (class 0)
    - this must be a parameterized classifier
"""
if args.c1:
    print("Training classifier 1...")
    denom_c1 = np.c_[samples_SM, theta_alt_sm/10.0] 
    numer_c1 = np.c_[samples_alt, theta_alt/10.0] 
    train_classifier(denom_c1, numer_c1, "Ssm_Salt")
    print("Done with classifier 1!\n")

"""
TRAIN CLASSIFIER 2
    - learn LR of alternative S (class 1) to  B (class 0)
    - parameterized classifier
"""
if args.c2:
    print("Training classifier 2...")
    denom_c2 = np.c_[samples_bkg, theta_alt_sm/10.0] 
    numer_c2 = np.c_[samples_alt, theta_alt/10.0] 
    train_classifier(denom_c2, numer_c2, "B_Salt")
    print("Done with classifier 2!\n")

"""
TRAIN CLASSIFIER 3
    - learn LR of B (class 1) to SM S (class 0)
    - non-parameterized classifier
    - only needs to be run once for a given feature set
"""
if args.c3:
    print("Training classifier 3...")
    train_classifier(samples_SM, samples_bkg, "Ssm_B")
    print("Done with classifier 3!\n")