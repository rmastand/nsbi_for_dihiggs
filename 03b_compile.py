#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
import math

from madminer.sampling import combine_and_shuffle
import argparse

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)
 
import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
parser = argparse.ArgumentParser()
parser.add_argument("-p","--process_code",help="process_code",default="Choose signal or background")
parser.add_argument("-n","--num_batch",help="num_background_batches",default=1,type=int)

args = parser.parse_args()   

# batch indices for the SM benchmark
signal_batches = [0, 1, 2, 3] # CHANGE THIS

# {morphing basis: batch indices} for non-SM benchmarks
supp_batches = {1:[0,1],2:[0,1],3:[0,1],4:[0,1],5:[0],6:[0],7:[0],8:[0,1],9:[0,1]} # CHANGE THIS



if args.process_code == "signal":
    to_combine = []
    for i in signal_batches: # signal sm
        to_combine.append('{long_term_storage_dir}/delphes_signal_sm_batch_{batch_num}.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"], batch_num=i))
    for supp_id in range(1, 10): # signal supp
        for i in supp_batches[supp_id]:
            to_combine.append('{long_term_storage_dir}/delphes_signal_supp_{supp_id}_batch_{batch_num}.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"], batch_num=i, supp_id=supp_id))

    combine_and_shuffle(
        to_combine,
        '{long_term_storage_dir}/delphes_s_shuffled.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"])
    )

elif args.process_code == "background": # i.e. background only
    to_combine = []
    k_factors_background = [1 for x in range(args.num_batch)]  

    print(f"Adding in {args.num_batch} batches of background 0...")
    for i in range(args.num_batch):
        to_combine.append('{long_term_storage_dir}/delphes_background_0_batch_{batch_num}.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"], batch_num=i))
    combine_and_shuffle(
        to_combine,
        '{long_term_storage_dir}/delphes_b0_shuffled.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"]),
        k_factors=k_factors_background
    )

