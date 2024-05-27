#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
import math
import os

from madminer.delphes import DelphesReader
import argparse

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.DEBUG)

        
import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
    
parser = argparse.ArgumentParser()
parser.add_argument("-p","--process_code",help="Choose from signal_sm, signal_supp, or background_0")
parser.add_argument("-b","--batch_index",help="batch_index")
parser.add_argument("-supp_id","--supp_id",help="Index of non_SM benchmark that events were generated at")
parser.add_argument("-dr","--delphes_run",action="store_true",help="Whether Delphes has been run on the events or not")
parser.add_argument("-start","--start",help="MadGraph run start index")
parser.add_argument("-stop","--stop",help="Madgraph run stop index")
args = parser.parse_args()

    
n_runs = workflow["madgraph"]["n_runs"]
mg_dir = workflow["madgraph"]["dir"]
delphes = DelphesReader(workflow["morphing_setup"])
  
if "background" in args.process_code:
    is_background = True
elif "signal" in args.process_code:
    is_background = False
    
if args.process_code != "signal_supp":
    path_to_events_dir = "{input_dir_prefix}/{process_code}/batch_{batch_index}/".format(input_dir_prefix=workflow["delphes"]["input_dir_prefix"], process_code = args.process_code, batch_index=args.batch_index)
    sampled_from_benchmark = "sm"
else:
    path_to_events_dir = "{input_dir_prefix}/{process_code}/mb_vector_{supp_id}/batch_{batch_index}/".format(input_dir_prefix=workflow["delphes"]["input_dir_prefix"], process_code = args.process_code, batch_index=args.batch_index, supp_id = args.supp_id)
    sampled_from_benchmark = f"morphing_basis_vector_{args.supp_id}"
    
    
for run_id in range(int(args.start), int(args.stop)+1):   
    
    # background events have not gone through MadSpin
    if "background" in args.process_code:
        loc_dir = f"{path_to_events_dir}/run_{str(run_id).zfill(2)}"
    else:
        loc_dir = f"{path_to_events_dir}/run_{str(run_id).zfill(2)}_decayed_1"
    
    print(loc_dir)

    if args.delphes_run: 
        delphes.add_sample(lhe_filename=f"{loc_dir}/unweighted_events.lhe.gz",
                            hepmc_filename=f"{loc_dir}/tag_1_pythia8_events.hepmc.gz",
                               delphes_filename=f"{loc_dir}/tag_1_pythia8_events_delphes.root",
                               weights="lhe",
                            sampled_from_benchmark=sampled_from_benchmark,
                            is_background=is_background,
                            k_factor= 1.0,)
    else: 
        delphes.add_sample(lhe_filename=f"{loc_dir}/unweighted_events.lhe.gz",
                            hepmc_filename=f"{loc_dir}/tag_1_pythia8_events.hepmc.gz",
                               weights="lhe",
                            sampled_from_benchmark=sampled_from_benchmark,
                            is_background=is_background,
                            k_factor= 1/0,)

                                                                       
# Now we run Delphes on these samples (you can also do this externally and then add the keyword `delphes_filename` when calling `DelphesReader.add_sample()`):


if args.delphes_run: 
    print("Delphes has already been run.")
else:
    delphes.run_delphes(
        delphes_directory=mg_dir + "/Delphes",
        delphes_card="cards/delphes_card_HLLHC.tcl",
        log_file="logs/delphes.log",
    )
"""
CUSTOM FUNCTIONS TO ISOLATE THE BJETS
"""

def count_bjets(l,a,j,met):
    btag_sum = 0
    for jet in j:
        btag_sum += jet.b_tag
    return btag_sum

def get_hardest_b_pt(l,a,j,met):
    for jet in j:
        if jet.b_tag == 1:
            return jet.pt
    return np.nan

def get_second_b_pt(l,a,j,met):
    num_bs = 0
    for jet in j:
        if jet.b_tag == 1:
            num_bs += 1
        if num_bs == 2:
            return jet.pt
    return np.nan

def get_hardest_b_eta(l,a,j,met):
    for jet in j:
        if jet.b_tag == 1:
            return jet.eta
    return np.nan

def get_second_b_eta(l,a,j,met):
    num_bs = 0
    for jet in j:
        if jet.b_tag == 1:
            num_bs += 1
        if num_bs == 2:
            return jet.eta
    return np.nan

def get_two_bjets(j):
    num_bjets_found = 0
    for jet in j:
        if jet.b_tag == 1:
            num_bjets_found += 1
            if num_bjets_found == 1:
                hardest_b = jet
            elif num_bjets_found == 2:
                second_b = jet
                return (True, hardest_b, second_b)
    return (False, False, False)

def get_bb_deltaR(l,a,j,met):
    found, hardest_b, second_b = get_two_bjets(j)
    if found:
        return hardest_b.deltaR(second_b)
    else:
        return np.nan
    
def get_b0a0_deltaR(l,a,j,met):
    found, hardest_b, second_b = get_two_bjets(j)
    if found:
        return hardest_b.deltaR(a[0])
    else:
        return np.nan
    
def get_b0a1_deltaR(l,a,j,met):
    found, hardest_b, second_b = get_two_bjets(j)
    if found:
        return hardest_b.deltaR(a[1])
    else:
        return np.nan
    
def get_b1a0_deltaR(l,a,j,met):
    found, hardest_b, second_b = get_two_bjets(j)
    if found:
        return second_b.deltaR(a[0])
    else:
        return np.nan  

def get_b1a1_deltaR(l,a,j,met):
    found, hardest_b, second_b = get_two_bjets(j)
    if found:
        return second_b.deltaR(a[1])
    else:
        return np.nan
    
def get_mbb(l,a,j,met):
    found, hardest_b, second_b = get_two_bjets(j)
    if found:
        return (hardest_b+second_b).m
    else:
        return np.nan
    
def get_ptbb(l,a,j,met):
    found, hardest_b, second_b = get_two_bjets(j)
    if found:
        return (hardest_b+second_b).pt
    else:
        return np.nan
    
def get_mtot(l,a,j,met):
    found, hardest_b, second_b = get_two_bjets(j)
    if found:
        return (hardest_b+second_b+a[0]+a[1]).m
    else:
        return np.nan
    
        
"""
MAIN ANALYSIS
"""

def add_observables(delphes):
    
    # photons
    delphes.add_observable( "a0_pt", "a[0].pt", required=True )
    delphes.add_observable( "a0_eta", "a[0].eta", required=True )
    delphes.add_observable( "a1_pt", "a[1].pt", required=True )
    delphes.add_observable( "a1_eta", "a[1].eta", required=True )
    
    # b-jets
    delphes.add_observable_from_function( "num_bjets", count_bjets, required=True )
    delphes.add_observable_from_function( "b0_pt", get_hardest_b_pt, required=True )
    delphes.add_observable_from_function( "b0_eta", get_hardest_b_eta, required=True )
    delphes.add_observable_from_function( "b1_pt", get_second_b_pt, required=True )
    delphes.add_observable_from_function( "b1_eta", get_second_b_eta, required=True )
    
    # deltaR
    delphes.add_observable_from_function( "bb_deltaR", get_bb_deltaR, required=True )
    delphes.add_observable( "aa_deltaR", "a[0].deltaR(a[1])", required=True )
    delphes.add_observable_from_function( "b0a0_deltaR", get_b0a0_deltaR, required=True )
    delphes.add_observable_from_function( "b0a1_deltaR", get_b0a1_deltaR, required=True )
    delphes.add_observable_from_function( "b1a0_deltaR", get_b1a0_deltaR, required=True )
    delphes.add_observable_from_function( "b1a1_deltaR", get_b1a1_deltaR, required=True )

    # misc
    delphes.add_observable_from_function( "m_tot", get_mtot, required=True )
    delphes.add_observable_from_function( "pt_bb", get_ptbb, required=True )
    delphes.add_observable( "pt_aa", "(a[0]+a[1]).pt", required=True )
    delphes.add_observable_from_function( "m_bb", get_mbb, required=True )
    delphes.add_observable( "m_aa", "(a[0]+a[1]).m", required=True )


def add_cuts_and_efficiencies(delphes, region=None):


    # pt cuts
    delphes.add_cut('a0_pt>30')
    delphes.add_cut('a1_pt>30')
    delphes.add_cut('b0_pt>30')
    delphes.add_cut('b1_pt>30')
    
    # eta cuts
    delphes.add_cut('abs(a0_eta)<2.4')
    delphes.add_cut('abs(a1_eta)<2.4')
    delphes.add_cut('abs(b0_eta)<2.4')
    delphes.add_cut('abs(b1_eta)<2.4')

    # well-separated leptons, jets
    delphes.add_cut('bb_deltaR>0.4')
    delphes.add_cut('aa_deltaR>0.4')
                                   
    # changed
    delphes.add_cut('b0a0_deltaR>1')
    delphes.add_cut('b0a1_deltaR>1')
    delphes.add_cut('b1a0_deltaR>1')
    delphes.add_cut('b1a1_deltaR>1')
    # Higgs mass window 

    delphes.add_cut('abs(m_bb-125)<25')
    delphes.add_cut('abs(m_aa-125)<3') 

    # misc
    delphes.add_cut('num_bjets==2') 
    delphes.add_cut('aa_deltaR<2')
                                   

add_observables(delphes)
add_cuts_and_efficiencies(delphes)
    
# 4. Run analysis
delphes.analyse_delphes_samples()

# 5. Save results into new .h5 file
if args.process_code != "signal_supp":
    delphes.save("{delphes_output_data}_{process_code}_batch_{batch_index}.h5".format(delphes_output_data=workflow["delphes"]["output_file"], process_code=args.process_code, batch_index=args.batch_index))
else: 
    delphes.save("{delphes_output_data}_{process_code}_{supp_id}_batch_{batch_index}.h5".format(delphes_output_data=workflow["delphes"]["output_file"], process_code=args.process_code, batch_index=args.batch_index, supp_id = args.supp_id))
