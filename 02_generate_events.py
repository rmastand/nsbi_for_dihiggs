#!/usr/bin/env python
# coding: utf-8
import os
import logging
import numpy as np

from madminer.core import MadMiner
import argparse

# MadMiner output
logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG,
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)
        
import yaml
working_dir = "/global/u1/r/rmastand/trihiggs_coupling/hh_prod/" # CHANGE THIS LINE
with open(f"{working_dir}/workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
parser = argparse.ArgumentParser()
parser.add_argument("-sm",action="store_true",help="Generate events only at the SM benchmark")
parser.add_argument("-supp",action="store_true",help="Generate events at a non-SM benchmark")
parser.add_argument("-supp_id",help="Index of non_SM benchmark to generate events")
parser.add_argument("-b",action="store_true",help="Generate background events (no reweighting needed)")

args = parser.parse_args()

    
mg_dir = workflow["madgraph"]["dir"]

"""
GENERATE EVENTS
"""

n_runs = workflow["madgraph"]["n_runs"]

miner = MadMiner()
miner.load(workflow["morphing_setup"])

if int(workflow["madgraph"]["energy"]) == 14:
    run_cards_signal = [f"{working_dir}/cards/run_cards/run_card_signal_14TeV.dat" for i in range(n_runs)]
    run_cards_background = [f"{working_dir}/cards/run_cards/run_card_background_14TeV.dat" for i in range(n_runs)]
elif int(workflow["madgraph"]["energy"]) == 100:
    run_cards_signal = [f"{working_dir}/cards/run_cards/run_card_signal_100TeV.dat" for i in range(n_runs)]
    run_cards_background = [f"{working_dir}/cards/run_cards/run_card_background_100TeV.dat" for i in range(n_runs)]
else:
    print("No run card for desired energy.")
    exit()

print(f"\nRunning at energy", workflow["madgraph"]["energy"], "TeV\n")

additional_benchmarks = ["morphing_basis_vector_1", "morphing_basis_vector_2", "morphing_basis_vector_3", "morphing_basis_vector_4", "morphing_basis_vector_5", "morphing_basis_vector_6", "morphing_basis_vector_7", "morphing_basis_vector_8", "morphing_basis_vector_9" ]


if args.sm:

    miner.run_multiple(
        sample_benchmarks=["sm"],
        mg_directory=mg_dir,
        mg_process_directory="{mg_process_output_dir}/signal_sm".format(mg_process_output_dir = workflow["madgraph"]["output_dir"]),
        proc_card_file=f"{working_dir}/cards/proc_card_signal.dat",
        param_card_template_file=f"{working_dir}/cards/restrict_LO.dat",
        madspin_card_file=f"{working_dir}/cards/madspin_card.dat",
        run_card_files=run_cards_signal,
        pythia8_card_file=f"{working_dir}/cards/pythia8_card.dat", 
        log_directory=f"{working_dir}/logs/signal_sm",
        python_executable="python3",
        order="LO",
        #systematics=["signal_norm"]
    )

if args.supp:
    
    miner.run_multiple(
        #sample_benchmarks=additional_benchmarks,
        sample_benchmarks=[f"morphing_basis_vector_{args.supp_id}"],
        mg_directory=mg_dir,
        mg_process_directory="{mg_process_output_dir}/signal_supp/morphing_basis_vector_{supp_id}".format(mg_process_output_dir = workflow["madgraph"]["output_dir"], supp_id = args.supp_id),
        proc_card_file=f"{working_dir}/cards/proc_card_signal.dat",
        param_card_template_file=f"{working_dir}/cards/restrict_LO.dat",
        madspin_card_file=f"{working_dir}/cards/madspin_card.dat",
        run_card_files=run_cards_signal,
        pythia8_card_file=f"{working_dir}/cards/pythia8_card.dat", 
        log_directory=f"{working_dir}/logs/signal_supp",
        python_executable="python3",
        order="LO",
        #systematics=["signal_norm"]
    )


if args.b:
    for i in range(1):
        miner.run_multiple(
            is_background=True,
            sample_benchmarks=["sm"],
            mg_directory=mg_dir,
            mg_process_directory="{mg_process_output_dir}_2/background_{idd}".format(mg_process_output_dir = workflow["madgraph"]["output_dir"], idd = i),
            proc_card_file="{working_dir}/cards/proc_card_background_{idd}.dat".format(idd = i, working_dir = working_dir),
            param_card_template_file=f"{working_dir}/cards/restrict_LO.dat",
            pythia8_card_file=f"{working_dir}/cards/pythia8_card.dat", 
            run_card_files=run_cards_background,
            log_directory=f"{working_dir}/logs_2/background_{i}",
        )
