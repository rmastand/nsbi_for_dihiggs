# Neural Simulation-based Inference: Full Analysis Pipeline
This is the repository for the code used in paper "Constraining the Higgs Potential with Neural Simulation-based Inference for Di-Higgs Production" ()

Authors: Radha Mastandrea, Benjamin Nachman, and Tilman Plehn

Dataset:[ https://zenodo.org/uploads/10154213](https://zenodo.org/uploads/11222924)

### General version notes
- Use Python 3.8 and MadGraph 3.5.1 with this repository. 
- Within your MadGraph installation, you will need LHAPDF and the [SMEFT@NLO model](https://feynrules.irmp.ucl.ac.be/wiki/SMEFTatNLO)


## Analysis flow

The file `workflow.yaml` is used for I/O management, so such folders don't have to be defined in every script.

Most of the event generation (steps 1-4) is done with [MadMiner](https://github.com/madminer-tool/madminer)

1. `01_setup_morphing_basis.ipynb`: specify the SMEFT operators (in the SMEFT@NLO basis) and define a set of "benchmark points". For every event generated in MadGraph, weights corresponding to each benchmark will be computed. 

2. `02_generate_events.py`: generate the events with MadGraph. MadGraph proc., MadSpin, and Pythia cards can be found in the `cards` folder. Actual run cards are in `cards/run_cards`.

There are run cards for both the 14 TeV and 100 TeV collider setups. Signal runs cards have no cuts on the decay products, since MadGraph is only used for the $gg \rightarrow hh$ decays, and MadSpin in used for the higgs decays. Background run cards have stricted angular and mass window cuts corresponding to those specified in the main paper. 

The script can be run with various flags set. As an example, you could generate events at the non-SM banchmark 2 by running `python 02_generate_events.py -supp -supp_id 2`