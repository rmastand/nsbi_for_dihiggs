# Neural Simulation-based Inference: Full Analysis Pipeline
This is the repository for the code used in paper "Constraining the Higgs Potential with Neural Simulation-based Inference for Di-Higgs Production" (https://arxiv.org/abs/2405.15847)

Authors: Radha Mastandrea, Benjamin Nachman, and Tilman Plehn

Dataset: https://zenodo.org/records/11222924

### General version notes
- Use Python 3.8 and MadGraph 3.5.1 with this repository. 
- Within your MadGraph installation, you will need LHAPDF and the [SMEFT@NLO model](https://feynrules.irmp.ucl.ac.be/wiki/SMEFTatNLO)


## Analysis flow

The file `workflow.yaml` is used for I/O management, so such folders don't have to be defined in every script.


### Event generation
Most of the event generation is done with [MadMiner](https://github.com/madminer-tool/madminer)

1. `01_setup_morphing_basis.ipynb`: specify the SMEFT operators (in the SMEFT@NLO basis) and define a set of "benchmark points". For every event generated in MadGraph, weights corresponding to each benchmark will be computed. 

2. `02_generate_events.py`: generate the events with MadGraph. MadGraph commands, MadSpin cards, and Pythia cards can be found in the `cards` folder. MadGraph run cards are in `cards/run_cards`.

   There are run cards for both the 14 TeV and 100 TeV collider setups. Signal runs cards have no cuts on the decay products, since MadGraph is only used for the $gg \rightarrow hh$ decays, and MadSpin in used for the higgs decays. Background run cards have stricter angular and mass window cuts corresponding to those specified in the main paper. 

   The script can be run with various flags set. As an example, you could generate events at the non-SM benchmark 2 by running `python 02_generate_events.py -supp -supp_id 2`. You can specify the desired number of Madgraph runs in the `workflow.yaml`.

3. `03a_read_delphes.py`: Run Delphes on the previously generated files and make selection cuts on the events. 

   This script assumes that you have a specific directory setup, namely that the outputs of step 2 are in `</path_from_workflow_yaml_delphes_input_dir_prefix/process_id/batch_<i>/`. `process_id` is an argument to the script (`signal_sm`, `signal_supp` for non-SM benchmarks, or `background_0`), and the batch is indexed by an integer. That directory can contain any number of Madgraph output directories `run_j`. 

   n.b. This directory setup must be manually created, but I have found that it works well when generating a large number of events. especially when events are generated in parallel on a cluster setup with separate scratch and long-term storage directories. 

   As an example, you could run Delphes and apply kinematic cuts on events from 20 MadGraph runs that have been generated at the non-SM benchmark 2 by running `python 03a_read_delphes.py -p signal_supp -supp_id 2 -b 0 -start 0 -stop 20`. 

   Finally, compile events over batches and all signal benchmarks with `python 03b_compile.py -p signal` and `python 03b_compile.py -p background`.

4. `04_make_samples.ipynb`: generate samples of signal events at arbitrary benchmark points, using MadMiner. These samples will be used for network training and testing. You can generate multiple datasets (identified by `parameter_code`) depending on which SMEFT Wilson coefficients you want to vary.


### Likelihood rato evaluation
5. `05_train_network.py`: Train the neural networks (classifiers). Specify the dataset that you want to run over be changing `sampling.output_dir` in `workflow.yaml` and by providing the correct `parameter_code` for the argument. Both simple dense nets and Bayesian nets are implemented. Network architecture and hyperparameters are hard-coded in the script, but they are all saved out into a config `yaml` with a particular run id (`rid`, specified in the arguments). 

   As described in the paper, there are 3 classifiers that need to be trained: (1) likelihood ratio of BSM signal to SM signal, (2) likelihood ratio of BSM signal to background, (3) likelihood ratio of background to SM signal. 

   Trainable kinematic features are:  $m_{hh}$, $p_{T_{bb}}$, $p_{T_{aa}}$, $\Delta R_{aa}$, $\Delta R_{bb}$, $p_{T_{a0}}$, $p_{T_{a1}}$, $p_{T_{b0}}$, $p_{T_{b1}}$. Train on the first $i$ features with the flag `-f`.

   As an example, you could train classifier 1 on a set of data that only varies over the first Wilson coefficient, using 5 kinematic features, with `python 05_train_network.py -p c0 -rid test_run -f 5 -c1`.

6. `06a_evaluate_test_statistic.ipynb` and `06b_evaluate_coverage.ipynb`: calculate likelihood ratios on previously generated test sets (or multiple likelihood ratios over different test set instatiations). It is possible to ensemble over several networks.

7. `07_nice_plots.ipynb`: nicer plot formatting.

Finally, `visualize_features.ipynb` may be helpful to quickly visualize how kinematic features change as a function of Wilson coefficients.
