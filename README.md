# protein-complex-ml
This repository contains code for the paper [NAME HERE] [LINK HERE]. It is organized in the following way.

# Data
This contains the datasets used in the paper, taken from [DIPS-PLUS ZENODO LINK]. These files are organized as provided by the authors of that paper, with separate .dill files for each complex organized according to the second and third characters of their PDB IDs. These .dill files contains the residue-residue contacts as well as DataFrames containing information on atomic coordinates and other data. We also provide a folder detailing the precise ablations performed throughout the paper, and the metrics used to ablate the dataset.

# Src

This contains source code for the models and data processing used in the paper, as well as functions for training and evaluating our models. The readme in there contains further information on each of the scripts and the functions they contain. 

# Scripts

This contains the scripts used in the ablation, generation, and augmentation protocols described in the Methods of our paper, as well as scripts for producing the figures shown in the main text and SI of the associated paper. The folders within are as follows:
- aminobert_runs, esm2_runs, protbert_runs: These scripts carry out training of each of the models tested and the ablation procedures performed for each trial.
- make_joblibs: Sequence embeddings for each of the DIPS-Plus complexes were calculated to reduce compute load when training the models
- similarity_data: Analyzes the CKA simlarities for the ablation trials
- data_augmentation_runs, augmentation_size_effects: Compares model training performance with and without augmentation with generated complexes

