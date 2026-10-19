# protein-complex-ml
This repository contains code for the paper "Building from the ground up: augmentation strategies to improve protein language model performance on protein sets" [LINK HERE]. It is organized in the following way.

# Src

This contains source code for the models and data processing used in the paper, as well as functions for training and evaluating our models. The readme in there contains further information on each of the scripts and the functions they contain. 

# Scripts

This contains the scripts used in the ablation, generation, and augmentation protocols described in the Methods of our paper, as well as scripts for producing the figures shown in the main text and SI of the associated paper. The folders within are as follows:
- aminobert_runs, esm2_runs, protbert_runs: These scripts carry out training of each of the models tested and the ablation procedures performed for each trial.
- make_joblibs: Sequence embeddings for each of the DIPS-Plus complexes were calculated to reduce compute load when training the models
- similarity_data: Analyzes the CKA simlarities for the ablation trials
- data_augmentation_runs, augmentation_size_effects: Compares model training performance with and without augmentation with generated complexes

