import numpy as np
import scipy as sp
import joblib
import sys
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sas
import matrix_compare

model_dir = os.environ["PROJ_CM_MODELS"]
data_dir = os.environ["PROJ_DATASETS"]
MAX_SEQ_LEN = 650
#


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', '-s', help='which set to compare contact maps', type=str, required=True, choices=['train', 'val', "holdout"])
    parser.add_argument('--predictionsA', '-A', help='model A', type=str, required=True)
    parser.add_argument('--predictionsB', '-B', help='model B', type=str, required=True)
    parser.add_argument('--predictionsC', '-C', help='model C', type=str, required=False, default='')
    parser.add_argument('--output', '-o', help='output prefix', type=str, required=True)
    parser.add_argument('--parameters', '-p', help='parameters file', type=str, required=True)
    parser.add_argument('--index', '-i', help='index of parameter set in file', type=int, required=True)
    parser.add_argument('--append', '-a', help="whether to append to existing file or overwrite", type=int, default=1, required=False)
    
    args = parser.parse_args()
    dips_meta = pd.read_csv(f"{data_dir}/dips_meta.dat", header=0, index_col=0, sep=' ')
    #

    
    s1_encodings_esm2 = joblib.load(f'{model_dir}/esm_s1_encodings_dipsplus.joblib')
    s2_encodings_esm2 = joblib.load(f'{model_dir}/esm_s2_encodings_dipsplus.joblib')
    keys = []
    if args.set == "train":
        parameter_sets = pd.read_csv(args.parameters, index_col=0, header=0, sep='\t')
        parameters = parameter_sets.loc[args.index, 'parameters'].split(' ')
        reduction = f"{parameters[1]}/{parameters[2]}"
        trial = parameters[3]
        with open(f'{model_dir}/reductions_aminobert/{reduction}/{trial}/removed_train_keys.txt', 'r') as f:
            removed_train_keys = [line.strip() for line in f.readlines()]
        with open(f'{model_dir}/pairs-postprocessed-train-duplicates-removed.txt', 'r') as f:
            all_train_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings_esm2.keys()]
        #
        for key in all_train_keys:
            if key in removed_train_keys:
                continue
            if dips_meta.loc[key, 'l_len'] > MAX_SEQ_LEN or dips_meta.loc[key, 'r_len'] > MAX_SEQ_LEN:
                continue
            keys.append(key)
    elif args.set == "val":
        parameter_sets = pd.read_csv(args.parameters, index_col=0, header=0, sep='\t')
        parameters = parameter_sets.loc[args.index, 'parameters'].split(' ')
        reduction = f"{parameters[1]}/{parameters[2]}"
        trial = parameters[3]
        with open(f'{model_dir}/reductions_aminobert/{reduction}/{trial}/removed_val_keys.txt', 'r') as f:
            removed_val_keys = [line.strip() for line in f.readlines()]
        with open(f'{model_dir}/pairs-postprocessed-val.txt', 'r') as f:
            all_val_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings_esm2.keys()]
        #
        with open(f'{model_dir}/holdout_val_keys.txt', 'r') as f:
            lines = f.readlines()[1:]  #skip first line its just a comment
            holdout_keys = [line.strip() for line in lines]
        #
        
        for key in all_val_keys:
            if key in removed_val_keys:
                continue
            if key in holdout_keys:
                continue
            if dips_meta.loc[key, 'l_len'] > MAX_SEQ_LEN or dips_meta.loc[key, 'r_len'] > MAX_SEQ_LEN:
                continue
            keys.append(key)
        #
    elif args.set == "holdout":
        with open(f'{model_dir}/holdout_val_keys.txt', 'r') as f:
            lines = f.readlines()[1:]  #skip first line its just a comment
            keys = [line.strip() for line in lines]
        #
    modelA_cmps = np.load(args.predictionsA)[args.set]
    modelB_cmps = np.load(args.predictionsB)[args.set]
    if args.predictionsC != '':
        modelC_cmps = np.load(args.predictionsC)[args.set]

    if args.append:
        similarities = pd.read_csv(args.output, header=0, index_col=0, sep=' ')
    else:
        #similarities = pd.DataFrame(columns=["SAS", "LinearCKA", "Frobenius", "Nuclear"])
        similarities = pd.DataFrame(columns=["LinearCKA"])
    
    for i in tqdm(range(modelA_cmps.shape[0]), ascii=True):
        complx = keys[i]
        if complx in similarities.index:
            # already computed, skip
            continue
        l_len = dips_meta.loc[complx, 'l_len']
        r_len = dips_meta.loc[complx, 'r_len']
        A_cmp = modelA_cmps[i, :l_len, :r_len]
        B_cmp = modelB_cmps[i, :l_len, :r_len]
        #
        
        if args.predictionsC != '':
            
            C_cmp = modelC_cmps[i, :l_len, :r_len]
            #similarity_sas =                 matrix_compare.similarity_compute(A_cmp, B_cmp, C_cmp, f=sas.compare)
            #similarity_sas =                 np.mean(similarity_sas, axis=0)

            similarity_linear_cka =          matrix_compare.similarity_compute(A_cmp, B_cmp, C_cmp, f=matrix_compare.linear_cka)
            similarity_linear_cka =          np.mean(similarity_linear_cka, axis=0)

            #similarity_procrustes_distance = matrix_compare.similarity_compute(A_cmp, B_cmp, C_cmp, f=matrix_compare.procrustes_size_shape_distance)
            #similarity_procrustes_distance = np.mean(similarity_procrustes_distance, axis=0)

            #similarity_riemann =             matrix_compare.similarity_compute(A_cmp, B_cmp, C_cmp, f=matrix_compare.riemannian_shape_distance)
            #similarity_riemann =             np.mean(similarity_riemann, axis=0)

            #similarity_frob =                matrix_compare.frob_similarity(   A_cmp, B_cmp, C_cmp)
            #similarity_frob =                np.mean(similarity_frob, axis=0)
            
            #similarity_nuc =                 matrix_compare.nuc_similarity(    A_cmp, B_cmp, C_cmp)
            #similarity_nuc =                 np.mean(similarity_nuc, axis=0)
            #
        else:
            
            #similarity_sas =                 matrix_compare.similarity_compute(A_cmp, B_cmp, f=sas.compare)
            #similarity_sas =                 np.mean(similarity_sas, axis=0)
            
            similarity_linear_cka =          matrix_compare.similarity_compute(A_cmp, B_cmp, f=matrix_compare.linear_cka)
            similarity_linear_cka =          np.mean(similarity_linear_cka, axis=0)
            
            #similarity_procrustes_distance = matrix_compare.similarity_compute(A_cmp, B_cmp, f=matrix_compare.procrustes_size_shape_distance)
            #similarity_procrustes_distance = np.mean(similarity_procrustes_distance, axis=0)
            
            #similarity_riemann =             matrix_compare.similarity_compute(A_cmp, B_cmp, f=matrix_compare.riemannian_shape_distance)
            #similarity_riemann =             np.mean(similarity_riemann, axis=0)
            
            #similarity_frob =                matrix_compare.frob_similarity(   A_cmp, B_cmp)
            #similarity_frob =                np.mean(similarity_frob, axis=0)
            
            #similarity_nuc =                 matrix_compare.nuc_similarity(    A_cmp, B_cmp)
            #similarity_nuc =                 np.mean(similarity_nuc, axis=0)
            #
        
        #similarities.loc[complx] = [similarity_sas, similarity_linear_cka, similarity_frob, similarity_nuc]
        similarities.loc[complx] = [similarity_linear_cka]
        #
    #
    
    #similarities = pd.DataFrame(index=keys, data={"SAS":similarities_sas, "LinearCKA":similarities_linear_cka, "Frobenius":similarities_frob, \
    #                                              "Nuclear":similarities_nuc, "Procrustes":similarities_procrustes, "Riemann":similarities_riemann})
    #similarities = pd.DataFrame(index=keys, data={"SAS":similarities_sas, "LinearCKA":similarities_linear_cka, "Frobenius":similarities_frob, \
    #                                              "Nuclear":similarities_nuc})
    similarities.to_csv(args.output, sep=' ', index=True, header=True)
    #
    sys.exit()

