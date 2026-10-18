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

MAX_SEQ_LEN = 650
#


def weight(a, b):
    return (a+b)/2

def SAS(A, B):
    assert A.shape == B.shape, 'matrices must be same shape to compute SAS'
    
    if A.shape[0] > A.shape[1]:
        A = A.T
        B = B.T
    
    A_U, A_S, A_Vh = sp.linalg.svd(A) # returns singular values in decreasing order
    B_U, B_S, B_Vh = sp.linalg.svd(B)

    top_frac = []
    denominator = np.sum([weight(A_S[i], B_S[i]) for i in range(A_S.shape[0])])
    
    for i in range(A_S.shape[0]):
        singvecA_U = A_U[:,i]
        singvecB_U = B_U[:,i]

        singvecA_V = A_Vh.T[:,i]
        singvecB_V = B_Vh.T[:,i]

        pos_U_angle = np.arccos(np.dot(singvecA_U, singvecB_U))
        neg_U_angle = np.arccos(np.dot(singvecA_U, -singvecB_U))
        
        pos_V_angle = np.arccos(np.dot(singvecA_V, singvecB_V))
        neg_V_angle = np.arccos(np.dot(singvecA_V, -singvecB_V))
        singular_angle = np.min([(pos_U_angle + pos_V_angle) / 2, (neg_U_angle + neg_V_angle) / 2])
        
        Δi = 1 - (2*singular_angle/np.pi)
        w = weight(A_S[i], B_S[i])
        top_frac.append(w*Δi)
    numerator = np.sum(top_frac)
    return numerator/denominator
#



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', '-s', help='which set to compare contact maps', type=str, required=True, choices=['train', 'val', 'holdout'])
    parser.add_argument('--predictionsA', '-A', help='model A', type=str, required=True)
    parser.add_argument('--predictionsB', '-B', help='model B', type=str, required=True)
    parser.add_argument('--predictionsC', '-C', help='model C', type=str, required=False, default='')
    parser.add_argument('--keys', '-k', help='keys', required=False, default='', type=str)
    parser.add_argument('--meta', '-m', help='dataset metadata', type=str, required=True)
    parser.add_argument('--output', '-o', help='output prefix', type=str, required=True)
    

    
    args = parser.parse_args()
    
    dips_meta = pd.read_csv(args.meta, header=0, index_col=0, sep=' ')
    #
    # if args.set == 'holdout':
    #     with open(f'{model_dir}/holdout_val_keys.txt', 'r') as f:
    #         lines = f.readlines()[1:]  #skip first line its just a comment
    #         keys = [line.strip() for line in lines]
    # else:
    keys = list(pd.read_csv(args.keys, header=0, index_col=0, sep=' ').index) # using a file with sampled keys instead of the entire set
    
    
    modelA_cmps = np.load(args.predictionsA)[args.set]
    modelB_cmps = np.load(args.predictionsB)[args.set]
    if args.predictionsC != '':
        modelC_cmps = np.load(args.predictionsC)[args.set]
    
    similarities_sas = []
    similarities_linear_cka = []
    #similarities_riemann = []
    #similarities_procrustes = []
    similarities_frob = []
    similarities_nuc = []
    
    #
    
    for i in tqdm(range(modelA_cmps.shape[0]), ascii=True):
        
        complx = keys[i]
        l_len = dips_meta.loc[complx, 'l_len']
        r_len = dips_meta.loc[complx, 'r_len']
        A_cmp = modelA_cmps[i, :l_len, :r_len] # don't want to include padding 0's in similarity computations so slice out only the grid of probabilities
        B_cmp = modelB_cmps[i, :l_len, :r_len]
        #
        
        if args.predictionsC != '':
            
            C_cmp = modelC_cmps[i, :l_len, :r_len]
            similarity_sas =                 matrix_compare.similarity_compute(A_cmp, B_cmp, C_cmp, f=sas.compare)
            similarity_sas =                 np.mean(similarity_sas, axis=0)

            similarity_linear_cka =          matrix_compare.similarity_compute(A_cmp, B_cmp, C_cmp, f=matrix_compare.linear_cka)
            similarity_linear_cka =          np.mean(similarity_linear_cka, axis=0)

            #similarity_procrustes_distance = matrix_compare.similarity_compute(A_cmp, B_cmp, C_cmp, f=matrix_compare.procrustes_size_shape_distance)
            #similarity_procrustes_distance = np.mean(similarity_procrustes_distance, axis=0)

            #similarity_riemann =             matrix_compare.similarity_compute(A_cmp, B_cmp, C_cmp, f=matrix_compare.riemannian_shape_distance)
            #similarity_riemann =             np.mean(similarity_riemann, axis=0)

            similarity_frob =                matrix_compare.frob_similarity(   A_cmp, B_cmp, C_cmp)
            similarity_frob =                np.mean(similarity_frob, axis=0)
            
            similarity_nuc =                 matrix_compare.nuc_similarity(    A_cmp, B_cmp, C_cmp)
            similarity_nuc =                 np.mean(similarity_nuc, axis=0)
            #
        else:
            
            similarity_sas =                 matrix_compare.similarity_compute(A_cmp, B_cmp, f=sas.compare)
            similarity_sas =                 np.mean(similarity_sas, axis=0)
            
            similarity_linear_cka =          matrix_compare.similarity_compute(A_cmp, B_cmp, f=matrix_compare.linear_cka)
            similarity_linear_cka =          np.mean(similarity_linear_cka, axis=0)
            
            #similarity_procrustes_distance = matrix_compare.similarity_compute(A_cmp, B_cmp, f=matrix_compare.procrustes_size_shape_distance)
            #similarity_procrustes_distance = np.mean(similarity_procrustes_distance, axis=0)
            
            #similarity_riemann =             matrix_compare.similarity_compute(A_cmp, B_cmp, f=matrix_compare.riemannian_shape_distance)
            #similarity_riemann =             np.mean(similarity_riemann, axis=0)
            
            similarity_frob =                matrix_compare.frob_similarity(   A_cmp, B_cmp)
            similarity_frob =                np.mean(similarity_frob, axis=0)
            
            similarity_nuc =                 matrix_compare.nuc_similarity(    A_cmp, B_cmp)
            similarity_nuc =                 np.mean(similarity_nuc, axis=0)
            #
        
        similarities_sas.append(similarity_sas)
        similarities_linear_cka.append(similarity_linear_cka)
        #similarities_procrustes.append(similarity_procrustes_distance)
        #similarities_riemann.append(similarity_riemann)
        similarities_frob.append(similarity_frob)
        similarities_nuc.append(similarity_nuc)
        #
    #
    
    #similarities = pd.DataFrame(index=keys, data={"SAS":similarities_sas, "LinearCKA":similarities_linear_cka, "Frobenius":similarities_frob, \
    #                                              "Nuclear":similarities_nuc, "Procrustes":similarities_procrustes, "Riemann":similarities_riemann})
    similarities = pd.DataFrame(index=keys, data={"SAS":similarities_sas, "LinearCKA":similarities_linear_cka, "Frobenius":similarities_frob, \
                                                  "Nuclear":similarities_nuc})
    similarities.to_csv(args.output, sep=' ', index=True, header=True)
    #
    sys.exit()

