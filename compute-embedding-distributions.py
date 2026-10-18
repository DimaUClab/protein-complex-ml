import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import math
from itertools import combinations
import sys
import scipy
from tqdm import tqdm
import gc
import argparse

model_dir = os.environ["PROJ_CM_MODELS"]
data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
similarity_dir = os.path.join(data_dir, "intermediates", "similarity_data")


MAX_SEQ_LEN = 650
# key lists for indexing into dips dataframe
with open(os.path.join(model_dir, 'holdout_val_keys.txt'), 'r') as f:
    lines = f.readlines()[1:]
    holdout_keys = [line.strip() for line in lines]

dips_meta = pd.read_csv(os.path.join(data_dir, "dips_meta.dat"), header=0, index_col=0, sep=' ')


parser = argparse.ArgumentParser()


parser.add_argument("--distribution", "-d", help="Which distribution to compute.", type=str, required=True)
parser.add_argument("--embedding-model", "-m", help="which model was used to compute embeddings", type=str, required=True, choices=["esmc-6b-2024-12", "esmc-600m-2024-12"])
parser.add_argument("--generated-dir", "-e", help="location of generated proteins", required=False, type=str)
parser.add_argument("--out-file", "-o", help="output filename", required=True, type=str)
args = parser.parse_args()

# out_dir = os.path.dirname(args.distribution)
# args.distribution = os.path.basename(args.distribution)
# replicate = os.path.split(out_dir)[1]
# args.generated_proteins = os.path.join(args.generated_proteins, replicate)
# print(args.generated_proteins)
embeddings = joblib.load(os.path.join(data_dir, "intermediates", f"dips_plus-holdout_val-complex-embeddings-{args.embedding_model}.joblib"))


s1_encodings = joblib.load(os.path.join(model_dir, 'esm_s1_encodings_dipsplus.joblib'))
s2_encodings = joblib.load(os.path.join(model_dir, 'esm_s2_encodings_dipsplus.joblib'))

with open(os.path.join(model_dir, 'pairs-postprocessed-val.txt'), 'r') as f:
    all_val_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings.keys()]

val_keys = []
for key in all_val_keys:
    if key in holdout_keys:
        continue
    if dips_meta.loc[key, 'l_len'] > MAX_SEQ_LEN or dips_meta.loc[key, 'r_len'] > MAX_SEQ_LEN:
        continue
    val_keys.append(key)



metric = 'cosine'


quantile = 0.20

def embedding_cdist(embeddings, keys_A, keys_B, metric='cosine'):
    embeddings_A = []
    for complx in keys_A:
        mean_embedding = np.mean(embeddings[complx][0,...], axis=0)
        embeddings_A.append(mean_embedding)
    embeddings_A = np.array(embeddings_A)
    embeddings_B = []
    for complx in keys_B:
        mean_embedding = np.mean(embeddings[complx][0,...], axis=0)
        embeddings_B.append(mean_embedding)
    embeddings_B = np.array(embeddings_B)
    dist_vec = scipy.spatial.distance.cdist(embeddings_A, embeddings_B, metric=metric).flatten()
    return dist_vec


def removed_pdist(embeddings, keys, metric='cosine'):
    embeddings_ = []
    for complx in keys:
        mean_embedding = np.mean(embeddings[complx][0,...], axis=0)
        embeddings_.append(mean_embedding)
    embeddings_ = np.array(embeddings_)
    dist_vec = scipy.spatial.distance.pdist(embeddings_, metric=metric).flatten()
    return dist_vec

trials = list(range(5))
if args.distribution == f"removed_vs_dissimilar":
    collected_dist_vecs = {}
    for i, ablation_type in enumerate(['uniform', 'targeted']):
        collected_dist_vecs[ablation_type] = dict()
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
            collected_dist_vecs[ablation_type][ablation_p] = list()
            for trial in trials:
                with open(f'{model_dir}/reductions_aminobert/{ablation_p}/{ablation_type}/trial_{trial}/removed_val_keys.txt', 'r') as f:
                    trial_removed_val_keys = [line.strip() for line in f.readlines()]
                    #
                
                similarity_table = pd.read_csv(f"{similarity_dir}/similarities/aminobert-esm2-{ablation_p}-{ablation_type}-trial_{trial}-val-similarities.dat", header=0, index_col=0, sep=" ")
                similarity_table = similarity_table.sort_values(by="LinearCKA", ascending=True)
                dissimilar_quantile = np.quantile(similarity_table['LinearCKA'], quantile, axis=None, method='linear')
                trial_dissimilar = list(similarity_table[similarity_table['LinearCKA'] <= dissimilar_quantile].index)
                collected_dist_vecs[ablation_type][ablation_p].append(embedding_cdist(embeddings, trial_removed_val_keys, trial_dissimilar))
    
    joblib.dump(collected_dist_vecs, f"{args.out_file}", compress=("zlib", 3))

if args.distribution == f"removed_vs_similar":
    collected_dist_vecs = {}
    for i, ablation_type in enumerate(['uniform', 'targeted']):
        collected_dist_vecs[ablation_type] = dict()
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
            collected_dist_vecs[ablation_type][ablation_p] = list()
            for trial in trials:
                with open(f'{model_dir}/reductions_aminobert/{ablation_p}/{ablation_type}/trial_{trial}/removed_val_keys.txt', 'r') as f:
                    trial_removed_val_keys = [line.strip() for line in f.readlines()]
                
                similarity_table = pd.read_csv(f"{similarity_dir}/similarities/aminobert-esm2-{ablation_p}-{ablation_type}-trial_{trial}-val-similarities.dat", header=0, index_col=0, sep=" ")
                similarity_table = similarity_table.sort_values(by="LinearCKA", ascending=False)
                similar_quantile = np.quantile(similarity_table['LinearCKA'], 1-quantile, axis=None, method='linear')
                trial_similar = list(similarity_table[similarity_table['LinearCKA'] > similar_quantile].index)
                collected_dist_vecs[ablation_type][ablation_p].append(embedding_cdist(embeddings, trial_removed_val_keys, trial_similar))
                
    joblib.dump(collected_dist_vecs, f"{args.out_file}", compress=("zlib", 3))




gc.collect()


# if args.distribution == f"collated_removed_vs_dissimilar_{metric}.joblib":
#     quantile = 0.2
#     collected_dist_vecs = {}
#     for i, ablation_type in enumerate(['uniform', 'targeted']):
#         collected_dist_vecs[ablation_type] = dict()
#         for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
#             collected_dist_vecs[ablation_type][ablation_p] = list()
#             collated_removed_keys = []
#             for trial in range(5):
#                 with open(f'{model_dir}/reductions_aminobert/{ablation_p}/{ablation_type}/trial_{trial}/removed_val_keys.txt', 'r') as f:
#                     trial_removed_val_keys = [line.strip() for line in f.readlines()]
#                 collated_removed_keys.extend(trial_removed_val_keys)
#             collated_removed_keys = list(set(collated_removed_keys))
#             dissimilar_complexes = pd.read_csv(f"{similarity_dir}/dissimilar_complexes/aminobert-esm2-{ablation_p}-{ablation_type}-val-dissimilar_complexes.dat", header=0, index_col=0, sep=" ")
            
#             collected_dist_vecs[ablation_type][ablation_p].append(embedding_cdist(embeddings, collated_removed_keys, list(dissimilar_complexes.index)))
    
#     joblib.dump(collected_dist_vecs, f"{out_dir}/collated_removed_vs_dissimilar_{metric}.joblib", compress=("zlib", 3))

# if args.distribution == f"collated_removed_vs_similar_{metric}.joblib":
#     collected_dist_vecs = {}
#     for i, ablation_type in enumerate(['uniform', 'targeted']):
#         collected_dist_vecs[ablation_type] = dict()
#         for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
#             collected_dist_vecs[ablation_type][ablation_p] = list()
#             collated_removed_keys = []
#             for trial in range(5):
#                 with open(f'{model_dir}/reductions_aminobert/{ablation_p}/{ablation_type}/trial_{trial}/removed_val_keys.txt', 'r') as f:
#                     trial_removed_val_keys = [line.strip() for line in f.readlines()]
#                 collated_removed_keys.extend(trial_removed_val_keys)
#             collated_removed_keys = list(set(collated_removed_keys))
#             similar_complexes = pd.read_csv(f"{similarity_dir}/similar_complexes/aminobert-esm2-{ablation_p}-{ablation_type}-val-similar_complexes.dat", header=0, index_col=0, sep=" ")
            
#             collected_dist_vecs[ablation_type][ablation_p].append(embedding_cdist(embeddings, collated_removed_keys, list(similar_complexes.index)))
    
#     joblib.dump(collected_dist_vecs, f"{out_dir}/collated_removed_vs_similar_{metric}.joblib", compress=("zlib", 3))

# gc.collect()


if args.distribution == f"dissimilar_pdist":
    collected_dist_vecs = {}
    for i, ablation_type in enumerate(['uniform', 'targeted']):
        collected_dist_vecs[ablation_type] = dict()
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
            collected_dist_vecs[ablation_type][ablation_p] = list()
            for trial in trials:
                similarity_table = pd.read_csv(f"{similarity_dir}/similarities/aminobert-esm2-{ablation_p}-{ablation_type}-trial_{trial}-val-similarities.dat", header=0, index_col=0, sep=" ")
                dissimilar_quantile = np.quantile(similarity_table['LinearCKA'], quantile, axis=None, method='linear')
                trial_dissimilar = list(similarity_table[similarity_table['LinearCKA'] <= dissimilar_quantile].index)
                embeddings_dissimilar = []
                for complx in trial_dissimilar:
                    mean_embedding = np.mean(embeddings[complx][0,...], axis=0)
                    embeddings_dissimilar.append(mean_embedding)
                embeddings_dissimilar = np.array(embeddings_dissimilar)
                dist_mat = scipy.spatial.distance.pdist(embeddings_dissimilar, metric=metric)
                collected_dist_vecs[ablation_type][ablation_p].append(dist_mat.flatten())
    
    joblib.dump(collected_dist_vecs, f"{args.out_file}", compress=("zlib", 3))

if args.distribution == f"similar_pdist":
    collected_dist_vecs = {}
    for i, ablation_type in enumerate(['uniform', 'targeted']):
        collected_dist_vecs[ablation_type] = dict()
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
            collected_dist_vecs[ablation_type][ablation_p] = list()
            for trial in trials:
                similarity_table = pd.read_csv(f"{similarity_dir}/similarities/aminobert-esm2-{ablation_p}-{ablation_type}-trial_{trial}-val-similarities.dat", header=0, index_col=0, sep=" ")
                similar_quantile = np.quantile(similarity_table['LinearCKA'], 1-quantile, axis=None, method='linear')
                trial_similar = list(similarity_table[similarity_table['LinearCKA'] > similar_quantile].index)
                embeddings_similar = []
                for complx in trial_similar:
                    mean_embedding = np.mean(embeddings[complx][0,...], axis=0)
                    embeddings_similar.append(mean_embedding)
                embeddings_similar = np.array(embeddings_similar)
                dist_mat = scipy.spatial.distance.pdist(embeddings_similar, metric=metric)
                collected_dist_vecs[ablation_type][ablation_p].append(dist_mat.flatten())
    
    joblib.dump(collected_dist_vecs, f"{args.out_file}", compress=("zlib", 3))
    

if args.distribution == f"removed_pdist":
    collected_dist_vecs = {}
    for i, ablation_type in enumerate(['uniform', 'targeted']):
        collected_dist_vecs[ablation_type] = dict()
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
            removed_keys = []
            for trial in trials:
                with open(f'{model_dir}/reductions_aminobert/{ablation_p}/{ablation_type}/trial_{trial}/removed_val_keys.txt', 'r') as f:
                    trial_removed_val_keys = [line.strip() for line in f.readlines()]
                    removed_keys.append(trial_removed_val_keys)
            collected_dist_vecs[ablation_type][ablation_p] = Parallel(n_jobs=5, return_as='list', verbose=10, require="sharedmem")(delayed(removed_pdist)(embeddings, removed_keys[k], metric) for k in range(len(removed_keys)))
    
    joblib.dump(collected_dist_vecs, f"{args.out_file}", compress=("zlib", 3))


gc.collect()


if args.distribution == f"generated1_pdist":
    collected_dist_vecs = {}
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        collected_dist_vecs[ablation_type] = dict()
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
            collected_dist_vecs[ablation_type][ablation_p] = list()
            for trial in trials:
                embeddings_generated = joblib.load(f"{args.generated_dir}/{ablation_p}-{ablation_type}-trial_{trial}-generated_embeddings.joblib")
                embeddings_selected = []
                for seed_complx in embeddings_generated.keys():
                    complx_generated_embedding_dists = []
                    seed_embedding = np.mean(embeddings[seed_complx][0,...], axis=0)
                    for embedding in embeddings_generated[seed_complx]:
                        m_embedding = np.mean(embedding, axis=0)
                        complx_generated_embedding_dists.append(scipy.spatial.distance.cosine(seed_embedding, m_embedding))
                    #
                    closest_candidate_ndx = np.argsort(complx_generated_embedding_dists)[0]
                    closest_candidate = np.mean(embeddings_generated[seed_complx][closest_candidate_ndx], axis=0)
                    embeddings_selected.append(closest_candidate)
                embeddings_selected = np.array(embeddings_selected)
                dist_mat = scipy.spatial.distance.pdist(embeddings_selected, metric=metric)
                collected_dist_vecs[ablation_type][ablation_p].append(dist_mat.flatten())
    
    joblib.dump(collected_dist_vecs, f"{args.out_file}", compress=("zlib", 3))

gc.collect()
if args.distribution == f"generated1_vs_removed":
    collected_dist_vecs = {}
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        collected_dist_vecs[ablation_type] = dict()
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
            for trial in trials:
                with open(f'{model_dir}/reductions_aminobert/{ablation_p}/{ablation_type}/trial_{trial}/removed_val_keys.txt', 'r') as f:
                    trial_removed_val_keys = [line.strip() for line in f.readlines()]
                embeddings_removed = []
                for complx in trial_removed_val_keys:
                    mean_embedding = np.mean(embeddings[complx][0,...], axis=0)
                    embeddings_removed.append(mean_embedding)
                embeddings_removed = np.array(embeddings_removed)
                
                collected_dist_vecs[ablation_type][ablation_p] = list()
                embeddings_generated = joblib.load(f"{args.generated_dir}/{ablation_p}-{ablation_type}-trial_{trial}-generated_embeddings.joblib")
                embeddings_selected = []
                for seed_complx in embeddings_generated.keys():
                    complx_generated_embedding_dists = []
                    seed_embedding = np.mean(embeddings[seed_complx][0,...], axis=0)
                    for embedding in embeddings_generated[seed_complx]:
                        m_embedding = np.mean(embedding, axis=0)
                        complx_generated_embedding_dists.append(scipy.spatial.distance.cosine(seed_embedding, m_embedding))
                    #
                    closest_candidate_ndx = np.argsort(complx_generated_embedding_dists)[0]
                    closest_candidate = np.mean(embeddings_generated[seed_complx][closest_candidate_ndx], axis=0)
                    embeddings_selected.append(closest_candidate)
                embeddings_selected = np.array(embeddings_selected)
                dist_mat = scipy.spatial.distance.cdist(embeddings_selected, embeddings_removed, metric=metric).flatten()
                collected_dist_vecs[ablation_type][ablation_p].append(dist_mat.flatten())
    
    joblib.dump(collected_dist_vecs, f"{args.out_file}", compress=("zlib", 3))


#gc.collect()

# if args.distribution == f"concatenated_generated1_vs_removed":
#     collected_dist_vecs = {}
#     for i, ablation_type in enumerate(['uniform', 'targeted']):
#         collected_dist_vecs[ablation_type] = dict()
#         for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
#             embeddings_removed = []
#             val_removed_s = []
#             removed_keys = []
#             for fname in [f'{model_dir}/reductions_aminobert/{ablation_p}/{ablation_type}/trial_{trial}/removed_val_keys.txt' for trial in range(5)]:
#                 with open(fname, 'r') as f:
#                     new_removed_keys = [line.strip() for line in f.readlines()]
#                 removed_keys.extend(new_removed_keys)
#             removed_keys = list(set(removed_keys))
#             for complx in removed_keys:
#                 mean_embedding = np.mean(embeddings[complx][0,...], axis=0)
#                 embeddings_removed.append(mean_embedding)
#             embeddings_removed = np.array(embeddings_removed)
            
#             collected_dist_vecs[ablation_type][ablation_p] = list()
#             embeddings_generated_all = joblib.load(f"{args.generated_proteins}/{args.embedding_model}/esmc-val-{ablation_p}-{ablation_type}-generated_embeddings.joblib")
#             embeddings_generated = []
#             for complx in embeddings_generated_all.keys():
#                 dissimilar_embedding = np.mean(embeddings[complx][0,...], axis=0)
#                 mean_embeddings = [np.mean(embeddings_generated_all[complx][trial], axis=0) for trial in range(len(embeddings_generated_all[complx]))]
#                 dists = [scipy.spatial.distance.cosine(dissimilar_embedding, mean_embeddings[trial]) for trial in range(len(embeddings_generated_all[complx]))]
#                 embeddings_generated.append(mean_embeddings[np.argsort(dists)[0]])
#             dist_mat = scipy.spatial.distance.cdist(embeddings_generated, embeddings_removed, metric=metric)
#             collected_dist_vecs[ablation_type][ablation_p].append(dist_mat.flatten())
    
#     joblib.dump(collected_dist_vecs, f"{out_dir}/generated1_vs_removed_{metric}.joblib", compress=("zlib", 3))


gc.collect()


if args.distribution == f"generated1_vs_dissimilar":
    collected_dist_vecs = {}
    for i, ablation_type in enumerate(['uniform', 'targeted']):
        collected_dist_vecs[ablation_type] = dict()
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
            for trial in trials:
                embeddings_dissimilar = []
                similarity_table = pd.read_csv(f"{similarity_dir}/similarities/aminobert-esm2-{ablation_p}-{ablation_type}-trial_{trial}-val-similarities.dat", header=0, index_col=0, sep=" ")
                dissimilar_quantile = np.quantile(similarity_table['LinearCKA'], quantile, axis=None, method='linear')
                trial_dissimilar = list(similarity_table[similarity_table['LinearCKA'] <= dissimilar_quantile].index)
                for complx in trial_dissimilar:
                    mean_embedding = np.mean(embeddings[complx][0,...], axis=0)
                    embeddings_dissimilar.append(mean_embedding)
                embeddings_dissimilar = np.array(embeddings_dissimilar)
                
                collected_dist_vecs[ablation_type][ablation_p] = list()
                embeddings_generated = joblib.load(f"{args.generated_dir}/{ablation_p}-{ablation_type}-trial_{trial}-generated_embeddings.joblib")
                embeddings_selected = []
                for seed_complx in embeddings_generated.keys():
                    complx_generated_embedding_dists = []
                    seed_embedding = np.mean(embeddings[seed_complx][0,...], axis=0)
                    for embedding in embeddings_generated[seed_complx]:
                        m_embedding = np.mean(embedding, axis=0)
                        complx_generated_embedding_dists.append(scipy.spatial.distance.cosine(seed_embedding, m_embedding))
                    #
                    closest_candidate_ndx = np.argsort(complx_generated_embedding_dists)[0]
                    closest_candidate = np.mean(embeddings_generated[seed_complx][closest_candidate_ndx], axis=0)
                    embeddings_selected.append(closest_candidate)
                embeddings_selected = np.array(embeddings_selected)
                dist_mat = scipy.spatial.distance.cdist(embeddings_selected, embeddings_dissimilar, metric=metric).flatten()
                collected_dist_vecs[ablation_type][ablation_p].append(dist_mat.flatten())
    
    joblib.dump(collected_dist_vecs, f"{args.out_file}", compress=("zlib", 3))

if args.distribution == f"generated1_vs_similar":
    collected_dist_vecs = {}
    for i, ablation_type in enumerate(['uniform', 'targeted']):
        collected_dist_vecs[ablation_type] = dict()
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80", "interfacearea20", "interfacearea50", "interfacearea80"]):
            for trial in trials:
                embeddings_similar = []
                similarity_table = pd.read_csv(f"{similarity_dir}/similarities/aminobert-esm2-{ablation_p}-{ablation_type}-trial_{trial}-val-similarities.dat", header=0, index_col=0, sep=" ")
                similar_quantile = np.quantile(similarity_table['LinearCKA'], 1-quantile, axis=None, method='linear')
                trial_similar = list(similarity_table[similarity_table['LinearCKA'] > similar_quantile].index)
                for complx in trial_similar:
                    mean_embedding = np.mean(embeddings[complx][0,...], axis=0)
                    embeddings_similar.append(mean_embedding)
                embeddings_similar = np.array(embeddings_similar)
                
                collected_dist_vecs[ablation_type][ablation_p] = list()
                embeddings_generated = joblib.load(f"{args.generated_dir}/{ablation_p}-{ablation_type}-trial_{trial}-generated_embeddings.joblib")
                embeddings_selected = []
                for seed_complx in embeddings_generated.keys():
                    complx_generated_embedding_dists = []
                    seed_embedding = np.mean(embeddings[seed_complx][0,...], axis=0)
                    for embedding in embeddings_generated[seed_complx]:
                        m_embedding = np.mean(embedding, axis=0)
                        complx_generated_embedding_dists.append(scipy.spatial.distance.cosine(seed_embedding, m_embedding))
                    #
                    closest_candidate_ndx = np.argsort(complx_generated_embedding_dists)[0]
                    closest_candidate = np.mean(embeddings_generated[seed_complx][closest_candidate_ndx], axis=0)
                    embeddings_selected.append(closest_candidate)
                embeddings_selected = np.array(embeddings_selected)
                dist_mat = scipy.spatial.distance.cdist(embeddings_selected, embeddings_similar, metric=metric).flatten()
                collected_dist_vecs[ablation_type][ablation_p].append(dist_mat.flatten())
    
    joblib.dump(collected_dist_vecs, f"{args.out_file}", compress=("zlib", 3))
