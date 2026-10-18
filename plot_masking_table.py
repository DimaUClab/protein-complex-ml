import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import joblib
import scipy
import os
from tqdm import tqdm
import Levenshtein
import math
from itertools import combinations
from joblib import Parallel, delayed

model_dir = os.environ["PROJ_CM_MODELS"]
data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
proj_root = os.environ["PROJ_ROOT"]
int_dir = os.path.join(data_dir, "intermediates")
plt.style.use(os.path.join(proj_root, "stylesheet.mplstyle"))
rcParams["text.usetex"] = False
rcParams["font.sans-serif"] = "FreeSans"
#generated_embeddings = joblib.load(f"{art_dir}/generated_proteins/esmc-val-interfacearea80-uniform-generated_embeddings.joblib")

#pca_model = joblib.load(f"{int_dir}/esmc-embeddings-complex-holdout_val-pca.joblib")

MAX_SEQ_LEN = 650

AA_CODES = """C M F I L V W Y A G T S Q N E D H R K P""".split()
AA_PROBS = np.array([0.06146967, 0.0559202 , 0.05050506, 0.04350903, 0.03991305, \
       0.04594497, 0.05351416, 0.07171458, 0.0413661, 0.0427031, \
       0.05857854, 0.04680383, 0.05046692, 0.05351796, 0.04060263, \
       0.04913945, 0.06421999, 0.04527684, 0.04111708, 0.04371685])

import random
# previously_masked
# 0     ,     1
mask_p = 0
add_p  = 1
del_p  = 2
mut_p  = 3
no_op  = 4
c = [mask_p, add_p, del_p, mut_p, no_op]



probs = [[0.100, 0.30], # mask_p
         [0.010, 0.20], # add_p
         [0.010, 0.10], # del_p
         [0.001, 0.00], # mut_p
         [0.879, 0.40]] # no_op
probs = np.array(probs)

def operate_seq(l):
    previously_masked = 0
    running = True
    l_ndx = 0
    l_out = []
    muts = 0
    while running:
        choice = np.random.choice(c, p=probs[:,previously_masked])
        if choice == mask_p:
            l_out.append('_')
            l_ndx = l_ndx + 1
            previously_masked = 1
        elif choice == add_p:
            l_out.append('_')
            previously_masked = 1
        elif choice == del_p:
            l_ndx = l_ndx + 1
            previously_masked = 0
        elif choice == no_op:
            l_out.append(l[l_ndx])
            l_ndx = l_ndx + 1
            previously_masked = 0
        elif choice == mut_p:
            l_ndx = l_ndx + 1
            l_out.append(get_random_aa())
            previously_masked = 0
            muts += 1
        if l_ndx >= len(l):
            running = False
    return ''.join(l_out), muts


def get_random_aa():
    aa_ndx = np.random.choice(range(len(AA_CODES)), p=AA_PROBS)
    return AA_CODES[aa_ndx]

def scramble_complex(seq):
    split_seq = seq.split('|')
    current_split = len(split_seq[0])
    current_split_ratio = current_split/len(seq) if current_split != -1 else 0.5
    new_l, muts_l = operate_seq(split_seq[0])
    new_r, muts_r = operate_seq(split_seq[1])
    
    if len(new_l) > MAX_SEQ_LEN:
        new_l = new_l[:MAX_SEQ_LEN]
    if len(new_r) > MAX_SEQ_LEN:
        new_r = new_r[:MAX_SEQ_LEN]

    new_complex = list('|'.join([new_l, new_r]))
    num_steps = sum([(lambda x: 1 if x == '_' else 0)(aa) for aa in new_complex])
    muts = muts_l + muts_r
    
    return ''.join(new_complex), num_steps, muts

def get_scrambled_complexes(complex_length):
    new_complexes = []
    complexes_muts = []
    for trial in range(5):
        complex_seq = ['X']*(complex_length+1)
        split_loc = random.randint(5, complex_length-5)
        complex_seq[split_loc] = '|'
        complex_seq = ''.join(complex_seq)
        new_complex, num_steps, muts = scramble_complex(complex_seq)
        new_complex = ''.join(new_complex.split('|')) # get rid of '|' token
        new_complexes.append(new_complex)
        complexes_muts.append(muts)
    return new_complexes, complexes_muts

def distance_group(split):
    dist_vec = []
    for combo in split:
        dist_vec.append(Levenshtein.distance(combo[0], combo[1]))
    return dist_vec

def pdist_levenshtein_distance(seqs):
    combinations_ = combinations(seqs, 2)
    combination_strs = [(a, b) for a, b in combinations_]
    dist_vec = distance_group(combination_strs)
    return np.array(dist_vec).flatten()

from collections import defaultdict
masks_per_length = defaultdict(list)
muts_per_length = defaultdict(list)

complexes_per_length = defaultdict(list)
leven_diff = defaultdict(list)
for i, complex_length in enumerate(range(20, 1301)):
    new_complexes, complexes_muts = get_scrambled_complexes(complex_length)
    for i, complx in enumerate(new_complexes):
        ln = len(complx)
        leven_diff[ln].append(Levenshtein.ratio(['X']*(complex_length+1), complx))
        complexes_per_length[ln].append(complx)
        masks = sum([(lambda x: 1 if x == '_' else 0)(aa) for aa in complx])
        masks_per_length[ln].append(masks)
        muts_per_length[ln].append(complexes_muts[i])

leven_dist = []
for n in leven_diff.keys():
    leven_dist.extend(leven_diff[n])
leven_dist = np.array(leven_dist)
fig, ax = plt.subplots()
ax.hist(leven_dist, bins=20, density=True)
ax.set_xlabel("Levenshtein ratio")
ax.set_ylabel("probability")
fig.savefig("leven_ratio_dist.png")
fig.savefig("leven_ratio_dist.pdf")

mut_means = {}
mut_stds = {}
for total_len in muts_per_length.keys():
    mut_means[total_len] = np.mean(muts_per_length[total_len])
    mut_stds[total_len] = np.std(muts_per_length[total_len])

fig, ax = plt.subplots()
x = np.array(list(muts_per_length.keys()))
ax.scatter(x, list(mut_means.values()), label="mean", color="midnightblue", marker='.')
ax.set_xlabel("protein complex size")
ax.set_ylabel("mean number of mutations")
ax.set_title("Mean number of mutations for generated protein complexes")
ax.legend()
fig.tight_layout()
fig.savefig(f"{art_dir}/masked_mut_means.png", bbox_inches="tight")
fig.savefig(f"{art_dir}/masked_mut_means.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
x = np.array(list(muts_per_length.keys()))
ax.scatter(x, list(mut_stds.values()), label="std dev", color="midnightblue", marker='.')
ax.set_xlabel("protein complex size")
ax.set_ylabel("std dev of number of mutations")
ax.set_title("Standard deviation of mutations for generated protein complexes")
ax.legend()
fig.tight_layout()
fig.savefig(f"{art_dir}/masked_mut_stds.png", bbox_inches="tight")
fig.savefig(f"{art_dir}/masked_mut_stds.pdf", bbox_inches="tight")

levens_per_length = {}
for total_len in complexes_per_length.keys():
    levens_per_length[total_len] = pdist_levenshtein_distance(complexes_per_length[total_len])

leven_means = {}
leven_stds = {}
for total_len in levens_per_length.keys():
    leven_means[total_len] = np.mean(levens_per_length[total_len])
    leven_stds[total_len] = np.std(levens_per_length[total_len])

fig, ax = plt.subplots()
x = np.array(list(levens_per_length.keys()))
ax.scatter(x, list(leven_means.values()), label="mean", color="midnightblue", marker='.')
ax.set_xlabel("protein complex size")
ax.set_ylabel("mean Levenshtein distance")
ax.set_title("Mean Levenshtein distance for generated protein complexes")
ax.legend()
fig.tight_layout()
fig.savefig(f"{art_dir}/masked_leven_means.png", bbox_inches="tight")
fig.savefig(f"{art_dir}/masked_leven_means.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
x = np.array(list(levens_per_length.keys()))
ax.scatter(x, list(leven_stds.values()), label="mean", color="midnightblue", marker='.')
ax.set_xlabel("protein complex size")
ax.set_ylabel("std dev Levenshtein distance")
ax.set_title("Standard deviation of Levenshtein distance for generated protein complexes")
ax.legend()
fig.tight_layout()
fig.savefig(f"{art_dir}/masked_leven_stds.png", bbox_inches="tight")
fig.savefig(f"{art_dir}/masked_leven_stds.pdf", bbox_inches="tight")


lengths = np.array(list(masks_per_length.keys()))
print(np.max(lengths))
print(np.min(lengths))
x = []
average_percent_masked_per_length = []
num_masks_per_length = []
for total_len in range(20, 1301):
    if total_len in masks_per_length.keys():
        m_ = np.array(masks_per_length[total_len])
        num_masks_per_length.append(np.mean(m_))
        
        p = m_/total_len * 100
        average_percent_masked_per_length.append(np.mean(p))
        x.append(total_len)

import matplotlib as mpl

with mpl.rc_context({'ytick.right':True, 'xtick.minor.visible':True, 'ytick.minor.visible':True}):
    fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, sharex=False, sharey=False)
    ax[0,0].plot(x, average_percent_masked_per_length, color='midnightblue')
    ax[0,0].set_xlim(left=20, right=1302)
    ax[0,0].set_ylabel('average %\nmasked residues')
     
    ax[1,0].plot(x, num_masks_per_length, color='midnightblue')
    ax[1,0].set_xlim(left=20, right=1302)
    ax[1,0].set_ylabel("# masks")
    #ax[1,0].axhline(y=100, linestyle='--', color='coral')
    ax[1,0].set_xlabel('protein complex size')
fig.tight_layout()
ax[0,0].set_title('Masking outcomes for probability table')
ax[1,0].set_ylim(bottom=0, top=300)
fig.savefig(f"{art_dir}/prob_masking_table.png", bbox_inches="tight")
fig.savefig(f"{art_dir}/prob_masking_table.pdf", bbox_inches="tight")
