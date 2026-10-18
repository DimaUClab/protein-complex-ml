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
proj_root = os.environ["PROJ_ROOT"]

plt.style.use(f"{proj_root}/stylesheet.mplstyle")
dips_meta = pd.read_csv(f"{data_dir}/dips_meta.dat", header=0, index_col=0, sep=' ')


parser = argparse.ArgumentParser()

parser.add_argument("--distribution", "-d", help="Which distribution to compute.", type=str, required=True)
parser.add_argument("--out-dir", "-o", help="Where to place output figure.", type=str, required=True)
parser.add_argument("--embedding-model", "-m", help="which model was used for embeddings", type=str, required=True)

args = parser.parse_args()
# if "summary" in args.distribution.split("-"):
#     args.replicate = ""
# else:
#     args.replicate = os.path.split(os.path.split(args.out_dir)[0])[1]

# print(args.replicate)

metric = 'cosine'
x = np.linspace(0, 1, num=2000)


colors = {1:"#4477AA", 2:"#EE6677", 3:"#228833"}
kde_color = "red"
med_line_color = "forestgreen"
line_color = "midnightblue"
kde_alpha = 0.2
sample_size = 4000
trials = list(range(5))
figsize = (7, 5)

if args.distribution == f"concatenated_rescount_removed_vs_dissimilar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_dissimilar-cosine.joblib")
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=50, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            xmed = np.median(signal_sampled)
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between dissimilar and ablated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_vs_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_vs_dissimilar_cosine.pdf", bbox_inches="tight")

if args.distribution == f"concatenated_interfacearea_removed_vs_dissimilar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_dissimilar-cosine.joblib")
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            dist_vec = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(dist_vec, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            xmed = np.median(signal_sampled)
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("\\textbf{probability density}")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between dissimilar and ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_vs_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_vs_dissimilar_cosine.pdf", bbox_inches="tight")

if args.distribution == f"concatenated_rescount_removed_vs_similar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_similar-cosine.joblib")
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            xmed = np.median(signal_sampled)
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between similar and ablated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_vs_similar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_vs_similar_cosine.pdf", bbox_inches="tight")

if args.distribution == f"concatenated_interfacearea_removed_vs_similar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_similar-cosine.joblib")
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            dist_vec = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(dist_vec, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            xmed = np.median(signal_sampled)
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between similar and ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_vs_similar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_vs_similar_cosine.pdf", bbox_inches="tight")


if args.distribution == f"overlaid_rescount_removed_vs_dissimilar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_dissimilar-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            for trial in trials:
                signal = collected_dist_vecs[ablation_type][ablation_p][trial]
                signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
                kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
                #ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid')
                y = kernel([x])
                maxy = max(maxy, np.max(y))
                ax[i, j].plot(x, y, linewidth=2, label=f"trial {trial}")
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title("extensive ablation")
    ax[0, 1].set_title("moderate ablation")
    ax[0, 2].set_title("limited ablation")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("overlaid distributions of ablated to dissimilar protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/overlaid_rescount_removed_vs_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/overlaid_rescount_removed_vs_dissimilar_cosine.pdf", bbox_inches="tight")


if args.distribution == f"overlaid_interfacearea_removed_vs_dissimilar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_dissimilar-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            for trial in trials:
                signal = collected_dist_vecs[ablation_type][ablation_p][trial]
                signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
                kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
                #ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid')
                y = kernel([x])
                maxy = max(maxy, np.max(y))
                ax[i, j].plot(x, y, linewidth=2, label=f"trial {trial}")
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title("extensive ablation")
    ax[0, 1].set_title("moderate ablation")
    ax[0, 2].set_title("limited ablation")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("overlaid distributions of ablated to dissimilar protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/overlaid_interfacearea_removed_vs_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/overlaid_interfacearea_removed_vs_dissimilar_cosine.pdf", bbox_inches="tight")





if args.distribution == f"collated_rescount_removed_vs_dissimilar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{args.replicate}/collated_removed_vs_dissimilar_cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = np.array(collected_dist_vecs[ablation_type][ablation_p]).flatten()
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("collated distributions of ablated to dissimilar protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/collated_rescount_removed_vs_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/collated_rescount_removed_vs_dissimilar_cosine.pdf", bbox_inches="tight")

if args.distribution == f"collated_interfacearea_removed_vs_dissimilar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{args.replicate}/collated_removed_vs_dissimilar_cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = np.array(collected_dist_vecs[ablation_type][ablation_p]).flatten()
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("collated distributions of ablated to dissimilar protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/collated_interfacearea_removed_vs_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/collated_interfacearea_removed_vs_dissimilar_cosine.pdf", bbox_inches="tight")








if args.distribution == f"overlaid_rescount_dissimilar_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/dissimilar_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            for trial in trials:
                signal = collected_dist_vecs[ablation_type][ablation_p][trial]
                signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
                kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
                #ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid')
                y = kernel([x])
                maxy = max(maxy, np.max(y))
                ax[i, j].plot(x, y, linewidth=2, label=f"trial {trial}")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title("extensive ablation")
    ax[0, 1].set_title("moderate ablation")
    ax[0, 2].set_title("limited ablation")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("overlaid distributions of dissimilar protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/overlaid_rescount_dissimilar_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/overlaid_rescount_dissimilar_pdist_cosine.pdf", bbox_inches="tight")

if args.distribution == f"overlaid_interfacearea_dissimilar_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/dissimilar_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            for trial in trials:
                signal = collected_dist_vecs[ablation_type][ablation_p][trial]
                signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
                kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
                #ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid')
                y = kernel([x])
                maxy = max(maxy, np.max(y))
                ax[i, j].plot(x, y, linewidth=2, label=f"trial {trial}")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title("extensive ablation")
    ax[0, 1].set_title("moderate ablation")
    ax[0, 2].set_title("limited ablation")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("overlaid distributions of dissimilar protein complexes\ninterface-area count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/overlaid_interfacearea_dissimilar_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/overlaid_interfacearea_dissimilar_pdist_cosine.pdf", bbox_inches="tight")



if args.distribution == f"concatenated_rescount_dissimilar_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/dissimilar_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("concatenated distributions of dissimilar protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_rescount_dissimilar_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_rescount_dissimilar_pdist_cosine.pdf", bbox_inches="tight")


if args.distribution == f"concatenated_interfacearea_dissimilar_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/dissimilar_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("concatenated distributions of dissimilar protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_dissimilar_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_dissimilar_pdist_cosine.pdf", bbox_inches="tight")



if args.distribution == f"concatenated_rescount_similar_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/similar_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("concatenated distributions of similar protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_rescount_similar_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_rescount_similar_pdist_cosine.pdf", bbox_inches="tight")


if args.distribution == f"concatenated_interfacearea_similar_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/similar_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("concatenated distributions of similar protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_similar_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_similar_pdist_cosine.pdf", bbox_inches="tight")







if args.distribution == f"overlaid_rescount_removed_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            for trial in trials:
                signal = collected_dist_vecs[ablation_type][ablation_p][trial]
                signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
                kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
                #ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid')
                y = kernel([x])
                maxy = max(maxy, np.max(y))
                ax[i, j].plot(x, y, linewidth=2, label=f"trial {trial}")
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title("extensive ablation")
    ax[0, 1].set_title("moderate ablation")
    ax[0, 2].set_title("limited ablation")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
     
    #fig.suptitle("overlaid distributions of ablated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/overlaid_rescount_removed_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/overlaid_rescount_removed_pdist_cosine.pdf", bbox_inches="tight")

if args.distribution == f"overlaid_interfacearea_removed_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            for trial in trials:
                signal = collected_dist_vecs[ablation_type][ablation_p][trial]
                signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
                kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
                #ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid')
                y = kernel([x])
                maxy = max(maxy, np.max(y))
                ax[i, j].plot(x, y, linewidth=2, label=f"trial {trial}")
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title("extensive ablation")
    ax[0, 1].set_title("moderate ablation")
    ax[0, 2].set_title("limited ablation")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("overlaid distributions of ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/overlaid_interfacearea_removed_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/overlaid_interfacearea_removed_pdist_cosine.pdf", bbox_inches="tight")

if args.distribution == f"concatenated_rescount_removed_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i,j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("concatenated distributions of ablated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_pdist_cosine.pdf", bbox_inches="tight")

if args.distribution == f"concatenated_interfacearea_removed_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_pdist-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = np.hstack(collected_dist_vecs[ablation_type][ablation_p])
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("concatenated distributions of ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_pdist_cosine.pdf", bbox_inches="tight")


if args.distribution == f"collated_rescount_generated_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{args.replicate}/collated_generated_pdist_cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = np.array(collected_dist_vecs[ablation_type][ablation_p]).flatten()
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].set_xlim(left=0, right=1)
            ax[i, j].legend()
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("collated distributions of generated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/collated_rescount_generated_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/collated_rescount_generated_pdist_cosine.pdf", bbox_inches="tight")

if args.distribution == f"collated_interfacearea_generated_pdist_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{args.replicate}/collated_generated_pdist_cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = np.array(collected_dist_vecs[ablation_type][ablation_p]).flatten()
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].set_xlim(left=0, right=1)
            ax[i, j].legend()
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("collated distributions of generated protein complexes\ninterface-area count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/collated_interfacearea_generated_pdist_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/collated_interfacearea_generated_pdist_cosine.pdf", bbox_inches="tight")





if args.distribution == f"collated_rescount_generated_vs_removed_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{args.replicate}/collated_generated_vs_removed_cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = np.array(collected_dist_vecs[ablation_type][ablation_p]).flatten()
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between generated and removed protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/collated_rescount_generated_vs_removed_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/collated_rescount_generated_vs_removed_cosine.pdf", bbox_inches="tight")

if args.distribution == f"collated_interfacearea_generated_vs_removed_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{args.replicate}/collated_generated_vs_removed_cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = np.array(collected_dist_vecs[ablation_type][ablation_p]).flatten()
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between generated and removed protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/collated_interfacearea_generated_vs_removed_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/collated_interfacearea_generated_vs_removed_cosine.pdf", bbox_inches="tight")



if args.distribution == f"concatenated_generated1_rescount_vs_removed_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/generated1_vs_removed-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            dist_vec = collected_dist_vecs[ablation_type][ablation_p]
            #kernel = scipy.stats.gaussian_kde(dist_vec, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(dist_vec, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(dist_vec)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title("extensive ablation")
    ax[0, 1].set_title("moderate ablation")
    ax[0, 2].set_title("limited ablation")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("Distributions of generated vs ablated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/generated1_rescount_vs_removed_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/generated1_rescount_vs_removed_cosine.pdf", bbox_inches="tight")


if args.distribution == f"concatenated_generated1_interfacearea_vs_removed_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/generated1_vs_removed-cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            dist_vec = collected_dist_vecs[ablation_type][ablation_p]
            #kernel = scipy.stats.gaussian_kde(dist_vec, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(dist_vec, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(n))
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(dist_vec)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 2].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("Distributions of generated vs ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/generated1_interfacearea_vs_removed_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/generated1_interfacearea_vs_removed_cosine.pdf", bbox_inches="tight")











if args.distribution == f"collated_rescount_generated_vs_dissimilar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{args.replicate}/collated_generated_vs_removed_cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = np.array(collected_dist_vecs[ablation_type][ablation_p]).flatten()
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between dissimilar and generated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/collated_rescount_generated_vs_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/collated_rescount_generated_vs_dissimilar_cosine.pdf", bbox_inches="tight")


if args.distribution == f"collated_interfacearea_generated_vs_dissimilar_cosine.pdf":
    collected_dist_vecs = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{args.replicate}/collated_generated_vs_removed_cosine.joblib")
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = np.array(collected_dist_vecs[ablation_type][ablation_p]).flatten()
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].legend()
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between dissimilar and generated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/collated_interfacearea_generated_vs_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/collated_interfacearea_generated_vs_dissimilar_cosine.pdf", bbox_inches="tight")


if args.distribution == f"concatenated_rescount_removed_vs_dissimilar_cosine-summary.pdf":
    replicates = []
    for replicate in [4, 5, 6]:
        replicates.append(joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{replicate}/overlaid_removed_vs_dissimilar-cosine.joblib"))
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = []
            for replicate in replicates:
                signal.append(np.hstack(replicate[ablation_type][ablation_p]))
            signal = np.hstack(signal)
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            xmed = np.median(signal_sampled)
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between dissimilar and ablated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_vs_dissimilar_cosine-summary.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_vs_dissimilar_cosine-summary.pdf", bbox_inches="tight")

if args.distribution == f"concatenated_interfacearea_removed_vs_dissimilar_cosine-summary.pdf":
    replicates = []
    for replicate in [4, 5, 6]:
        replicates.append(joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{replicate}/overlaid_removed_vs_dissimilar_cosine.joblib"))
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = []
            for replicate in replicates:
                signal.append(np.hstack(replicate[ablation_type][ablation_p]))
            signal = np.hstack(signal)
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            xmed = np.median(signal_sampled)
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between dissimilar and ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_vs_dissimilar_cosine-summary.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_vs_dissimilar_cosine-summary.pdf", bbox_inches="tight")

if args.distribution == f"concatenated_interfacearea_removed_vs_similar_cosine-summary.pdf":
    replicates = []
    for replicate in [4, 5, 6]:
        replicates.append(joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{replicate}/overlaid_removed_vs_similar_cosine.joblib"))
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = []
            for replicate in replicates:
                signal.append(np.hstack(replicate[ablation_type][ablation_p]))
            signal = np.hstack(signal)
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            xmed = np.median(signal_sampled)
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between similar and ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_vs_similar_cosine-summary.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_removed_vs_similar_cosine-summary.pdf", bbox_inches="tight")

if args.distribution == f"concatenated_rescount_removed_vs_similar_cosine-summary.pdf":
    replicates = []
    for replicate in [4, 5, 6]:
        replicates.append(joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{replicate}/overlaid_removed_vs_similar_cosine.joblib"))
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = []
            for replicate in replicates:
                signal.append(np.hstack(replicate[ablation_type][ablation_p]))
            signal = np.hstack(signal)
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            xmed = np.median(signal_sampled)
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    fig.suptitle("cosine distances between similar and ablated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_vs_similar_cosine-summary.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_rescount_removed_vs_similar_cosine-summary.pdf", bbox_inches="tight")


if args.distribution == f"generated1_rescount_vs_removed_cosine-summary.pdf":
    replicates = []
    for replicate in [4, 5, 6]:
        replicates.append(joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{replicate}/generated1_vs_removed_cosine.joblib"))
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
            signal = []
            for replicate in replicates:
                signal.append(np.hstack(replicate[ablation_type][ablation_p]))
            signal = np.hstack(signal)
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    fig.suptitle("Distributions of generated vs ablated protein complexes\nresidue-residue count ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/generated1_rescount_vs_removed_cosine-summary.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/generated1_rescount_vs_removed_cosine-summary.pdf", bbox_inches="tight")

if args.distribution == f"generated1_interfacearea_vs_removed_cosine-summary.pdf":
    replicates = []
    for replicate in [4, 5, 6]:
        replicates.append(joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/{replicate}/generated1_vs_removed_cosine.joblib"))
    maxy = 0
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            signal = []
            for replicate in replicates:
                signal.append(np.hstack(replicate[ablation_type][ablation_p]))
            signal = np.hstack(signal)
            signal_sampled = np.random.choice(signal, size=sample_size, replace=True)
            kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color)
            y = kernel([x])
            maxy = max(maxy, np.max(y), np.max(n))
            ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            xmed = np.median(signal_sampled)
            ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_xlabel(f"cosine distance")
            ax[i, j].set_ylabel("probability density")
            ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\nprobability density")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\nprobability density")
    ax[0, 0].set_title(r"20\% retained")
    ax[0, 1].set_title(r"50\% retained")
    ax[0, 2].set_title(r"80\% retained")
    for i in range(2):
        for j in range(3):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    fig.suptitle("Distributions of generated vs ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/generated1_interfacearea_vs_removed_cosine-summary.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/generated1_interfacearea_vs_removed_cosine-summary.pdf", bbox_inches="tight")




if args.distribution == f"concatenated_interfacearea_similar_dissimilar_cosine.pdf":
    collected_dist_vecs_dissimilar = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_dissimilar-cosine.joblib")
    collected_dist_vecs_similar = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_similar-cosine.joblib")
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, squeeze=False, figsize=(6.4, 4.8))
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["interfacearea20", "interfacearea80"]):
            dist_vec_dissimilar = np.hstack(collected_dist_vecs_dissimilar[ablation_type][ablation_p])
            dist_vec_similar = np.hstack(collected_dist_vecs_similar[ablation_type][ablation_p])
            signal_sampled_dissimilar = np.random.choice(dist_vec_dissimilar, size=sample_size, replace=False)
            signal_sampled_similar = np.random.choice(dist_vec_similar, size=sample_size, replace=False)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            nd, bins, patches = ax[i, j].hist(signal_sampled_dissimilar, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color, label="dissimilar")
            ns, bins, patches = ax[i, j].hist(signal_sampled_similar, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=kde_color, label="similar", alpha=0.7)
            #kl_div = scipy.stats.entropy(nd, qk=ns, axis=0)
            #js = scipy.spatial.distance.jensenshannon(nd, ns, axis=0)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(nd), np.max(ns))
            #xmed = np.median(signal_sampled)
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            #ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("\\textbf{probability density}")
            
            #label_text = r"$D_{JS} (\text{B} \parallel \text{R}) = " + f"{js:.2f}$"
            #ax[i, j].text(0.95, 0.95, label_text,
            #               transform=ax[i, j].transAxes,
            #               verticalalignment="top",
            #               horizontalalignment="right",
            #               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9))
            if i == 0 and j == 1:
                #ax[i, j].legend(loc='lower right', bbox_to_anchor=(1, 0.5))
                ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    #ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 1].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(2):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between dissimilar and ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_similar_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_interfacearea_similar_dissimilar_cosine.pdf", bbox_inches="tight")

if args.distribution == f"concatenated_rescount_similar_dissimilar_cosine.pdf":
    collected_dist_vecs_dissimilar = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_dissimilar-cosine.joblib")
    collected_dist_vecs_similar = joblib.load(f"{data_dir}/intermediates/{args.embedding_model}/removed_vs_similar-cosine.joblib")
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, squeeze=False, figsize=(6.4, 4.8))
    maxy = 0
    for i, ablation_type in enumerate(["uniform", "targeted"]):
        for j, ablation_p in enumerate(["rescount20", "rescount80"]):
            dist_vec_dissimilar = np.hstack(collected_dist_vecs_dissimilar[ablation_type][ablation_p])
            dist_vec_similar = np.hstack(collected_dist_vecs_similar[ablation_type][ablation_p])
            signal_sampled_dissimilar = np.random.choice(dist_vec_dissimilar, size=sample_size, replace=False)
            signal_sampled_similar = np.random.choice(dist_vec_similar, size=sample_size, replace=False)
            #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
            nd, bins, patches = ax[i, j].hist(signal_sampled_dissimilar, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=line_color, label="dissimilar")
            ns, bins, patches = ax[i, j].hist(signal_sampled_similar, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=kde_color, label="similar", alpha=0.7)
            #js = scipy.spatial.distance.jensenshannon(nd, ns, axis=0)
            #y = kernel([x])
            #maxy = max(maxy, np.max(y), np.max(n))
            maxy = max(maxy, np.max(nd), np.max(ns))
            #xmed = np.median(signal_sampled)
            #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
            #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
            #ax[i, j].axvline(xmed, ymin=0, ymax=1, label=f"{xmed:.3f}", color=med_line_color, linestyle="--")
            ax[i, j].set_ylim(bottom=0, top=5)
            ax[i, j].set_xlabel("\\textbf{cosine distance}")
            #ax[i, j].set_ylabel("\\textbf{probability density}")
            
            #label_text = r"$D_{JS} (\text{B} \parallel \text{R}) = " + f"{js:.2f}$"
            #ax[i, j].text(0.95, 0.95, label_text,
            #               transform=ax[i, j].transAxes,
            #               verticalalignment="top",
            #               horizontalalignment="right",
            #               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9))
            if i == 0 and j == 1:
                #ax[i, j].legend(loc='lower right', bbox_to_anchor=(1, 0.5))
                ax[i, j].legend()
            ax[i, j].set_xlim(left=0, right=1)
    ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
    ax[0, 0].set_title("\\textbf{extensive ablation}")
    #ax[0, 1].set_title("\\textbf{moderate ablation}")
    ax[0, 1].set_title("\\textbf{limited ablation}")
    for i in range(2):
        for j in range(2):
            ax[i, j].set_ylim(bottom=0, top=maxy*1.1)
    #fig.suptitle("cosine distances between dissimilar and ablated protein complexes\ninterface-area ablation condition")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/concatenated_rescount_similar_dissimilar_cosine.png", bbox_inches="tight")
    fig.savefig(f"{args.out_dir}/concatenated_rescount_similar_dissimilar_cosine.pdf", bbox_inches="tight")

sys.exit()

s = {}
max_marker=50
for i, complx in enumerate(embeddings.keys()):
    def concat_embedding(complx):
        l = embeddings[complx]['l']
        l.embeddings = l.embeddings.numpy()
        r = embeddings[complx]['r']
        r.embeddings = r.embeddings.numpy()
        #print(l.embeddings)
        
        mean_embeddings_l = np.mean(l.embeddings[0,...], axis=0)
        mean_embeddings_r = np.mean(r.embeddings[0,...], axis=0)
        return np.hstack((np.hstack((mean_embeddings_l, mean_embeddings_r))))
        
    #mean_embeddings.append(concat_embedding(complx))
    
    mean_embeddings[complx] = np.mean(embeddings[complx][0,...], axis=0)
    #c.append(dips_meta.loc[complx, 'l_len'] + dips_meta.loc[complx, 'r_len'])
    s[complx] = (dips_meta.loc[complx, 'l_len']+dips_meta.loc[complx, 'r_len'])/1300*max_marker


#mean_embeddings = np.array(mean_embeddings)
#print(mean_embeddings.shape)

pca_model = joblib.load('esmc-embeddings-complex-holdout_val-pca.joblib')

fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=True, figsize=(11.0, 8.5))
print(ax)
holdout_embeddings = np.array([mean_embeddings[complx] for complx in holdout_keys])
holdout_s = [s[complx] for complx in holdout_keys]
holdout_pc_embeddings = pca_model.transform(holdout_embeddings)

for i, filtering in enumerate(['uniform', 'targeted']):
    for j, reduction_type in enumerate(['interfacearea20', 'interfacearea50', 'interfacearea80']):#, 'rescount20', 'rescount50', 'rescount80']:
        with open(f'{model_dir}/reductions_aminobert/{reduction_type}/{filtering}/trial_0/removed_val_keys.txt', 'r') as f:
            trial0_removed_val_keys = [line.strip() for line in f.readlines()]
        
        trial_val = list(set(val_keys) - set(trial0_removed_val_keys))
        trial_val_embeddings = np.array([mean_embeddings[complx] for complx in trial_val])
        trial_val_s = [s[complx] for complx in trial_val]
        trial_val_pc_embeddings = pca_model.transform(trial_val_embeddings)
        
        val_pts = ax[i,j].scatter(trial_val_pc_embeddings[:,0], trial_val_pc_embeddings[:,1], \
                        marker=markers['val'], s=trial_val_s, c=colors['val'], alpha=alphas['val'])

        trial_val_removed_embeddings = np.array([mean_embeddings[complx] for complx in trial0_removed_val_keys])
        trial_val_removed_s = [s[complx] for complx in trial0_removed_val_keys]
        trial_val_removed_pc_embeddings = pca_model.transform(trial_val_removed_embeddings)
        removed_pts = ax[i,j].scatter(trial_val_removed_pc_embeddings[:,0], trial_val_removed_pc_embeddings[:,1], \
                        marker=markers['val_removed'], s=trial_val_removed_s, \
                        c=colors['val_removed'], alpha=alphas['val_removed'])

        holdout_pts = ax[i,j].scatter(holdout_pc_embeddings[:,0], holdout_pc_embeddings[:,1], \
                        marker=markers['hold-out'], s=holdout_s, c=colors['hold-out'], alpha=alphas['hold-out'])
        #ax[i,j].legend()
        ax[i,j].legend(handles=[holdout_pts, val_pts, removed_pts], labels=['hold-out', 'val', 'removed val'], loc='lower right')

ax[0,0].set_ylabel('uniform')
ax[1,0].set_ylabel('targeted')

ax[0,0].set_title(r'80% removed')
ax[0,1].set_title(r'50% removed')
ax[0,2].set_title(r'20% removed')

ax[0,0].set_xlabel('pc 1')
ax[0,1].set_xlabel('pc 1')
ax[0,2].set_xlabel('pc 1')

ax[1,0].set_xlabel('pc 1')
ax[1,1].set_xlabel('pc 1')
ax[1,2].set_xlabel('pc 1')

ax[0,1].set_ylabel('pc 2')
ax[0,2].set_ylabel('pc 2')
ax[1,1].set_ylabel('pc 2')
ax[1,2].set_ylabel('pc 2')

fig.suptitle(f'ESM-C val interfacearea dissimilar vs removed complexes')
fig.savefig(f'esmc-val-interfacearea-complex-dissimilar_vs_removed-pca.png', bbox_inches='tight')
