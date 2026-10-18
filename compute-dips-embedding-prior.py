import joblib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import scipy
import sys
import os
import argparse
import torch


data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
proj_root = os.environ["PROJ_ROOT"]
model_dir = os.environ["PROJ_CM_MODELS"]
plt.style.use(os.path.join(proj_root, "stylesheet.mplstyle"))

ablation_map = {"interfacearea20":"20% retained", "interfacearea50":"50% retained", "interfacearea80":"80% retained",
                "rescount20":"20% retained", "rescount50":"50% retained", "rescount80":"80% retained"}

hist_color = "midnightblue"
kde_color = "red"
med_line_color = "forestgreen"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", "-e", default="esmc-600m-2024-12", required=False, type=str, choices=["esmc-6b-2024-12", "esmc-600m-2024-12"])
    args = parser.parse_args()
    
    try:
        pdist = joblib.load(os.path.join(data_dir, f"dips-plus-pdist_cosine-{args.embedding_model}.joblib"))
    except FileNotFoundError:
        embeddings_train = joblib.load(os.path.join(data_dir, "intermediates", f"dips_plus-train-complex-embeddings-{args.embedding_model}.joblib"))
        embeddings_hval = joblib.load(os.path.join(data_dir, "intermediates", f"dips_plus-holdout_val-complex-embeddings-{args.embedding_model}.joblib"))
        mean_embeddings = []
        for complx_name in embeddings_train.keys():
            mean_embeddings.append(np.mean(embeddings_train[complx_name][0,...], axis=0))
        for complx_name in embeddings_hval.keys():
            mean_embeddings.append(np.mean(embeddings_hval[complx_name][0,...], axis=0))
        mean_embeddings = np.array(mean_embeddings)
        chosen_ndxs = np.random.choice(mean_embeddings.shape[0], size=3000, replace=True)
        mean_embeddings = mean_embeddings[chosen_ndxs]
        pdist = scipy.spatial.distance.pdist(mean_embeddings, metric="cosine")
        joblib.dump(pdist, os.path.join(data_dir, f"dips-plus-pdist_cosine-{args.embedding_model}.joblib"), compress=("zlib", 3))
    ptm_dist = joblib.load(os.path.join(data_dir, "intermediates", "dips_ptm_scores.joblib"))
    ptm_dist = torch.tensor(list(ptm_dist.values())).cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(7, 4))
    n, bins, patches = ax[0].hist(ptm_dist, bins=20, range=(0, 1), density=False, bottom=0, align='mid', color=hist_color)
    #kernel = scipy.stats.gaussian_kde(pdist, bw_method="silverman")
    #x = np.linspace(0, 1, 1000)
    #y = kernel([x])
    med = np.median(ptm_dist)
    #maxy = max(np.max(n), np.max(y))
    maxy = np.max(n)
    #dense_art = ax.plot(x, y, color=kde_color, lw=2)
    #ax.fill_between(x, y, y2=0, alpha=0.5, color=kde_color)
    line_art = ax[0].axvline(x=med, ymin=0, ymax=1, color=med_line_color, linestyle='--', lw=2)
    legend = f"{med:.3f}"
    ax[0].legend(handles=[line_art], labels=[rf"{legend}"])
    ax[0].set_xlabel("\\textbf{pTM score}")
    ax[0].set_ylabel("\\textbf{counts}")
    #ax[0].set_title("\\textbf{DIPS-Plus cosine distance}\n\\textbf{distribution}")
    #fig.suptitle("distribution of cosine distances for DIPS+ dataset")
    
    ax[0].set_ylim(0, maxy*1.1)
    ax[0].set_xlim(0, 1)

    n, bins, patches = ax[1].hist(pdist, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=hist_color)
    #kernel = scipy.stats.gaussian_kde(pdist, bw_method="silverman")
    #x = np.linspace(0, 1, 1000)
    #y = kernel([x])
    med = np.median(pdist)
    #maxy = max(np.max(n), np.max(y))
    maxy = np.max(n)
    #dense_art = ax.plot(x, y, color=kde_color, lw=2)
    #ax.fill_between(x, y, y2=0, alpha=0.5, color=kde_color)
    line_art = ax[1].axvline(x=med, ymin=0, ymax=1, color=med_line_color, linestyle='--', lw=2)
    legend = f"{med:.3f}"
    ax[1].legend(handles=[line_art], labels=[rf"{legend}"])
    ax[1].set_xlabel("\\textbf{cosine distance}")
    ax[1].set_ylabel("\\textbf{probability density}")
    #ax[1].set_title("\\textbf{DIPS-Plus pTM score}\n\\textbf{distribution}")
    ax[1].set_ylim(0, maxy*1.1)
    ax[1].set_xlim(0, 1)
    #fig.suptitle("distribution of cosine distances for DIPS+ dataset")
    fig.tight_layout()
    fig.savefig(os.path.join(art_dir, f"dips_plus_ptm_and_embeddings-{args.embedding_model}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(art_dir, f"dips_plus_ptm_and_embeddings-{args.embedding_model}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(art_dir, f"dips_plus_ptm_and_embeddings-{args.embedding_model}.svg"), bbox_inches="tight")
