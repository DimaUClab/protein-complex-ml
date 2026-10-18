import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
import sys
import os
import argparse
import scipy
from tqdm import tqdm

data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
proj_root = os.environ["PROJ_ROOT"]
plt.style.use(f"{proj_root}/stylesheet.mplstyle")

ablation_map = {"interfacearea20":"extensive ablation", "interfacearea50":"moderate ablation", "interfacearea80":"limited ablation",
                "rescount20":"extensive ablation", "rescount50":"moderate ablation", "rescount80":"limited ablation"}

hist_color = "midnightblue"
kde_color = "red"
med_line_color = "forestgreen"
figsize = (7, 4)
trials = list(range(5))

def plot_summary_distribution(args):
    if args.filtering == "interfacearea":
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
        maxy = -1
        for j, ablation_c in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
            ax[0, j].set_title("\\textbf{" + ablation_map[ablation_c] + "}")
            for i, filtering in enumerate(["uniform", "targeted"]):
                signal = np.concatenate([joblib.load(f"{args.generated_dir}/{ablation_c}-{filtering}-trial_{k}-generated_ptm.joblib") for k in trials], axis=None)
                print(np.min(signal))
                signal_sampled = np.random.choice(signal, size=1000, replace=True)
                n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=False, bottom=0, align='mid', color=hist_color)
                #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
                #x = np.linspace(0, 1, 1000)
                #y = kernel([x])
                mn = np.mean(signal)
                var = np.var(signal)
                print(f"N({mn}, {var})")
                maxy = max(maxy, np.max(n))
                #dense_art = ax[i, j].plot(x, y, color=kde_color, lw=2)
                #ax[i, j].fill_between(x, y, y2=0, alpha=0.5, color=kde_color)
                #line_art = ax[i, j].axvline(x=mn, ymin=0, ymax=1, color=med_line_color, linestyle='--', lw=2)
                #ax[i, j].set_ylabel("counts")
                #legend = r"$\mathcal{N}" + f"({mn:.3f}, {var:.4f})$"
                #ax[i, j].legend(handles=[dense_art, line_art], labels=["pTM", rf"{legend}"])
        ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{counts}")
        ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{counts}")
        ax[1, 0].set_xlabel("\\textbf{pTM score}")
        ax[1, 1].set_xlabel("\\textbf{pTM score}")
        ax[1, 2].set_xlabel("\\textbf{pTM score}")
        #fig.suptitle("distribution of protein template modelling scores\ninterface-area ablation condition")
        for j in range(3):
            for i in range(2):
                ax[i, j].set_ylim(0, maxy*1.1)
                ax[i, j].set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(f"{args.out_file}.png", bbox_inches="tight")
        fig.savefig(f"{args.out_file}.pdf", bbox_inches="tight")
    elif args.filtering == "rescount":
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=figsize)
        maxy = -1
        for j, ablation_c in enumerate(["rescount20", "rescount50", "rescount80"]):
            ax[0, j].set_title("\\textbf{" + ablation_map[ablation_c] + "}")
            for i, filtering in enumerate(["uniform", "targeted"]):
                signal = np.concatenate([joblib.load(f"{args.generated_dir}/{ablation_c}-{filtering}-trial_{k}-generated_ptm.joblib") for k in trials], axis=None)
                print(np.min(signal))
                signal_sampled = np.random.choice(signal, size=1000, replace=True)
                n, bins, patches = ax[i, j].hist(signal_sampled, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=hist_color)
                #kernel = scipy.stats.gaussian_kde(signal_sampled, bw_method="silverman")
                #x = np.linspace(0, 1, 1000)
                #y = kernel([x])
                mn = np.mean(signal)
                var = np.var(signal)
                print(f"N({mn}, {var})")
                maxy = max(np.max(n), maxy)
                #dense_art = ax[i, j].plot(x, y, color=kde_color, lw=2)
                #ax[i, j].fill_between(x, y, y2=0, alpha=0.5, color=kde_color)
                # line_art = ax[i, j].axvline(x=mn, ymin=0, ymax=1, color=med_line_color, linestyle='--', lw=2)
                #ax[i, j].set_ylabel("counts")
                # legend = r"$\mathcal{N}" + f"({mn:.3f}, {var:.4f})$"
                # ax[i, j].legend(handles=[dense_art, line_art], labels=["pTM", rf"{legend}"])
        ax[0, 0].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{counts}")
        ax[1, 0].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{counts}")
        ax[1, 0].set_xlabel("\\textbf{pTM score}")
        ax[1, 1].set_xlabel("\\textbf{pTM score}")
        ax[1, 2].set_xlabel("\\textbf{pTM score}")
        #fig.suptitle("distribution of protein template modelling scores\nresidue-residue count ablation condition")
        for j in range(3):
            for i in range(2):
                ax[i, j].set_ylim(0, maxy*1.1)
                ax[i, j].set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(f"{args.out_file}.png", bbox_inches="tight")
        fig.savefig(f"{args.out_file}.pdf", bbox_inches="tight")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", "-g", help="output filename", required=True, type=str)
    parser.add_argument("--generated-dir", "-d", required=True, type=str)
    parser.add_argument("--filtering", "-f", required=False, type=str, default="interfacearea", choices=["interfacearea", "rescount"])
    
    args = parser.parse_args()
    plot_summary_distribution(args)
    
    # if "summary" in os.path.splitext(args.out_file)[0].split("-"):
    #     args.filtering = os.path.splitext(args.out_file)[0].split("-")[-1]
    #     plot_summary_distribution(args)
    #     sys.exit()
    #args.replicate = os.path.split(os.path.split(args.out_dir)[0])[1]
    #ptm_distribution = joblib.load(f"{data_dir}/intermediates/generated_ptm_distributions/{args.replicate}/ptm_scores.joblib")
    # args.filtering = os.path.splitext(args.out_file)[0].split("-")[-1]
    # if args.filtering == "interfacearea":
    #     fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=(11, 8.5))
    #     maxy = -1
    #     for j, ablation in enumerate(["interfacearea20", "interfacearea50", "interfacearea80"]):
    #         ax[0, j].set_title(fr"{ablation_map[ablation]}")
    #         for i, filtering in enumerate(["uniform", "targeted"]):
    #             signal = ptm_distribution[f"{ablation}-{filtering}"]
    #             n, bins, patches = ax[i, j].hist(signal, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=hist_color)
    #             kernel = scipy.stats.gaussian_kde(signal, bw_method="silverman")
    #             x = np.linspace(0, 1, 1000)
    #             y = kernel([x])
    #             mn = np.mean(signal)
    #             var = np.var(signal)
    #             maxy = max(np.max(n), maxy)
    #             dense_art = ax[i, j].plot(x, y, color=kde_color, lw=2)
    #             ax[i, j].fill_between(x, y, y2=0, alpha=0.5, color=kde_color)
    #             line_art = ax[i, j].vlines(x=mn, ymin=0, ymax=np.max(n)*1.1, color=med_line_color, linestyle='--', lw=2)
    #             ax[i, j].set_xlabel("pTM score")
    #             ax[i, j].set_ylabel("probability density")
    #             legend = r"$\mathcal{N}" + f"({mn:.3f}, {var:.4f})$"
    #             ax[i, j].legend(handles=[dense_art, line_art], labels=["pTM", fr"{legend}"])
    #     ax[0, 0].set_ylabel("uniform filtering")
    #     ax[1, 0].set_ylabel("targeted filtering")
    #     fig.suptitle("distribution of protein template modelling scores\ninterface-area ablation condition")
    #     for j in range(3):
    #         for i in range(2):
    #             ax[i, j].set_ylim(0, maxy*1.1)
    #             ax[i, j].set_xlim(0, 1)
    #     fig.tight_layout()
    #     fig.savefig(f"{args.out_dir}/{args.out_file}", bbox_inches="tight")
    # elif args.filtering == "rescount":
    #     fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=False, figsize=(11, 8.5))
    #     maxy = -1
    #     for j, ablation in enumerate(["rescount20", "rescount50", "rescount80"]):
    #         ax[0, j].set_title(fr"{ablation_map[ablation]}")
    #         for i, filtering in enumerate(["uniform", "targeted"]):
    #             signal = ptm_distribution[f"{ablation}-{filtering}"]
    #             n, bins, patches = ax[i, j].hist(signal, bins=20, range=(0, 1), density=True, bottom=0, align='mid', color=hist_color)
    #             kernel = scipy.stats.gaussian_kde(signal, bw_method="silverman")
    #             x = np.linspace(0, 1, 1000)
    #             y = kernel([x])
    #             mn = np.mean(signal)
    #             var = np.var(signal)
    #             maxy = max(np.max(n), maxy)
    #             dense_art = ax[i, j].plot(x, y, color=kde_color, lw=2)
    #             ax[i, j].fill_between(x, y, y2=0, alpha=0.5, color=kde_color)
    #             line_art = ax[i, j].vlines(x=mn, ymin=0, ymax=np.max(n)*1.1, color=med_line_color, linestyle='--', lw=2)
    #             ax[i, j].set_xlabel("pTM score")
    #             ax[i, j].set_ylabel("probability density")
    #             legend = r"$\mathcal{N}" + f"({mn:.3f}, {var:.4f})$"
    #             ax[i, j].legend(handles=[dense_art, line_art], labels=["pTM", fr"{legend}"])
    #     ax[0, 0].set_ylabel("uniform filtering")
    #     ax[1, 0].set_ylabel("targeted filtering")
    #     fig.suptitle("distribution of protein template modelling scores\nresidue-residue count ablation condition")
    #     for j in range(3):
    #         for i in range(2):
    #             ax[i, j].set_ylim(0, maxy*1.1)
    #             ax[i, j].set_xlim(0, 1)
    #     fig.tight_layout()
    #     fig.savefig(f"{args.out_dir}/{args.out_file}", bbox_inches="tight")
