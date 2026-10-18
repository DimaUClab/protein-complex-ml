import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rcParams
import matplotlib as mpl
from scipy.stats import gaussian_kde, skew



model_dir = os.environ["PROJ_CM_MODELS"]
data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
proj_root = os.environ["PROJ_ROOT"]

plt.style.use(os.path.join(proj_root, "stylesheet.mplstyle"))
similarity_dir = os.path.join(data_dir, "intermediates", "similarity_data")

colors = {'SAS': '#2244EE', 'LinearCKA': '#AA4422', 'Frobenius': '#22AA44', 'Nuclear': '#9913FF'}
markers = {'train': '1', 'val': '+', 'holdout': '.'}
train_color = '#2244EE'
val_color = '#AA4422'
holdout_color = '#22AA44'

color_cycle = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
# riemann_color =    '#44EE22'
# procrustes_color =    '#EE2244'

def keyword_replace(keyword):
    if keyword == 'interfacearea':
        return 'interface-area'
    if keyword == 'rescount':
        return 'residue-residue count'

reductions = {"20":"extensive ablation", "50":"moderate ablation", "80":"limited ablation"}
marksize=6

sim_fn = sys.argv[1]
#
figsize=(7, 5)
if __name__ == '__main__':
    x = np.linspace(0, 1, 2000)
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=figsize)
    skewnesses = pd.DataFrame()
    for row_ndx, filtertype in enumerate(["interfacearea", "rescount"]):
        for col_ndx, reduction_p in enumerate(["20", "50", "80"]):
            train_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_0-train-similarities.dat',
                                     header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_{
                                         trial}-train-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                train_data = pd.concat([train_data, trial_data], axis=0, verify_integrity=False)
            train_data = train_data.groupby('complex ID').agg({sim_fn: 'mean'})


            val_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_0-val-similarities.dat',
                                   header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_{
                                         trial}-val-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                val_data = pd.concat([val_data, trial_data], axis=0, verify_integrity=False)
            val_data = val_data.groupby('complex ID').agg({sim_fn: 'mean'})
            
            holdout_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_0-holdout-similarities.dat',
                                       header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_{trial}-holdout-similarities.dat',
                                         header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                holdout_data = pd.concat([holdout_data, trial_data], axis=0, verify_integrity=False)
            holdout_data = holdout_data.groupby('complex ID').agg({sim_fn: 'mean'})
            assert len(set(val_data.index) & set(holdout_data.index)) == 0
            skew_rows = pd.DataFrame({
                'filtertype': [filtertype]*3,
                'ablation': ['uniform']*3,
                'reduction_p': [reduction_p]*3,
                'split': ['train', 'validation', 'test'],
                'skewness': [skew(train_data[sim_fn].dropna()), skew(val_data[sim_fn].dropna()), skew(holdout_data[sim_fn].dropna())]
            })
            skewnesses = pd.concat([skewnesses, skew_rows], ignore_index=True)

            train_kernel = gaussian_kde(train_data[sim_fn], bw_method='silverman')
            val_kernel = gaussian_kde(val_data[sim_fn], bw_method='silverman')
            holdout_kernel = gaussian_kde(holdout_data[sim_fn], bw_method='silverman')

            ax[row_ndx][col_ndx].plot(x, train_kernel([x]), color=train_color,
                                      label='train', marker=markers['train'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].plot(x, val_kernel([x]), color=val_color,
                                      label='validation', marker=markers['val'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].plot(x, holdout_kernel([x]), color=holdout_color,
                                      label='test', marker=markers['holdout'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].set_ylim(0, 3)
            if row_ndx == 0 and col_ndx == 2:
                ax[row_ndx][col_ndx].legend()

            #ax[row_ndx][col_ndx].set_ylabel("probability density")
            if row_ndx == 0:
                ax[row_ndx][col_ndx].set_title("\\textbf{"+reductions[reduction_p]+"}")
            elif row_ndx == 1:
                ax[row_ndx][col_ndx].set_xlabel("\\textbf{CKA similarity}")
    ax[0, 0].set_ylabel("\\textbf{interface-area}\n\n\\textbf{counts}")
    ax[1, 0].set_ylabel("\\textbf{res-res count}\n\n\\textbf{counts}")
    fig.suptitle('AminoBERT vs ESM2 uniform ablation')
    fig.tight_layout()
    fig.savefig(f'{art_dir}/aminobert-esm2-uniform-similarities-{sim_fn}.pdf', bbox_inches='tight')
    fig.savefig(f'{art_dir}/aminobert-esm2-uniform-similarities-{sim_fn}.png', bbox_inches='tight')


    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=figsize)
    for row_ndx, filtertype in enumerate(["interfacearea", "rescount"]):
        for col_ndx, reduction_p in enumerate(["20", "50", "80"]):
            train_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_0-train-similarities.dat',
                                     header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_{
                                         trial}-train-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                train_data = pd.concat([train_data, trial_data], axis=0, verify_integrity=False)
            train_data = train_data.groupby('complex ID').agg({sim_fn: 'mean'})


            val_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_0-val-similarities.dat',
                                   header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_{
                                         trial}-val-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                val_data = pd.concat([val_data, trial_data], axis=0, verify_integrity=False)
            val_data = val_data.groupby('complex ID').agg({sim_fn: 'mean'})

            holdout_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_0-holdout-similarities.dat',
                                       header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_{
                                         trial}-holdout-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                holdout_data = pd.concat([holdout_data, trial_data], axis=0, verify_integrity=False)
            holdout_data = holdout_data.groupby('complex ID').agg({sim_fn: 'mean'})
            assert len(set(val_data.index) & set(holdout_data.index)) == 0
            skew_rows = pd.DataFrame({
                'filtertype': [filtertype]*3,
                'ablation': ['targeted']*3,
                'reduction_p': [reduction_p]*3,
                'split': ['train', 'validation', 'test'],
                'skewness': [skew(train_data[sim_fn].dropna()), skew(val_data[sim_fn].dropna()), skew(holdout_data[sim_fn].dropna())]
            })
            skewnesses = pd.concat([skewnesses, skew_rows], ignore_index=True)

            train_kernel = gaussian_kde(train_data[sim_fn], bw_method='silverman')
            val_kernel = gaussian_kde(val_data[sim_fn], bw_method='silverman')
            holdout_kernel = gaussian_kde(holdout_data[sim_fn], bw_method='silverman')

            ax[row_ndx][col_ndx].plot(x, train_kernel([x]), color=train_color,
                                      label='train', marker=markers['train'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].plot(x, val_kernel([x]), color=val_color,
                                      label='validation', marker=markers['val'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].plot(x, holdout_kernel([x]), color=holdout_color,
                                      label='test', marker=markers['holdout'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].set_ylim(0, 3.6)
            if row_ndx == 0 and col_ndx == 2:
                ax[row_ndx][col_ndx].legend()
            #ax[row_ndx][col_ndx].set_ylabel("probability density")
            if row_ndx == 0:
                ax[row_ndx][col_ndx].set_title("\\textbf{"+reductions[reduction_p]+"}")
            elif row_ndx == 1:
                ax[row_ndx][col_ndx].set_xlabel(f'{sim_fn} similarity')
    ax[0, 0].set_ylabel("\\textbf{interface-area}\n\n\\textbf{probability density}")
    ax[1, 0].set_ylabel("\\textbf{res-res count}\n\n\\textbf{probability density}")
    #fig.suptitle('AminoBERT vs ESM2 targeted ablation')
    fig.tight_layout()
    fig.savefig(f'{art_dir}/aminobert-esm2-targeted-similarities-{sim_fn}.pdf', bbox_inches='tight')
    fig.savefig(f'{art_dir}/aminobert-esm2-targeted-similarities-{sim_fn}.png', bbox_inches='tight')

    print(skewnesses)
    skewnesses.to_csv(f"{data_dir}/intermediates/similarity_data/similarities_skewness-{sim_fn}.csv", index=False)

    skew_norm = mpl.colors.Normalize(vmin=-1.3, vmax=0.3)
    for filtertype in ["interfacearea", "rescount"]:
        fig_heat, ax_heat = plt.subplots(figsize=(6, 7))

        heatmap_data = np.zeros((6, 3))
        row_labels = ["\\textbf{uniform train}", "\\textbf{uniform validation}", "\\textbf{uniform test}",
                      "\\textbf{targeted train}", "\\textbf{targeted validation}", "\\textbf{targeted test}"]
        col_labels = ["\\textbf{extensive}", "\\textbf{moderate}", "\\textbf{limited}"]

        for r_idx, (ablation, split) in enumerate([("uniform", "train"), ("uniform", "validation"), ("uniform", "test"), ("targeted", "train"), ("targeted", "validation"), ("targeted", "test")]):
            for c_idx, red_p in enumerate(["20", "50", "80"]):
                val = skewnesses[(skewnesses['filtertype'] == filtertype) & (skewnesses['ablation'] == ablation) & (
                    skewnesses['split'] == split) & (skewnesses['reduction_p'] == red_p)]['skewness'].values[0]
                heatmap_data[r_idx, c_idx] = val

        im = ax_heat.imshow(heatmap_data, cmap="Blues_r", norm=skew_norm)

        ax_heat.set_xticks(np.arange(len(col_labels)), labels=col_labels, ha="left", rotation=-45)
        ax_heat.set_yticks(np.arange(len(row_labels)), labels=row_labels)
        ax_heat.tick_params(top=False, right=False)

        # Determine threshold for text color based on actual min/max of this heatmap
        thresh = 0
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax_heat.text(j, i, "\\textbf{"+f"{heatmap_data[i, j]:.2f}"+"}",
                                    ha="center", va="center", color="white" if heatmap_data[i, j] < thresh else "black")

        #ax_heat.set_title(f"Skewness Heatmap ({keyword_replace(filtertype)})")
        cbar = fig_heat.colorbar(im, ax=ax_heat)
        cbar.set_label("\\textbf{skewness}", rotation=-90, va="bottom")
        fig_heat.tight_layout()
        fig_heat.savefig(f'{art_dir}/aminobert-esm2-{filtertype}-skewness-heatmap-{sim_fn}.pdf', bbox_inches='tight')
        fig_heat.savefig(f'{art_dir}/aminobert-esm2-{filtertype}-skewness-heatmap-{sim_fn}.png', bbox_inches='tight')

    
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=figsize)
    for row_ndx, reduction in enumerate(["uniform", "targeted"]):
        for col_ndx, reduction_p in enumerate(["20", "50", "80"]):
            train_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-interfacearea{reduction_p}-{reduction}-trial_0-train-similarities.dat',
                                     header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-interfacearea{reduction_p}-{reduction}-trial_{
                                         trial}-train-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                train_data = pd.concat([train_data, trial_data], axis=0, verify_integrity=False)
            train_data = train_data.groupby('complex ID').agg({sim_fn: 'mean'})


            val_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-interfacearea{reduction_p}-{reduction}-trial_0-val-similarities.dat',
                                   header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-interfacearea{reduction_p}-{reduction}-trial_{
                                         trial}-val-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                val_data = pd.concat([val_data, trial_data], axis=0, verify_integrity=False)
            val_data = val_data.groupby('complex ID').agg({sim_fn: 'mean'})

            holdout_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-interfacearea{reduction_p}-{reduction}-trial_0-holdout-similarities.dat',
                                       header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-interfacearea{reduction_p}-{reduction}-trial_{
                                         trial}-holdout-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                holdout_data = pd.concat([holdout_data, trial_data], axis=0, verify_integrity=False)
            holdout_data = holdout_data.groupby('complex ID').agg({sim_fn: 'mean'})

            train_kernel = gaussian_kde(train_data[sim_fn], bw_method='silverman')
            val_kernel = gaussian_kde(val_data[sim_fn], bw_method='silverman')
            holdout_kernel = gaussian_kde(holdout_data[sim_fn], bw_method='silverman')

            ax[row_ndx][col_ndx].plot(x, train_kernel([x]), color=train_color,
                                      label='train', marker=markers['train'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].plot(x, val_kernel([x]), color=val_color,
                                      label='validation', marker=markers['val'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].plot(x, holdout_kernel([x]), color=holdout_color,
                                      label='test', marker=markers['holdout'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].set_ylim(0, 4)
            if row_ndx == 0 and col_ndx == 2:
                ax[row_ndx][col_ndx].legend()
            if col_ndx == 0 and reduction == "uniform":
                ax[row_ndx][col_ndx].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
            elif col_ndx == 0 and reduction == "targeted":
                ax[row_ndx][col_ndx].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
            #elif col_ndx == 0:
                #ax[row_ndx][col_ndx].set_ylabel("probability density")
            if row_ndx == 0:
                ax[row_ndx][col_ndx].set_title("\\textbf{" + reductions[reduction_p] + "}")
            elif row_ndx == 1:
                ax[row_ndx][col_ndx].set_xlabel('\\textbf{CKA similarity}')
    #fig.suptitle('AminoBERT vs ESM2 interface-area ablation condition')
    fig.tight_layout()
    fig.savefig(f'{art_dir}/aminobert-esm2-interfacearea-similarities-{sim_fn}.pdf', bbox_inches='tight')
    fig.savefig(f'{art_dir}/aminobert-esm2-interfacearea-similarities-{sim_fn}.png', bbox_inches='tight')


    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=figsize)
    for row_ndx, reduction in enumerate(["uniform", "targeted"]):
        for col_ndx, reduction_p in enumerate(["20", "50", "80"]):
            train_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-rescount{reduction_p}-{reduction}-trial_0-train-similarities.dat',
                                     header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-rescount{reduction_p}-{reduction}-trial_{
                                         trial}-train-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                train_data = pd.concat([train_data, trial_data], axis=0, verify_integrity=False)
            train_data = train_data.groupby('complex ID').agg({sim_fn: 'mean'})


            val_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-rescount{reduction_p}-{reduction}-trial_0-val-similarities.dat',
                                   header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-rescount{reduction_p}-{reduction}-trial_{
                                         trial}-val-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                val_data = pd.concat([val_data, trial_data], axis=0, verify_integrity=False)
            val_data = val_data.groupby('complex ID').agg({sim_fn: 'mean'})

            holdout_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-rescount{reduction_p}-{reduction}-trial_0-holdout-similarities.dat',
                                       header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(f'{similarity_dir}/similarities/aminobert-esm2-rescount{reduction_p}-{reduction}-trial_{
                                         trial}-holdout-similarities.dat', header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                holdout_data = pd.concat([holdout_data, trial_data], axis=0, verify_integrity=False)
            holdout_data = holdout_data.groupby('complex ID').agg({sim_fn: 'mean'})

            train_kernel = gaussian_kde(train_data[sim_fn], bw_method='silverman')
            val_kernel = gaussian_kde(val_data[sim_fn], bw_method='silverman')
            holdout_kernel = gaussian_kde(holdout_data[sim_fn], bw_method='silverman')

            ax[row_ndx][col_ndx].plot(x, train_kernel([x]), color=train_color,
                                      label='train', marker=markers['train'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].plot(x, val_kernel([x]), color=val_color,
                                      label='validation', marker=markers['val'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].plot(x, holdout_kernel([x]), color=holdout_color,
                                      label='test', marker=markers['holdout'], markevery=0.1, markersize=marksize)
            ax[row_ndx][col_ndx].set_ylim(0, 4)
            if row_ndx == 0 and col_ndx == 2:
                ax[row_ndx][col_ndx].legend()
            if col_ndx == 0 and reduction == "uniform":
                ax[row_ndx][col_ndx].set_ylabel("\\textbf{uniform filtering}\n\n\\textbf{probability density}")
            elif col_ndx == 0 and reduction == "targeted":
                ax[row_ndx][col_ndx].set_ylabel("\\textbf{targeted filtering}\n\n\\textbf{probability density}")
            #elif col_ndx == 0:
            #    ax[row_ndx][col_ndx].set_ylabel("probability density")
            if row_ndx == 0:
                ax[row_ndx][col_ndx].set_title("\\textbf{" + reductions[reduction_p]+"}")
            elif row_ndx == 1:
                ax[row_ndx][col_ndx].set_xlabel("\\textbf{CKA similarity")
    #fig.suptitle('AminoBERT vs ESM2 residue-residue count ablation condition')
    fig.tight_layout()
    fig.savefig(f'{art_dir}/aminobert-esm2-rescount-similarities-{sim_fn}.pdf', bbox_inches='tight')
    fig.savefig(f'{art_dir}/aminobert-esm2-rescount-similarities-{sim_fn}.png', bbox_inches='tight')
