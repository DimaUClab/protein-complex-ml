import argparse
import sys
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, skew


model_dir = os.environ["PROJ_CM_MODELS"]
data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
proj_root = os.environ["PROJ_ROOT"]
plt.style.use(f"{proj_root}/stylesheet.mplstyle")

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

#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', '-d', type=str, required=False, default='')
    parser.add_argument('--out-dir', '-o', type=str, required=True)
    args = parser.parse_args()
    
    x = np.linspace(0, 1, 2000)
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(11.0, 8.5))
    skewnesses = pd.read_csv(f"{data_dir}/intermediates/similarity_data/similarities_skewness.csv", header=0, sep='\t', index_col=0)
    for filtertype in ["interfacearea", "rescount"]:
        fig_heat, ax_heat = plt.subplots(figsize=(6, 7))

        heatmap_data = np.zeros((6, 3))
        row_labels = ["uniform train", "uniform validation", "uniform test",
                      "targeted train", "targeted validation", "targeted test"]
        col_labels = [r"20%", r"50%", r"80%"]

        for r_idx, (ablation, split) in enumerate([l.split(" ") for l in row_labels]):
            for c_idx, red_p in enumerate(col_labels):
                red_p = ''.join([i for i in red_p if i.isdigit()])
                val = skewnesses.loc[f"{filtertype}{red_p}-{ablation}-{split}", "skewness"]
                heatmap_data[r_idx, c_idx] = val

        im = ax_heat.imshow(heatmap_data, cmap="Blues", vmin=-1.3, vmax=0.3)

        ax_heat.set_xticks(np.arange(len(col_labels)), labels=col_labels)
        ax_heat.set_yticks(np.arange(len(row_labels)), labels=row_labels)
        ax_heat.tick_params(top=False, right=False)

        # Determine threshold for text color based on actual min/max of this heatmap
        thresh = (np.max(heatmap_data) + np.min(heatmap_data)) / 2.0
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax_heat.text(j, i, f"{heatmap_data[i, j]:.2f}",
                                    ha="center", va="center", color="black" if heatmap_data[i, j] < thresh else "white")

        ax_heat.set_title(f"Skewness Heatmap ({keyword_replace(filtertype)})")
        cbar = fig_heat.colorbar(im, ax=ax_heat)
        cbar.set_label("skewness", rotation=-90, va="bottom")
        fig_heat.tight_layout()
        fig_heat.savefig(f'{args.out_dir}/aminobert-esm2-{filtertype}-skewness-heatmap.pdf', bbox_inches='tight')
        fig_heat.savefig(f'{args.out_dir}/aminobert-esm2-{filtertype}-skewness-heatmap.png', bbox_inches='tight')

    
