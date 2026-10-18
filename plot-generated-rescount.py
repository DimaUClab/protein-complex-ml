import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import scipy
import os
import sys
import joblib
import pandas as pd
from collections import defaultdict
from scipy.stats import entropy

proj_root = os.environ["PROJ_ROOT"]
data_dir = os.environ["PROJ_DATASETS"]
model_dir = os.environ["PROJ_CM_MODELS"]
plt.style.use(os.path.join(proj_root, "stylesheet.mplstyle"))
#rcParams["text.usetex"] = False
#rcParams["font.sans-serif"] = "FreeSans"
MAX_SEQ_LEN = 650
rescounts = pd.read_csv(f"{data_dir}/complex_info_with_interface_areas.csv", header=0, index_col=0, sep=',')
dips_meta = pd.read_csv(f"{data_dir}/dips_meta.dat", header=0, index_col=0, sep=' ')
with open(f"{model_dir}/holdout_val_keys.txt", 'r') as f:
    holdout_keys = [line.strip() for line in f.readlines()[1:]]

s1_encodings = joblib.load(os.path.join(model_dir, 'esm_s1_encodings_dipsplus.joblib'))
s2_encodings = joblib.load(os.path.join(model_dir, 'esm_s2_encodings_dipsplus.joblib'))

with open(os.path.join(model_dir, 'pairs-postprocessed-val.txt'), 'r') as f:
    all_val_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings.keys()]

val_keys = []
for key in all_val_keys:
    if dips_meta.loc[key, 'l_len'] > MAX_SEQ_LEN or dips_meta.loc[key, 'r_len'] > MAX_SEQ_LEN:
        continue
    val_keys.append(key)


distributions_original = defaultdict(list)
keyerrors = defaultdict((lambda :0))
for ablation_c in ["interfacearea", "rescount"]:
    for percent_p in [20, 50, 80]:
        for filtering in ["uniform", "targeted"]:
            for trial in range(5):
                with open(f'{model_dir}/reductions_aminobert/{ablation_c}{percent_p}/{filtering}/trial_{trial}/removed_val_keys.txt', 'r') as f:
                    trial_removed_val_keys = [line.strip() for line in f.readlines()]
                    
                ablation_keys = set(val_keys) - set(trial_removed_val_keys)
                if percent_p in [20, 50]:
                    ablation_keys = ablation_keys - set(holdout_keys)
                for key in ablation_keys:
                    try:
                        distributions_original[f"{ablation_c}{percent_p}-{filtering}-trial_{trial}"].append(rescounts.loc[key, "interface_res_count"])
                    except KeyError:
                        keyerrors[f"{ablation_c}{percent_p}-{filtering}-trial_{trial}"] += 1
print(keyerrors)
for key in distributions_original.keys():
    distributions_original[key] = np.array(distributions_original[key])
distributions_generated = {}

for filtering in ["uniform", "targeted"]:
    for trial in range(5):
        rescounts_ = list()
        data_ = joblib.load(os.path.join(data_dir, "intermediates",
                                      "generation_for_emily_sep14_2026", f"rescount20-{filtering}-trial_{trial}-generated_cm-contacts.joblib"))
        for cm in data_:
            rescounts_.append(np.sum(cm["contact_map"]))
        distributions_generated[f"rescount20-{filtering}-trial_{trial}"] = np.array(rescounts_)

for filtering in ["uniform", "targeted"]:
    for trial in range(5):
        rescounts_ = list()
        data_ = joblib.load(os.path.join(data_dir, "intermediates",
                                      "generation_for_emily_sep14_2026", f"rescount50-{filtering}-trial_{trial}-generated_cm-contacts.joblib"))
        for cm in data_:
            rescounts_.append(np.sum(cm["contact_map"]))
        distributions_generated[f"rescount50-{filtering}-trial_{trial}"] = np.array(rescounts_)

for percent_p in [80]:
    for filtering in ["uniform", "targeted"]:
        rescounts_ = []
        data_ = [joblib.load(os.path.join(data_dir, "intermediates",
                                              "generated_proteins", f"{k}", f"rescount{percent_p}-{filtering}-generated_cm-contacts.joblib"))
                 for k in (4, 5, 6)]
        for trial in data_:
            for cm in trial:
                rescounts_.append(np.sum(cm["contact_map"]))
        for k in range(5):
            distributions_generated[f"rescount{percent_p}-{filtering}-trial_{k}"] = np.array(rescounts_)


colors = {1:"#4477AA", 2:"#EE6677", 3:"#228833"}
med_line_color = "red"
orig_color = "midnightblue"
gen_color = "forestgreen"
kde_alpha = 0.2







fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, squeeze=False, figsize=(8, 11.5))
gen_ax = []
max_val_y = [0, 0, 0]
max_gen_y = [0, 0, 0]
for i, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
    gen_ax.append(list())
    for j, trial in enumerate([0, 4]):
        signalg = distributions_generated[f"{ablation_p}-targeted-trial_{trial}"]
        signalo = distributions_original[f"{ablation_p}-targeted-trial_{trial}"]
        print(ablation_p, len(signalg))
        #signal_sampled = np.random.choice(signal, size=sample_size, replace=False)
        #kernel = scipy.stats.gaussian_kde(signal, bw_method="silverman")
        minx = min(np.min(signalg), np.min(signalo))
        maxx = max(np.max(signalg), np.max(signalo))
        no, binso, patcheso = ax[i, j].hist(signalo, bins=50, range=(0, 750), density=False, bottom=0, align='mid', color=orig_color, label="validation")
        genax = ax[i, j].twinx()
        ng, binsg, patchesg = genax.hist(signalg, bins=50, range=(0, 750), density=False, bottom=0, align='mid', color=gen_color, label="generated", alpha=0.8)
        #y = kernel([x])
        max_val_y[i] = max(max_val_y[i], np.max(no))
        max_gen_y[i] = max(max_gen_y[i], np.max(ng))
        ok = scipy.stats.gaussian_kde(signalo, bw_method="silverman")
        gk = scipy.stats.gaussian_kde(signalg, bw_method="silverman")
        do = ok([np.linspace(0, 50, num=1000)])
        dg = gk([np.linspace(0, 50, num=1000)])
        kl_div = entropy(do, qk=dg, axis=0)
        print(kl_div)
        #xmed = np.mean(signalg)
        #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
        #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
        #ax[i, j].vlines(xmed, ymin=0, ymax=1, label=f"$\\mu = {xmed:.3f}$", color=med_line_color, linestyle="--")
        #ax[i, j].legend()
        #ax[i, j].set_xlabel(f"interface-area ($\\AA^2$)")
        #ax[i, j].set_ylabel("counts")
        ax[i, j].set_xlim(left=0, right=750)
        gen_ax[-1].append(genax)
        label_text = r"$D_{KL} (\text{O} \parallel \text{G}) = " + f"{kl_div:.2f}$"
        ax[i, j].text(0.95, 0.95, label_text,
                      transform=ax[i, j].transAxes,
                      verticalalignment="top",
                      horizontalalignment="right",
                      fontsize="large",
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9))
ax[0, 0].set_ylabel("\\textbf{extensive ablation}\n\ncounts")
ax[1, 0].set_ylabel("\\textbf{moderate ablation}\n\ncounts")
ax[2, 0].set_ylabel("\\textbf{limited ablation}\n\ncounts")
ax[0, 0].set_title(r"trial 1")
ax[0, 1].set_title(r"trial 5")
ax[-1, 0].set_xlabel(f"res-res count")
ax[-1, 1].set_xlabel(f"res-res count")
for i in range(3):
    for j in range(2):
        ax[i, j].set_ylim(bottom=0, top=max_val_y[i]*1.1)
        ax[i, j].tick_params(axis='y', colors=orig_color)
for i in range(3):
    for j in range(2):
        gen_ax[i][j].set_ylim(bottom=0, top=max_gen_y[i]*1.1)
        gen_ax[i][j].tick_params(axis='y', colors=gen_color)
fig.suptitle("res-res count of generated protein complexes (targeted ablation)")
fig.tight_layout()
fig.savefig(f"rescount-targeted-generated-areas.png")
fig.savefig(f"rescount-targeted-generated-areas.pdf")



fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, squeeze=False, figsize=(8, 11.5))
gen_ax = []
max_val_y = [0, 0, 0]
max_gen_y = [0, 0, 0]
for i, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
    gen_ax.append(list())
    for j, trial in enumerate([0, 4]):
        signalg = distributions_generated[f"{ablation_p}-uniform-trial_{trial}"]
        signalo = distributions_original[f"{ablation_p}-uniform-trial_{trial}"]
        #signal_sampled = np.random.choice(signal, size=sample_size, replace=False)
        #kernel = scipy.stats.gaussian_kde(signal, bw_method="silverman")
        minx = min(np.min(signalg), np.min(signalo))
        maxx = max(np.max(signalg), np.max(signalo))
        no, binso, patcheso = ax[i, j].hist(signalo, bins=50, range=(0, 750), density=False, bottom=0, align='mid', color=orig_color, label="validation")
        genax = ax[i, j].twinx()
        ng, binsg, patchesg = genax.hist(signalg, bins=50, range=(0, 750), density=False, bottom=0, align='mid', color=gen_color, label="generated", alpha=0.8)
        #y = kernel([x])
        max_val_y[i] = max(max_val_y[i], np.max(no))
        max_gen_y[i] = max(max_gen_y[i], np.max(ng))
        ok = scipy.stats.gaussian_kde(signalo, bw_method="silverman")
        gk = scipy.stats.gaussian_kde(signalg, bw_method="silverman")
        do = ok([np.linspace(0, 50, num=1000)])
        dg = gk([np.linspace(0, 50, num=1000)])
        kl_div = entropy(do, qk=dg, axis=0)
        print(kl_div)
        #xmed = np.mean(signalg)
        #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
        #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
        #ax[i, j].vlines(xmed, ymin=0, ymax=1, label=f"$\\mu = {xmed:.3f}$", color=med_line_color, linestyle="--")
        #ax[i, j].legend()
        #ax[i, j].set_xlabel(f"interface-area ($\\AA^2$)")
        #ax[i, j].set_ylabel("counts")
        ax[i, j].set_xlim(left=0, right=750)
        gen_ax[-1].append(genax)
        label_text = r"$D_{KL} (\text{O} \parallel \text{G}) = " + f"{kl_div:.2f}$"
        ax[i, j].text(0.95, 0.95, label_text,
                      transform=ax[i, j].transAxes,
                      verticalalignment="top",
                      horizontalalignment="right",
                      fontsize="large",
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9))
ax[0, 0].set_ylabel("\\textbf{extensive ablation}\n\ncounts")
ax[1, 0].set_ylabel("\\textbf{moderate ablation}\n\ncounts")
ax[2, 0].set_ylabel("\\textbf{limited ablation}\n\ncounts")
ax[0, 0].set_title(r"trial 1")
ax[0, 1].set_title(r"trial 5")
ax[-1, 0].set_xlabel(f"res-res count")
ax[-1, 1].set_xlabel(f"res-res count")
for i in range(3):
    for j in range(2):
        ax[i, j].set_ylim(bottom=0, top=max_val_y[i]*1.1)
        ax[i, j].tick_params(axis='y', colors=orig_color)
for i in range(3):
    for j in range(2):
        gen_ax[i][j].set_ylim(bottom=0, top=max_gen_y[i]*1.1)
        gen_ax[i][j].tick_params(axis='y', colors=gen_color)
fig.suptitle("res-res counts of generated protein complexes (uniform ablation)")
fig.tight_layout()
fig.savefig(f"rescount-uniform-generated-areas.png")
fig.savefig(f"rescount-uniform-generated-areas.pdf")














# fig, ax = plt.subplots(nrows=3, ncols=5, sharex=False, sharey="row", squeeze=False, figsize=(11.5, 8))
# maxy = [0, 0, 0]
# for i, ablation_p in enumerate(["rescount20", "rescount50", "rescount80"]):
#     for j, trial in enumerate(range(5)):
#         signalg = distributions_generated[f"{ablation_p}-{filtering}-trial_{trial}"]
#         signalo = distributions_original[f"{ablation_p}-{filtering}-trial_{trial}"]
#         #signal_sampled = np.random.choice(signal, size=sample_size, replace=False)
#         #kernel = scipy.stats.gaussian_kde(signal, bw_method="silverman")
#         minx = min(np.min(signalg), np.min(signalo))
#         maxx = max(np.max(signalg), np.max(signalo))
#         no, binso, patcheso = ax[i, j].hist(signalo, bins=50, range=(0, 750), density=False, bottom=0, align='mid', color=orig_color, label="validation")
#         ng, binsg, patchesg = ax[i, j].hist(signalg, bins=50, range=(0, 750), density=False, bottom=0, align='mid', color=gen_color, label="generated", alpha=0.8)
#         #y = kernel([x])
#         maxy[i] = max(maxy[i], np.max(ng), np.max(no))
#         #xmed = np.mean(signalg)
#         #ax[i, j].plot(x, y, linewidth=2, color=kde_color)
#         #ax[i, j].fill_between(x, y, y2=0, alpha=kde_alpha, color=kde_color)
#         #ax[i, j].vlines(xmed, ymin=0, ymax=1, label=f"$\\mu = {xmed:.3f}$", color=med_line_color, linestyle="--")
#         #ax[i, j].legend()
#         #ax[i, j].set_xlabel(f"interface residue-residue interactions")
#         #ax[i, j].set_ylabel("counts")
#         ax[i, j].set_xlim(left=0, right=750)
#         #ax[i, j].set_xticks([0, 2500, 5000], labels=["0", "2500", "5000"], minor=False)
# ax[0, 0].set_ylabel("\\textbf{extensive ablation}\n\ncounts")
# ax[1, 0].set_ylabel("\\textbf{moderate ablation}\n\ncounts")
# ax[2, 0].set_ylabel("\\textbf{limited ablation}\n\ncounts")
# ax[0, 0].set_title(r"trial 1")
# ax[0, 1].set_title(r"trial 2")
# ax[0, 2].set_title(r"trial 3")
# ax[0, 3].set_title(r"trial 4")
# ax[0, 4].set_title(r"trial 5")
# ax[-1, 0].set_xlabel(f"res-res count")
# ax[-1, 1].set_xlabel(f"res-res count")
# ax[-1, 2].set_xlabel(f"res-res count")
# ax[-1, 3].set_xlabel(f"res-res count")
# ax[-1, 4].set_xlabel(f"res-res count")
# for i in range(3):
#     for j in range(5):
#         ax[i, j].set_ylim(bottom=0, top=maxy[i]*1.1)
# fig.suptitle("residue-residue counts of generated protein complexes")
# fig.tight_layout()
# fig.savefig(f"rescount-generated-areas.png")
# fig.savefig(f"rescount-generated-areas.pdf")





























sys.exit()

data = []
for filename in sys.argv[3:]:
    data.append(joblib.load(filename))

data = [np.array(list(x.values())) for x in data]
data = np.concat(data)

filtered = data[data > 0]
kernel = scipy.stats.gaussian_kde(filtered)

x = np.linspace(np.min(filtered), np.max(filtered), num=2000)
y = kernel([x])

fig, ax = plt.subplots()
n, bins, patches = ax.hist(filtered, bins=50, density=True, bottom=0, align="mid", color="midnightblue")
ax.plot(x, y, linewidth=2, color="red")
ax.fill_between(x, y1=y, y2=0, color="red", alpha=0.5)
med = np.mean(filtered)
ax.axvline(med, ymin=0, ymax=1, color="limegreen", linestyle="--", label=f"mean = {med:.3f}")
ax.legend()
ax.set_xlabel(r"interface-area ($\AA^2$)")
ax.set_ylabel("probability density")
ax.set_title(sys.argv[2])
fig.savefig(sys.argv[1], bbox_inches="tight")
