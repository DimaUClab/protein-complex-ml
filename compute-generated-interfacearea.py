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
import biotite
import dill
from esm.sdk.api import ESMProtein, ESM3InferenceClient
from esm.models.esm3 import ESM3
from huggingface_hub import login
import torch
import gc
from collections import defaultdict
import tempfile
import warnings
import subprocess
import mdtraj

rcParams["figure.dpi"] = 300
data_dir = os.environ["PROJ_DATASETS"]
script_dir = os.environ["PROJ_SCRIPTS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
proj_root = os.environ["PROJ_ROOT"]
model_dir = os.environ["PROJ_CM_MODELS"]
hf_read_token = os.environ["HUGGINGFACE_READ_TOKEN"]

with open(os.path.join(script_dir, "psf.tcl"), "r") as f:
    vmd_script = f.read()

ablation_map = {"interfacearea20":"20% retained", "interfacearea50":"50% retained", "interfacearea80":"80% retained",
                "rescount20":"20% retained", "rescount50":"50% retained", "rescount80":"80% retained"}

# Dict for converting three letter codes to one letter codes
D3TO1 = defaultdict((lambda :'X'), {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
         'SEC': 'U'})

AA_to_maxAcc = {"A":106,
                "C":135,
                "D":163,
                "E":194,
                "F":197,
                "G": 84,
                "H":184,
                "I":169,
                "K":205,
                "L":164,
                "M":188,
                "N":157,
                "P":136,
                "Q":198,
                "R":248,
                "S":130,
                "T":142,
                "V":142,
                "W":227,
                "Y":222,
                "X":166
                }

dips_meta = pd.read_csv(f"{data_dir}/dips_meta.dat", header=0, index_col=0, sep=' ')
MAX_SEQ_LEN = 650
with open(f"{model_dir}/pairs-postprocessed-train-duplicates-removed.txt", 'r') as f:
    all_train_keys = [line.strip() for line in f.readlines()]

train_keys = []
for complx in all_train_keys:
    if dips_meta.loc[complx, 'l_len'] > MAX_SEQ_LEN or dips_meta.loc[complx, 'r_len'] > MAX_SEQ_LEN:
        continue
    train_keys.append(complx)
    #

with open(f"{model_dir}/holdout_val_keys.txt", 'r') as f:
    all_hval_keys = [line.strip() for line in f.readlines()[1:]]

with open(f"{model_dir}/pairs-postprocessed-val.txt", 'r') as f:
    all_hval_keys.extend([line.strip() for line in f.readlines()])

hval_keys = []
for complx in all_hval_keys:
    if dips_meta.loc[complx, 'l_len'] > MAX_SEQ_LEN or dips_meta.loc[complx, 'r_len'] > MAX_SEQ_LEN:
        continue
    hval_keys.append(complx)
    #

def extract_atom37_coordinates(complx_dill, atom_name="CA", atom_ndx=1):
    len_df0 = len(complx_dill.sequences["l_b"])
    len_df1 = len(complx_dill.sequences["r_b"])
    df0_tensor = torch.full((len_df0, 37, 3), torch.nan)
    df1_tensor = torch.full((len_df1, 37, 3), torch.nan)
    df0 = complx_dill.df0.loc[complx_dill.df0["atom_name"] == atom_name, ["x", "y", "z"]]
    df1 = complx_dill.df1.loc[complx_dill.df1["atom_name"] == atom_name, ["x", "y", "z"]]
    for residue in range(df0_tensor.shape[0]):
        try:
            df0_tensor[residue, atom_ndx] = torch.tensor(df0.iloc[residue].to_numpy())
        except IndexError:
            df0_tensor[residue, atom_ndx] = torch.full((3,), torch.nan)
    for residue in range(df1_tensor.shape[0]):
        try:
            df1_tensor[residue, atom_ndx] = torch.tensor(df1.iloc[residue].to_numpy())
        except IndexError:
            df1_tensor[residue, atom_ndx] = torch.full((3,), torch.nan)
    chainbreak = torch.full((1, 37, 3), torch.nan)
    full_coordinates = torch.concatenate([df0_tensor, chainbreak, df1_tensor], axis=0)
    return full_coordinates

def extract_backbone_coordinates(complx_dill):
    n_coords = extract_atom37_coordinates(complx_dill, atom_name="N", atom_ndx=0)
    ca_coords = extract_atom37_coordinates(complx_dill, atom_name="CA", atom_ndx=1)
    c_coords = extract_atom37_coordinates(complx_dill, atom_name="C", atom_ndx=2)
    o_coords = extract_atom37_coordinates(complx_dill, atom_name="O", atom_ndx=4)
    backbone_coordinates = ca_coords
    backbone_coordinates[:, 0, :] = n_coords[:, 0, :]
    backbone_coordinates[:, 2, :] = c_coords[:, 2, :]
    backbone_coordinates[:, 4, :] = o_coords[:, 4, :]
    return backbone_coordinates

def extract_ss_values(complx_dill):
    ca_a = complx_dill.df0.loc[complx_dill.df0["atom_name"] == "CA", ["ss_value"]]
    if len(ca_a) != len(complx_dill.sequences["l_b"]):
        return None
    ca_b = complx_dill.df1.loc[complx_dill.df1["atom_name"] == "CA", ["ss_value"]]
    if len(ca_b) != len(complx_dill.sequences["r_b"]):
        return None
    ca_a.loc[ca_a["ss_value"] == "-", "ss_value"] = "<pad>"
    ca_b.loc[ca_b["ss_value"] == "-", "ss_value"] = "<pad>"
    ss_values = list(ca_a["ss_value"]) + ["<pad>"] + list(ca_b["ss_value"])
    assert len(ss_values) == len("|".join(complx_dill.sequences.values()))
    return "".join(ss_values)

def extract_sasa_values(complx_dill):
    ca_a = complx_dill.df0.loc[complx_dill.df0["atom_name"] == "CA", ["rsa_value", "resname"]]
    if len(ca_a) != len(complx_dill.sequences["l_b"]):
        return None
    ca_b = complx_dill.df1.loc[complx_dill.df1["atom_name"] == "CA", ["rsa_value", "resname"]]
    if len(ca_b) != len(complx_dill.sequences["r_b"]):
        return None
    ca_a["rescode"] = [D3TO1[rescode] for rescode in ca_a["resname"]]
    ca_b["rescode"] = [D3TO1[rescode] for rescode in ca_b["resname"]]
    coerced_a = pd.to_numeric(ca_a["rsa_value"], errors='coerce')
    coerced_b = pd.to_numeric(ca_b["rsa_value"], errors='coerce')
    ca_a["sasa"] = [AA_to_maxAcc[r[0]]*r[1] for r in zip(ca_a["rescode"], coerced_a)]
    ca_b["sasa"] = [AA_to_maxAcc[r[0]]*r[1] for r in zip(ca_b["rescode"], coerced_b)]
    sasa_values = list(ca_a["sasa"]) + ["<pad>"] + list(ca_b["sasa"])
    assert len(sasa_values) == len("|".join(complx_dill.sequences.values()))
    return sasa_values

def compute_pdb_interface_area(pdb):
    chainA_ndxs = pdb.top.select("chainid 0")
    chainB_ndxs = pdb.top.select("chainid 1")
    chainA_sasa = mdtraj.shrake_rupley(pdb.atom_slice(chainA_ndxs), mode="residue")
    chainB_sasa = mdtraj.shrake_rupley(pdb.atom_slice(chainB_ndxs), mode="residue")
    total_sasa = mdtraj.shrake_rupley(pdb, mode="residue")
    return 100 * 0.5 * (np.sum(chainA_sasa) + np.sum(chainB_sasa) - np.sum(total_sasa))

def compute_complx_interface_area(complx):
    with warnings.catch_warnings(action="ignore"):
        f = tempfile.NamedTemporaryFile(mode="w")
        f.write(complx.to_pdb_string())
    subprocess.run(f"vmd -dispdev text -e $PROJ_SCRIPTS/psf.tcl -pdb {f.name}", capture_output=False, shell=True)
    without_tmp = os.path.basename(f.name)
    try:
        pdb = mdtraj.load_pdb(f"{without_tmp}_autopsf.pdb")
    except FileNotFoundError:
        return -1
    subprocess.run([f"rm {without_tmp}_autopsf.pdb"], shell=True)
    subprocess.run([f"rm {without_tmp}_autopsf.psf"], shell=True)
    subprocess.run([f"rm {without_tmp}_autopsf.log"], shell=True)
    subprocess.run([f"rm {without_tmp}_autopsf_formatted.pdb"], shell=True)
    subprocess.run("sync")
    return compute_pdb_interface_area(pdb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-proteins", "-g", help="set of generated proteins", required=True, type=str)
    parser.add_argument("--generated-embeddings", "-e", help="generated protein embeddings", required=False, type=str, default="")
    parser.add_argument("--out-file", "-o", help="out filename", required=True, type=str)
    parser.add_argument("--embedding-model", "-m", help="model used to compute embeddings", type=str, required=False, choices=["esmc-600m-2024-12", "esmc-6b-2024-12"], default="esmc-600m-2024-12")
    args = parser.parse_args()
    
    generated_proteins = joblib.load(args.generated_proteins)["generated"]
    if args.generated_embeddings != "":
        generated_embeddings = joblib.load(args.generated_embeddings)
        embeddings = joblib.load(f"{data_dir}/intermediates/dips_plus-holdout_val-complex-embeddings-{args.embedding_model}.joblib")
    interface_areas = {}
    for i, seed_complx in tqdm(enumerate(generated_proteins.keys()), ascii=True, total=len(generated_proteins.keys())):
        if args.generated_embeddings != "":
            complx_generated_embedding_dists = []
            seed_embedding = np.mean(embeddings[seed_complx][0,...], axis=0)
            for embedding in generated_embeddings[seed_complx]:
                m_embedding = np.mean(embedding, axis=0)
                complx_generated_embedding_dists.append(scipy.spatial.distance.cosine(seed_embedding, m_embedding))
            #
            closest_candidate_ndx = np.argsort(complx_generated_embedding_dists)[0]
        else:
            ptms = [float(candidate.ptm.cpu()) for candidate in generated_proteins[seed_complx]]
            closest_candidate_ndx = np.argsort(ptms)[-1]
        closest_candidate = generated_proteins[seed_complx][closest_candidate_ndx]
        area = compute_complx_interface_area(closest_candidate)
        interface_areas[seed_complx] = area
        #
    #
    joblib.dump(interface_areas, args.out_file, compress=0)

    
