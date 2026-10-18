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
rcParams["figure.dpi"] = 300
data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
proj_root = os.environ["PROJ_ROOT"]
model_dir = os.environ["PROJ_CM_MODELS"]
hf_read_token = os.environ["HUGGINGFACE_READ_TOKEN"]

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

def compute_complx_ptm(complx_dill, client):
    prot = ESMProtein()
    prot.sequence = "|".join(complx_dill.sequences.values())
    prot.sasa = extract_sasa_values(complx_dill)
    prot.ss = extract_ss_values(complx_dill)
    prot.coordinates = extract_backbone_coordinates(complx_dill)
    logits = client.encode(prot)
    decoded = client.decode(logits)
    return decoded.ptm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", "-g", help="name of output file", required=True, type=str)
    login(token=hf_read_token)
    torch.serialization.add_safe_globals([ESM3])
    try:
        client_esm3 = torch.load('esm3-open.pt', weights_only=False)
    except FileNotFoundError:
        #This will download the model weights and instantiate the model on your machine.
        client_esm3: ESM3InferenceClient = ESM3.from_pretrained().to("cuda")
    
    args = parser.parse_args()
    dips_prefix = os.path.join(proj_root, "external_tools", "DIPS_plus", "5134732", "raw")
    ptm_scores = {}
    for complx_name in tqdm(train_keys+hval_keys, ascii=True):
        complx_path = os.path.join(dips_prefix, complx_name)
        with open(complx_path, "rb") as f:
            complx_dill = dill.load(f)
        ptm_scores[complx_name] = compute_complx_ptm(complx_dill, client_esm3)
        torch.cuda.empty_cache()
        gc.collect()
    joblib.dump(ptm_scores, args.out_file, compress=("zlib", 3))
        
        
    
