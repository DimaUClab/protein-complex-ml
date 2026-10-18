from tqdm import tqdm
import time
import gc
import torch
import pandas as pd
import numpy as np
import sys
import argparse
import os
import scipy
import joblib
import esm
from esm.models.esmc import ESMC
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, GenerationConfig, ESMProtein, LogitsConfig, ESMProteinError
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk import batch_executor
import random
import math
#import util
from esm.sdk.experimental import ESM3GuidedDecoding, GuidedDecodingScoringFunction, ESM3GuidedDecodingWithConstraints, ConstraintType, GenerationConstraint
import dill
from collections import defaultdict

# from (1) Rost, B.; Sander, C. Conservation and Prediction of Solvent Accessibility in Protein Families. Proteins: Structure, Function, and Bioinformatics 1994, 20 (3), 216–226. https://doi.org/10.1002/prot.340200303.
# in angstrom ^ 2
AA_to_maxAcc = {"A":106,
                "C":135,
                "D":163,
                "E":194,
                "F":197,
                "G":84,
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
                "Y":222
                }

# from https://github.com/Biohub/esm/blob/main/esm/utils/constants/esm3.py#L77

SASA_PAD_TOKEN = 0

SASA_DISCRETIZATION_BOUNDARIES = [
    0.8,
    4.0,
    9.6,
    16.4,
    24.5,
    32.9,
    42.0,
    51.5,
    61.2,
    70.9,
    81.6,
    93.3,
    107.2,
    125.4,
    151.4,
]

# Dict for converting three letter codes to one letter codes
D3TO1 = defaultdict((lambda :'X'), {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
         'SEC': 'U'})



ALL_AAS = 'GSAPDNTEKQCVHIRLMYFW'

# frequencies from supplemental information of (1) Lobanov, M. Yu.; Surin, A. A.; Galzitskaya, O. V. What Can Be Learned
# by Knowing Only the Amino Acid Composition of Proteins? International Journal of Molecular Sciences 2024, 25 (24),
# 13680. https://doi.org/10.3390/ijms252413680.

AA_CODES = """C M F I L V W Y A G T S Q N E D H R K P""".split()
AA_PROBS = np.array([0.06146967, 0.05592020, 0.05050506, 0.04350903, 0.03991305,
                     0.04594497, 0.05351416, 0.07171458, 0.04136610, 0.04270310,
                     0.05857854, 0.04680383, 0.05046692, 0.05351796, 0.04060263,
                     0.04913945, 0.06421999, 0.04527684, 0.04111708, 0.04371685])


hf_read_token = "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"
es_forge_token = "49rQGwSdWwTrQcpOduZt7n"
MAX_SEQ_LEN = 650
GENERATION_TEMP = 0.3
CHECKPOINT_RATE=20
#

#train_keys, val_keys, holdout_keys = util.load_dips_plus_keys(model_dir, dips_meta, MAX_SEQ_LEN)



########################################


# previously_masked
# 0     ,     1
mask_p = 0
add_p  = 1
del_p  = 2
mut_p  = 3
no_op  = 4
operations = [mask_p, add_p, del_p, mut_p, no_op]

probs = [[0.100, 0.30], # mask_p
         [0.010, 0.20], # add_p
         [0.010, 0.20], # del_p
         [0.001, 0.00], # mut_p
         [0.879, 0.30]] # no_op
probs = np.array(probs)










def get_random_aa():
    aa_ndx = np.random.choice(range(len(AA_CODES)), p=AA_PROBS)
    return AA_CODES[aa_ndx]



def mask_complex(seed_prot: ESMProtein, max_num_steps=300):
    fuzzed_prot = ESMProtein()
    fuzzed_prot.sequence = list()
    fuzzed_prot.secondary_structure = list()
    fuzzed_prot.sasa = list()
    previously_masked = 0
    l_ndx = 0
    running = True
    while running:
        if seed_prot.sequence[l_ndx] == "|":
            fuzzed_prot.sequence.append("|")
            fuzzed_prot.secondary_structure.append("_")
            fuzzed_prot.sasa.append("<pad>")
            l_ndx = l_ndx + 1
            continue
        #
        choice = np.random.choice(operations, p=probs[:,previously_masked])
        if choice == mask_p:
            fuzzed_prot.sequence.append("_")
            fuzzed_prot.secondary_structure.append("_")
            fuzzed_prot.sasa.append("<pad>")
            previously_masked = 1
            l_ndx = l_ndx + 1
            #
        elif choice == add_p:
            fuzzed_prot.sequence.append("_")
            fuzzed_prot.secondary_structure.append("_")
            fuzzed_prot.sasa.append("<pad>")
            previously_masked = 1
            #
        elif choice == del_p:
            previously_masked = 0
            l_ndx = l_ndx + 1
            #
        elif choice == no_op:
            fuzzed_prot.sequence.append(seed_prot.sequence[l_ndx])
            fuzzed_prot.secondary_structure.append(seed_prot.secondary_structure[l_ndx])
            fuzzed_prot.sasa.append(seed_prot.sasa[l_ndx])
            previously_masked = 0
            l_ndx = l_ndx + 1
            #
        elif choice == mut_p:
            fuzzed_prot.sequence.append(get_random_aa())
            fuzzed_prot.secondary_structure.append("_")
            fuzzed_prot.sasa.append("<pad>")
            previously_masked = 0
            l_ndx = l_ndx + 1
            #
        if l_ndx >= len(seed_prot.sequence):
            running = False
    fuzzed_prot.sequence = ''.join(fuzzed_prot.sequence)
    fuzzed_prot.secondary_structure = ''.join(fuzzed_prot.secondary_structure)
    assert "|" in fuzzed_prot.sequence
    num_steps = sum([(lambda x: 1 if x == '_' else 0)(aa) for aa in fuzzed_prot.sequence])
    return fuzzed_prot, num_steps


def generate_new_protein(client, sequence, num_steps):
    masked_protein = ESMProtein(sequence=sequence)
    generated_protein = client.generate(masked_protein, GenerationConfig(track="sequence", num_steps=num_steps, temperature=GENERATION_TEMP))
    return generated_protein


def generate_new_protein_guided(ptm_guided_decoder, prompt_prot, num_steps):
    num_decoding_steps = num_steps//3
    if num_decoding_steps == 0:
        num_decoding_steps=num_steps
        num_samples_per_step=1
    else:
        num_samples_per_step = 3
    generated_protein = ptm_guided_decoder.guided_generate(
        protein=prompt_prot,
        num_decoding_steps=num_decoding_steps,
        num_samples_per_step=num_samples_per_step,
        track='sequence',
        denoised_prediction_temperature=GENERATION_TEMP
    )
    return generated_protein


def embed_gen_protein(client, protein):
    prot_tensor = client.encode(protein)
    prot_logits = client.logits(prot_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    prot_embeddings = prot_logits.embeddings.float().cpu().numpy()[0,...]
    return prot_embeddings


def check_sequence_of_concern(attempt_gen_proteins):
    for p in attempt_gen_proteins:
        if isinstance(p, (ESMProteinError, TypeError)):
            return True
    return False


# Create scoring function (e.g. PTM scoring function)
class PTMScoringFunction(GuidedDecodingScoringFunction):
    def __call__(self, candidate_protein: ESMProtein) -> float:
        total_score = float(candidate_protein.ptm)
        return total_score
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
    return backbone_coordinates

def extract_ss_values(complx_dill):
    ca_a = complx_dill.df0.loc[complx_dill.df0["atom_name"] == "CA", ["ss_value"]]
    if len(ca_a) != len(complx_dill.sequences["l_b"]):
        return None
    ca_b = complx_dill.df1.loc[complx_dill.df1["atom_name"] == "CA", ["ss_value"]]
    if len(ca_b) != len(complx_dill.sequences["r_b"]):
        return None
    ca_a.loc[ca_a["ss_value"] == "-", "ss_value"] = "_"
    ca_b.loc[ca_b["ss_value"] == "-", "ss_value"] = "_"
    ss_values = list(ca_a["ss_value"]) + ["_"] + list(ca_b["ss_value"])
    assert len(ss_values) == len("|".join(complx_dill.sequences.values()))
    return ''.join(ss_values)

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
    for i, value in enumerate(sasa_values):
        if value == "<pad>":
            continue
        if np.isnan(value):
            sasa_values[i] = "<pad>"
    assert len(sasa_values) == len("|".join(complx_dill.sequences.values()))
    return sasa_values

def gen_complex_prompts(complx_dill, n):
    complx_seq = "|".join(complx_dill.sequences.values())
    seed_prot = ESMProtein(sequence=complx_seq)
    seed_prot.secondary_structure = extract_ss_values(complx_dill)
    seed_prot.sasa = extract_sasa_values(complx_dill)
    prompts = [mask_complex(seed_prot) for j in range(n)]
    return prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dissimilarity', '-d', type=str, required=True, help='complex similarity table')
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--generate', '-g', type=int, required=False, default=1, help="whether to generate new proteins or just embed them")
    parser.add_argument('--percent', '-p', type=int, required=True, help="percent ablation")
    
    max_marker = 50
    args = parser.parse_args()
    trial_df = pd.read_csv(args.dissimilarity, header=0, index_col=0, sep=' ')
    quantile = 0.2
    trial_df = trial_df.sort_values(by="LinearCKA", axis=0, ascending=True)
    dissimilar_quantile = np.quantile(trial_df['LinearCKA'], quantile, axis=None, method='linear')
    trial_dissimilar = list(trial_df[trial_df['LinearCKA'] <= dissimilar_quantile].index)
    
    dips_meta = pd.read_csv(f"dips_meta.dat", header=0, index_col=0, sep=' ')
    
    login(token=hf_read_token)
    torch.serialization.add_safe_globals([ESM3])
    try:
        client_esm3 = torch.load('esm3-open.pt', weights_only=False)
    except FileNotFoundError:
        #This will download the model weights and instantiate the model on your machine.
        client_esm3: ESM3InferenceClient = ESM3.from_pretrained().to("cuda")
        torch.save(client_esm3, './esm3-open.pt')
    
    client_esmc = ESM3ForgeInferenceClient(
        model="esmc-6b-2024-12",
        url="https://biohub.ai",
        token=es_forge_token,
        min_retry_wait=2,
        max_retry_wait=20,
        max_retry_attempts=10
    )

    ptm_constraint = GenerationConstraint(
        scoring_function=PTMScoringFunction(),
        constraint_type=ConstraintType.GREATER_EQUAL,
        value=0.40
    )
    
    ptm_guided_decoder = ESM3GuidedDecodingWithConstraints(
        client=client_esm3, scoring_function=PTMScoringFunction(),
        constraints=[ptm_constraint]
    )
    
    # 10% of validation set is about 166 for the 20% reduction, 415 for 50%, and 663 for 80%
    generate_num = {20:166, 50:415, 80:663}
    num = generate_num[args.percent]
    if args.generate:
        generated_proteins = {'generated':dict(), 'prompts':dict()}
        print('Generating new proteins...')
        with torch.no_grad():
            for k, complx_name in tqdm(enumerate(trial_dissimilar[:num]), ascii=True):
                with open(f"raw/{complx_name}", "rb") as f:
                    complx_dill = dill.load(f)
                complx_gen_proteins = []
                target_count = 5
                complx_prompts = []
                while len(complx_gen_proteins) < target_count:
                    remaining_proteins = target_count - len(complx_gen_proteins)
                    complx_iteration_prompts = gen_complex_prompts(complx_dill, remaining_proteins)
                    attempt_gen_proteins = [generate_new_protein_guided(ptm_guided_decoder, complx_iteration_prompts[i][0], complx_iteration_prompts[i][1]) for i in range(remaining_proteins)]
                    print(attempt_gen_proteins)
                    # if check_sequence_of_concern(attempt_gen_proteins):
                    #     complx_prompts.extend(fuzzed_complex_seqs)
                    #     complx_gen_proteins.extend(attempt_gen_proteins)
                    #     break
                    def valid_protein(prot):
                        if isinstance(prot, ESMProteinError):
                            return False
                        return True
                    valid_ndxs = [ndx for ndx, p in enumerate(attempt_gen_proteins) if valid_protein(p)]
                    for ndx in valid_ndxs:
                        complx_gen_proteins.append(attempt_gen_proteins[ndx])
                        complx_prompts.append(complx_iteration_prompts[ndx][0])
                    gc.collect()
                    torch.cuda.empty_cache()
                generated_proteins['generated'][complx_name] = complx_gen_proteins
                generated_proteins['prompts'][complx_name] = complx_prompts
                if k%CHECKPOINT_RATE == 0:
                    joblib.dump(generated_proteins, f'{args.output}-generated_proteins.joblib', compress=('zlib', 3))
        joblib.dump(generated_proteins, f'{args.output}-generated_proteins.joblib', compress=('zlib', 3))
    else:
        generated_proteins = joblib.load(f'{args.output}-generated_proteins.joblib')
    print("Embedding generated proteins")
    generated_embeddings = {}
    with torch.no_grad():
        for complx in tqdm(list(generated_proteins['generated'].keys()), ascii=True):
            prots = generated_proteins['generated'][complx]
            embeddings = []
            for prot in prots:
                prot_ESMProtein = ESMProtein(sequence=prot.sequence)
                embeddings.append(embed_gen_protein(client_esmc, prot_ESMProtein))
            generated_embeddings[complx] = embeddings
    joblib.dump(generated_embeddings, f'{args.output}-generated_embeddings.joblib', compress=('zlib', 3))
    sys.exit()
