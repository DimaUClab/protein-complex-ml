import mdtraj
import joblib
import numpy as np
import pandas as pd
import os
import argparse
import io
from itertools import product
import scipy
from tqdm import tqdm
import subprocess
import warnings

data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-proteins", "-g", help="set of generated proteins", required=True, type=str)
    parser.add_argument("--generated-embeddings", "-e", help="generated protein embeddings", required=True, type=str)
    parser.add_argument("--model", "-m", help="which embedding model", choices=["esmc-600m-2024-12", "esmc-6b-2024-12"], type=str, required=False,
                        default="esmc-600m-2024-12")
    parser.add_argument("--out-file", "-o", help="out directory", required=True, type=str)

    args = parser.parse_args()

    generated_proteins = joblib.load(args.generated_proteins)["generated"]
    generated_embeddings = joblib.load(args.generated_embeddings)
    embeddings = joblib.load(f"{data_dir}/intermediates/dips_plus-holdout_val-complex-embeddings-{args.model}.joblib")
    computed_cm = []
    for i, seed_complx in tqdm(enumerate(generated_proteins.keys()), ascii=True, total=len(generated_proteins.keys())):
        complx_generated_embedding_dists = []
        seed_embedding = np.mean(embeddings[seed_complx][0,...], axis=0)
        for embedding in generated_embeddings[seed_complx]:
            m_embedding = np.mean(embedding, axis=0)
            complx_generated_embedding_dists.append(scipy.spatial.distance.cosine(seed_embedding, m_embedding))
        #
        closest_candidate_ndx = np.argsort(complx_generated_embedding_dists)[0]
        closest_candidate = generated_proteins[seed_complx][closest_candidate_ndx]
        seed_complx_nonpath = ''.join([(lambda x: x if x != "/" else "_")(j) for j in seed_complx])
        with warnings.catch_warnings(action="ignore"):
            closest_candidate.to_pdb(f"{args.out_file}/gen_complx_{seed_complx_nonpath}.pdb")
        subprocess.run("sync")
        #
    #
