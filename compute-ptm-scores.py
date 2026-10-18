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

rcParams["figure.dpi"] = 300
data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
ablation_map = {"interfacearea20":"20% retained", "interfacearea50":"50% retained", "interfacearea80":"80% retained",
                "rescount20":"20% retained", "rescount50":"50% retained", "rescount80":"80% retained"}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", "-g", help="name of output file", required=True, type=str, default='')
    parser.add_argument("--out-dir", "-o", help="output directory", required=True, type=str)
    parser.add_argument("--embedding-model", "-e", help="embedding model", required=False, type=str, default="esmc-600m-2024-12")
    parser.add_argument("--in-file", "-f", help="input generated proteins", required=False, type=str, default='')
    parser.add_argument("--in-embeddings", "-m", help="input generated embeddings", required=False, type=str, default='')
    args = parser.parse_args()

    embeddings = joblib.load(os.path.join(data_dir, "intermediates", f"dips_plus-holdout_val-complex-embeddings-{args.embedding_model}.joblib"))
    if args.in_file == '':
        args.replicate = os.path.split(os.path.split(args.out_dir)[0])[1]
        args.generated_proteins = os.path.join(data_dir, "intermediates", "generated_proteins", args.replicate)
        print(args.generated_proteins)
        ptm_distribution = {}
        for ablation_type in ["interfacearea", "rescount"]:
            for percent_p in [20, 50, 80]:
                for filtering in ["uniform", "targeted"]:
                    ptm_distribution[f"{ablation_type}{percent_p}-{filtering}"] = []
                    generated_proteins = joblib.load(os.path.join(args.generated_proteins, f"esmc-val-{ablation_type}{percent_p}-{filtering}-generated_proteins.joblib"))["generated"]
                    generated_embeddings = joblib.load(os.path.join(args.generated_proteins, args.embedding_model, f"esmc-val-{ablation_type}{percent_p}-{filtering}-generated_embeddings.joblib"))
                    for i, seed_complx in tqdm(enumerate(generated_proteins.keys()), ascii=True):
                        complx_generated_embedding_dists = []
                        seed_embedding = np.mean(embeddings[seed_complx][0,...], axis=0)
                        for embedding in generated_embeddings[seed_complx]:
                            m_embedding = np.mean(embedding, axis=0)
                            complx_generated_embedding_dists.append(scipy.spatial.distance.cosine(seed_embedding, m_embedding))
                        #
                        closest_candidate_ndx = np.argsort(complx_generated_embedding_dists)[0]
                        closest_candidate = generated_proteins[seed_complx][closest_candidate_ndx]
                        ptm_distribution[f"{ablation_type}{percent_p}-{filtering}"].append(closest_candidate.ptm.cpu().numpy())
                    #
                #
            #
        #
        for key in ptm_distribution.keys():
            ptm_distribution[key] = np.array(ptm_distribution[key]).flatten()
        joblib.dump(ptm_distribution, os.path.join(args.out_dir, args.out_file), compress=("zlib", 3))
    else:
        generated_proteins = joblib.load(args.in_file)["generated"]
        generated_embeddings = joblib.load(args.in_embeddings)
        ptm_distribution = []
        for i, seed_complx in tqdm(enumerate(generated_proteins.keys()), ascii=True):
            complx_generated_embedding_dists = []
            seed_embedding = np.mean(embeddings[seed_complx][0,...], axis=0)
            for embedding in generated_embeddings[seed_complx]:
                m_embedding = np.mean(embedding, axis=0)
                complx_generated_embedding_dists.append(scipy.spatial.distance.cosine(seed_embedding, m_embedding))
            #
            closest_candidate_ndx = np.argsort(complx_generated_embedding_dists)[0]
            closest_candidate = generated_proteins[seed_complx][closest_candidate_ndx]
            ptm_distribution.append(closest_candidate.ptm.cpu().numpy().flatten())
        joblib.dump(ptm_distribution, os.path.join(args.out_dir, args.out_file), compress=("zlib", 3))
    #
