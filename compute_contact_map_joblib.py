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
import tempfile

data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]



def compute_ca_contacts(protein_complex, dist_cutoff=0.45):
    chains = [c for c in protein_complex.top.chains]
    A_residues = [r.index for r in chains[0].residues]
    B_residues = [r.index for r in chains[1].residues]
    contacts_to_compute = [p for p in product(A_residues, B_residues)]
    distances = mdtraj.compute_contacts(protein_complex, contacts=contacts_to_compute, scheme="ca", soft_min=False)
    contacts = mdtraj.geometry.squareform(*distances)[0,...]
    contacts = contacts[:len(A_residues), len(A_residues):]
    assert contacts.shape == (len(A_residues), len(B_residues)), f"{contacts.shape}"
    bool_map = (lambda x: x <= dist_cutoff)(contacts)
    contact_map = np.zeros(bool_map.shape)
    contact_map[bool_map] = 1
    return scipy.sparse.csr_array(contact_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-proteins", "-g", help="set of generated proteins", required=True, type=str)
    parser.add_argument("--generated-embeddings", "-e", help="generated protein embeddings", required=False, type=str, default="")
    parser.add_argument("--out-file", "-o", help="out filename", required=True, type=str)
    parser.add_argument("--dist-cutoff", "-d", help="maximum distance in nm between alpha carbons for two residues to make a contact.", type=float, required=False, default=0.6)
    parser.add_argument("--embedding-model", "-m", help="model used to compute embeddings", type=str, required=False, choices=["esmc-600m-2024-12", "esmc-6b-2024-12"], default="esmc-600m-2024-12")
    parser.add_argument("--candidate-selection", "-c", help="what criterion is used to select protein candidate", type=str, required=False, choices=["ptm", "embeddings", "contacts"], default="ptm")
    args = parser.parse_args()

    generated_proteins = joblib.load(args.generated_proteins)["generated"]
    if args.candidate_selection == "embeddings":
        generated_embeddings = joblib.load(args.generated_embeddings)
        embeddings = joblib.load(f"{data_dir}/intermediates/dips_plus-holdout_val-complex-embeddings-{args.embedding_model}.joblib")
    computed_cm = []
    for i, seed_complx in tqdm(enumerate(generated_proteins.keys()), ascii=True, total=len(generated_proteins.keys())):
        if args.candidate_selection == "embeddings":
            complx_generated_embedding_dists = []
            seed_embedding = np.mean(embeddings[seed_complx][0,...], axis=0)
            for embedding in generated_embeddings[seed_complx]:
                m_embedding = np.mean(embedding, axis=0)
                complx_generated_embedding_dists.append(scipy.spatial.distance.cosine(seed_embedding, m_embedding))
            #
            closest_candidate_ndx = np.argsort(complx_generated_embedding_dists)[0]
        elif args.candidate_selection == "ptm":
            ptms = [float(candidate.ptm.cpu()) for candidate in generated_proteins[seed_complx]]
            closest_candidate_ndx = np.argsort(ptms)[-1]
        elif args.candidate_selection == "contacts":
            cms = []
            for candidate in generated_proteins[seed_complx]:
                with warnings.catch_warnings(action="ignore"):
                    f = tempfile.NamedTemporaryFile(mode="w")
                    f.write(candidate.to_pdb_string())
                complx_mdtraj = mdtraj.load_pdb(f.name, standard_names=True)
                cm = compute_ca_contacts(complx_mdtraj, dist_cutoff=args.dist_cutoff)
                cms.append(cm)
            closest_candidate_ndx = np.argsort([np.sum(cm) for cm in cms])[-1]
            #
        closest_candidate = generated_proteins[seed_complx][closest_candidate_ndx]
        with warnings.catch_warnings(action="ignore"):
            f = tempfile.NamedTemporaryFile(mode="w")
            f.write(closest_candidate.to_pdb_string())
        complx_mdtraj = mdtraj.load_pdb(f.name, standard_names=True)
        cm = compute_ca_contacts(complx_mdtraj, dist_cutoff=args.dist_cutoff)
        computed_cm.append({"seed":seed_complx, "sequence":closest_candidate.sequence, "contact_map":cm})
        #
    #
    assert not os.path.isfile(args.out_file)
    joblib.dump(computed_cm, args.out_file, compress=0)
