import argparse
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, skew


model_dir = os.environ["PROJ_CM_MODELS"]
data_dir = os.environ["PROJ_DATASETS"]
art_dir = os.environ["PROJ_ARTIFACTS"]
proj_root = os.environ["PROJ_ROOT"]


similarity_dir = os.path.join(data_dir, "intermediates", "similarity_data")

def keyword_replace(keyword):
    if keyword == 'interfacearea':
        return 'interface-area'
    if keyword == 'rescount':
        return 'residue-residue count'


#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", "-d", type=str, required=True)
    parser.add_argument("--out-dir", "-o", type=str, required=True)
    args = parser.parse_args()
    
    skewnesses = pd.DataFrame(columns=["skewness"])
    for row_ndx, filtertype in enumerate(["interfacearea", "rescount"]):
        for col_ndx, reduction_p in enumerate(["20", "50", "80"]):
            train_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_0-train-similarities.dat"),
                                     header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_{
                trial}-train-similarities.dat"), header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                train_data = pd.concat([train_data, trial_data], axis=0, verify_integrity=False)
            train_data = train_data.groupby('complex ID').agg({'LinearCKA': 'mean'})
            
            val_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_0-val-similarities.dat"),
                                    header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_{
                                          trial}-val-similarities.dat"), header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                val_data = pd.concat([val_data, trial_data], axis=0, verify_integrity=False)
            val_data = val_data.groupby('complex ID').agg({'LinearCKA': 'mean'})

            
            holdout_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_0-holdout-similarities.dat"),
                                        header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-uniform-trial_{trial}-holdout-similarities.dat"),
                                          header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                holdout_data = pd.concat([holdout_data, trial_data], axis=0, verify_integrity=False)
            holdout_data = holdout_data.groupby('complex ID').agg({'LinearCKA': 'mean'})

            train_index = f"{filtertype}{reduction_p}-targeted-train"
            val_index = f"{filtertype}{reduction_p}-targeted-validation"
            test_index = f"{filtertype}{reduction_p}-targeted-test"
            skewnesses.loc[train_index] = skew(train_data['LinearCKA'].dropna())
            skewnesses.loc[val_index] = skew(val_data['LinearCKA'].dropna())
            skewnesses.loc[test_index] = skew(holdout_data['LinearCKA'].dropna())
    
    
    for row_ndx, filtertype in enumerate(["interfacearea", "rescount"]):
        for col_ndx, reduction_p in enumerate(["20", "50", "80"]):
            train_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_0-train-similarities.dat"),
                                      header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_{
                                          trial}-train-similarities.dat"), header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                train_data = pd.concat([train_data, trial_data], axis=0, verify_integrity=False)
            train_data = train_data.groupby('complex ID').agg({'LinearCKA': 'mean'})


            val_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_0-val-similarities.dat"),
                                    header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_{
                                          trial}-val-similarities.dat"), header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                val_data = pd.concat([val_data, trial_data], axis=0, verify_integrity=False)
            val_data = val_data.groupby('complex ID').agg({'LinearCKA': 'mean'})

            holdout_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_0-holdout-similarities.dat"),
                                        header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
            for trial in range(1, 5):
                trial_data = pd.read_csv(os.path.join(similarity_dir, "similarities", f"aminobert-esm2-{filtertype}{reduction_p}-targeted-trial_{
                                          trial}-holdout-similarities.dat"), header=0, index_col=None, sep=' ', names=['complex ID', 'SAS', 'LinearCKA', 'Frobenius', 'Nuclear'])
                holdout_data = pd.concat([holdout_data, trial_data], axis=0, verify_integrity=False)
            holdout_data = holdout_data.groupby('complex ID').agg({'LinearCKA': 'mean'})
            train_index = f"{filtertype}{reduction_p}-uniform-train"
            val_index = f"{filtertype}{reduction_p}-uniform-validation"
            test_index = f"{filtertype}{reduction_p}-uniform-test"
            skewnesses.loc[train_index] = skew(train_data['LinearCKA'].dropna())
            skewnesses.loc[val_index] = skew(val_data['LinearCKA'].dropna())
            skewnesses.loc[test_index] = skew(holdout_data['LinearCKA'].dropna())
    

    print(skewnesses)
    skewnesses.to_csv(os.path.join(args.out_dir, args.distribution), index=True, header=True, sep='\t')
