'''This should only need to be ran once, to read through the pos_idx entries and convert them into contact maps, then store those in a joblib file for fast loading later. Then every model can just use that.
'''

import sys
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
#add src_code to path
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src_code')))

from data_utils import get_complex_dill

DATA_ROOT = '../../data/db5_raw'
OUTFILE = 'db5_targets.joblib'
PAIRSTXTTRAIN = f'{DATA_ROOT}/pairs-postprocessed.txt'
PAIRSTXTVAL = None

with open(PAIRSTXTTRAIN, 'r') as f:
    lines = f.readlines()

if PAIRSTXTVAL:
    with open(PAIRSTXTVAL, 'r') as f:
        lines += f.readlines()
        
targets_dict = {}
failed = 0
for line in tqdm(lines, 'Processing targets', unit='file'):
    try:
        line = line.strip()
        path = f'{DATA_ROOT}/{line}'
        s = get_complex_dill(path)
        
        posidx = s.pos_idx #list of i,j tuples
        l1, l2 = len(s.sequences['l_b']), len(s.sequences['r_b']) #used to define the shape of the map
        
        #the pos_idx are actually CA indices in the full atom list df0, df1. I need to map to residue indices.
        df0_cas = s.df0[s.df0['atom_name'] == 'CA'].index.astype(int)
        df0_mapping = pd.Series(data=np.arange(len(df0_cas)), index=df0_cas)

        
        df1_cas = s.df1[s.df1['atom_name'] == 'CA'].index.astype(int)
        df1_mapping = pd.Series(data=np.arange(len(df1_cas)), index=df1_cas)

        
        contact_map = np.zeros((l1, l2), dtype=np.uint8)
        for i, j in posidx:
            res_i = df0_mapping[i]
            res_j = df1_mapping[j]
            contact_map[res_i, res_j] = 1
            
        contact_map_sparse = csr_matrix(contact_map)
        targets_dict[line] = contact_map_sparse
    except Exception as e:
        print(f"Error processing {line}: {e}")
        failed += 1
        continue    
joblib.dump(targets_dict, OUTFILE)
print(f"Saved targets to {OUTFILE}")
print(f"Failed to process {failed} targets")