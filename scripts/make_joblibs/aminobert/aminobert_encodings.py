'''
Main script in this folder, generates the s1 and s2 AminoBERT encodings for the data
'''

import joblib
import dill
from typing import List, Dict, Any
from tqdm import tqdm
import tokenization
import numpy as np
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src_code')))

from data_utils import get_complex_dill

TOKENIZER = tokenization.FullTokenizer(k=1, token_to_replace_with_mask='X')
DATA_ROOT = "../../../data/dipsplus_raw"
MODEL_NAME = "aminobert"
DATASET_NAME = "dipsplus_raw"
TARGET_JOBLIB = '../../../data/dipsplus_raw/dipsplus_targets.joblib'

out_name1 = f'./{MODEL_NAME}_{DATASET_NAME}_s1_encodings.joblib'
out_name2 = f'./{MODEL_NAME}_{DATASET_NAME}_s2_encodings.joblib'


#####Functions (Some Copied from get_aminobert_embedding.py)#####

def find_dill_files(root_dir: str) -> List[str]:
    dill_file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".dill"):
                dill_file_paths.append(os.path.abspath(os.path.join(dirpath, filename)))
    return dill_file_paths


def generate_input_features_from_seq(seq, labels, tokenizer):
    '''
    function adapted from get_aminobert_embedding.py

    TODO where is acknowledgement to original source code listed? in paper or github?
    
    - padding unnecessary because padding is performed dynamically, removed from function.
    - function originally operates on list of sequences, here its changed to individual sequences to match structure of other models
    - model must return a dictionary of 'input_ids', 'token_type_ids', and 'attention_mask'. The latter two are all 0s and all ones respectively.
    - outputs are all pytorch tensors of shape (1, seq_len)
    '''
    tseq = [tokenization.CLS_TOKEN] + tokenizer.tokenize(seq)
    token_type_ids = [0] * len(tseq)
    input_mask = [1] * len(tseq)    
            
    input_ids = tokenizer.convert_tokens_to_ids(tseq)
    
    input_dict = {'input_ids': torch.tensor([input_ids], dtype=torch.int64),
                  'token_type_ids': torch.tensor([token_type_ids], dtype=torch.int64),
                  'attention_mask': torch.tensor([input_mask], dtype=torch.int64)}
            
    return input_dict



#####################################################################
paths = find_dill_files(DATA_ROOT)
legal_paths = list(joblib.load(TARGET_JOBLIB).keys())
print(legal_paths[:5])
print(paths[:5])
s1_encodings = {}
s2_encodings = {}

for path in tqdm(paths):
    reduced_path = '/'.join(path.split('/')[-2:])
    if reduced_path not in legal_paths:
        continue
    else:
        s = get_complex_dill(path)
        
        seq1 = s.sequences['l_b'].replace(" ", "") + '*' #the star seems to be the Aminobert end token
        seq2 = s.sequences['r_b'].replace(" ", "") + '*'
        
        s1_encodings[reduced_path] = generate_input_features_from_seq(seq1, labels=None, tokenizer=TOKENIZER)
        s2_encodings[reduced_path] = generate_input_features_from_seq(seq2, labels=None, tokenizer=TOKENIZER)

print(s1_encodings[legal_paths[0]])
joblib.dump(s1_encodings, out_name1)
joblib.dump(s2_encodings, out_name2)
        
    
    