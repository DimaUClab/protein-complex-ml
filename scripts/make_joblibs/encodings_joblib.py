import joblib
from transformers import EsmTokenizer, BertTokenizer
import dill
from typing import List, Dict, Any
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src_code')))

from data_utils import get_complex_dill

TOKENIZER = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
DATA_ROOT = "../../data/dipsplus_raw"
MODEL_NAME = "prot_bert"
DATASET_NAME = "dipsplus_raw"
TARGET_JOBLIB = '../../data/dipsplus_raw/dipsplus_targets.joblib'

out_name1 = f'./{MODEL_NAME}_{DATASET_NAME}_s1_encodings.joblib'
out_name2 = f'./{MODEL_NAME}_{DATASET_NAME}_s2_encodings.joblib'

def find_dill_files(root_dir: str) -> List[str]:
    dill_file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".dill"):
                dill_file_paths.append(os.path.abspath(os.path.join(dirpath, filename)))
    return dill_file_paths

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
        
        seq1 = s.sequences['l_b']
        seq2 = s.sequences['r_b']
        
        s1_encodings[reduced_path] = TOKENIZER(" ".join(list(seq1)), return_tensors='pt', padding=True, truncation=True)
        s2_encodings[reduced_path] = TOKENIZER(" ".join(list(seq2)), return_tensors='pt', padding=True, truncation=True)
        
joblib.dump(s1_encodings, out_name1)
joblib.dump(s2_encodings, out_name2)
        
    
    