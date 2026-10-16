#I'm processing the data from Hayden to fit it into a format like the rest of the data_utils functions expect. This is a one-time script to generate the embeddings for the generated proteins. The embeddings will be saved in a format that can be used by the rest of the pipeline.

import joblib
from transformers import EsmTokenizer, BertTokenizer
from typing import List, Dict, Any
from tqdm import tqdm

import sys
import os
import pathlib

TOKENIZER = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
DATA_ROOT = "./new20augments"
MODEL_NAME = "esm2"
DATASET_NAME = "dipsplus_raw"


for gen_file in pathlib.Path(DATA_ROOT).glob("*cm.joblib"):
    print(f"Processing {gen_file}")
    gen_data = joblib.load(gen_file)
    s1_encodings = {}
    s2_encodings = {}

    variable, ablate_type, trial_num, _ = gen_file.stem.split("-")
    # _, set_name, variable, ablation, _ = gen_file.stem.split("-") #use for larger_augments


    for i, x in enumerate(gen_data):
        key = f'new20_generated_{i}'
        seq1, seq2 = x['sequence'].split('|')

        s1_encodings[key] = TOKENIZER(" ".join(list(seq1)), return_tensors='pt', padding=True, truncation=True)
        s2_encodings[key] = TOKENIZER(" ".join(list(seq2)), return_tensors='pt', padding=True, truncation=True)
        
    out_name1 = f'./esm2_encodings/new20_encodings_woutholdout/{MODEL_NAME}_new20_{variable}_{ablate_type}_{trial_num}_s1_encodings.joblib'
    out_name2 = f'./esm2_encodings/new20_encodings_woutholdout/{MODEL_NAME}_new20_{variable}_{ablate_type}_{trial_num}_s2_encodings.joblib'

    # print(len(s1_encodings))
    
    joblib.dump(s1_encodings, out_name1)
    joblib.dump(s2_encodings, out_name2)