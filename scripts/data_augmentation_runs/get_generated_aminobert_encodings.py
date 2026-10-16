#I'm processing the data from Hayden to fit it into a format like the rest of the data_utils functions expect. This is a one-time script to generate the embeddings for the generated proteins. The embeddings will be saved in a format that can be used by the rest of the pipeline.

import joblib
import torch
from typing import List, Dict, Any
from tqdm import tqdm
import tokenization

import sys
import os
import pathlib

TOKENIZER = tokenization.FullTokenizer(k=1, token_to_replace_with_mask='X')
DATA_ROOT = "./larger_augments"
MODEL_NAME = "aminobert"
DATASET_NAME = "dipsplus_raw"


def generate_input_features_from_seq(seq, labels, tokenizer):
    '''
    function adapted from get_aminobert_embedding.py, written in make_joblibs/aminobert/aminobert_encodings.py

    '''
    tseq = [tokenization.CLS_TOKEN] + tokenizer.tokenize(seq)
    token_type_ids = [0] * len(tseq)
    input_mask = [1] * len(tseq)    
            
    input_ids = tokenizer.convert_tokens_to_ids(tseq)
    
    input_dict = {'input_ids': torch.tensor([input_ids], dtype=torch.int64),
                  'token_type_ids': torch.tensor([token_type_ids], dtype=torch.int64),
                  'attention_mask': torch.tensor([input_mask], dtype=torch.int64)}
            
    return input_dict

for gen_file in pathlib.Path(DATA_ROOT).glob("*cm.joblib"):
    print(f"Processing {gen_file}")
    gen_data = joblib.load(gen_file)
    s1_encodings = {}
    s2_encodings = {}

    _, set_name, variable, ablation, _ = gen_file.stem.split("-")


    for i, x in enumerate(gen_data):
        key = f'{set_name}_generated_{i}'
        seq1, seq2 = x['sequence'].split('|')

        s1_encodings[key] = generate_input_features_from_seq(seq1, None, TOKENIZER)
        s2_encodings[key] = generate_input_features_from_seq(seq2, None, TOKENIZER)
        
    out_name1 = f'./aminobert_encodings/{MODEL_NAME}_{set_name}_{variable}_{ablation}_s1_encodings.joblib'
    out_name2 = f'./aminobert_encodings/{MODEL_NAME}_{set_name}_{variable}_{ablation}_s2_encodings.joblib'

    print(len(s1_encodings), len(s2_encodings))
    
    joblib.dump(s1_encodings, out_name1)
    joblib.dump(s2_encodings, out_name2)
    