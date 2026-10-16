'''
I keep having to check and recheck my paths and the like for loading models, data, etc for all three of my encoder types, so let's just make some basic functions to do that for me so scripts can be easier and less error-prone to write
'''

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import sys
import argparse
from transformers import set_seed as hf_set_seed
from transformers import BertTokenizer
import random 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src_code')))

from data_utils import filter_random_range, filter_interpercentile_range, stratify_sample, DipsDataset, MyDataCollator, filter_keys
from models import ProtBertEncoder, ContactMapPredictor



def load_protbert_model(FFN_DEPTH, FFN_DROPOUT, FFN_WIDTH, FFN_ACTIVATION, device,  weights_path=None):
    model = ContactMapPredictor(encoder_module = ProtBertEncoder(device=device), num_linear_layers = FFN_DEPTH, dropout_rate=FFN_DROPOUT, ffn_width=FFN_WIDTH, activation_fn = FFN_ACTIVATION)
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model.to(device)
    model.eval()
    return model



def process_removed_keys(remove_keys_path):
    '''
    does the process of removing keys based off a prexisting file. At this point i've done all the actual random filtering so any future runs just use the same removed keys.
    '''
    pass

def build_datasets():
    '''
    builds the datasets and dataloaders for training and testing, given keylists and encodings
    '''
    pass