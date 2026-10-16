'''
Protbert's Optuna never finished running due to a glitch, so there actually is no full dataset model. This script just rectifies that.
'''

'''
this script performs a single run of data filtering and training, validation, testing using the ESM2 model. The hyperparameters are taken from what was found running esm2_optuna.py. This same script will be ran for each trial, with appropriate arguments.

REQUIRES
 - the pairs-postprocessed.txt files for train, val, and test
 - the complex_info.csv file, used for filtering down the dataset


ARGUMENTS
 - the column in complex_info.csv to use for filtering (e.g. 'residue count')
 - the size of the percentage range (20, 50, 80)
 - whether the column is actually being used for filtering (True/False) or if its just going to sample uniformly
 - the path to store outputs (models, logs, etc)
 - Random seed. This has to be done so additional models can be compared on matching grounds.
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
from train_test_routines import train_one_epoch, evaluate
from criterions import MaskedBCEWithLogitsLoss, MaskedMCC


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#set hyperparams found from HPO#
BATCH_SIZE = 4
LEARNING_RATE = 1.16e-6
NUM_EPOCHS = 15
FFN_DEPTH = 2
FFN_WIDTH = 128
FFN_DROPOUT = 0.0049
FFN_ACTIVATION = 'tanh'
BCE_POS_WEIGHT = 98.93

##Handle Arguments##


##Load Data Keys##
s1_encodings = joblib.load('./prot_bert_dipsplus_s1_encodings.joblib')
s2_encodings = joblib.load('./prot_bert_dipsplus_s2_encodings.joblib')
targets = joblib.load('../../data/dipsplus_raw/dipsplus_targets.joblib')


with open('../../data/dipsplus_raw/pairs-postprocessed-train-duplicates-removed.txt', 'r') as f:
    train_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings.keys()]
    
with open('../../data/dipsplus_raw/pairs-postprocessed-val.txt', 'r') as f:
    val_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings.keys()]
    


##Filter Data (ONLY train and val sets)##
metrics_df = pd.read_csv('../../data/dipsplus_raw/complex_info.csv')
#fix the paths in complex_info to just be XX/XXXX.dill to match the keys
metrics_df['complex_path'] = metrics_df['complex_path'].apply(
    lambda x: '/'.join(x.split('/')[-2:])
    )

#filter sequences longer than 650
train_keys = filter_keys(train_keys, metrics_df, 'seq1_len', 650, operator='<=')
train_keys = filter_keys(train_keys, metrics_df, 'seq2_len', 650, operator='<=')

val_keys = filter_keys(val_keys, metrics_df, 'seq1_len', 650, operator='<=')
val_keys = filter_keys(val_keys, metrics_df, 'seq2_len', 650, operator='<=')


#create holdout test keys
print("holdout_val_keys.txt already exists. Using existing file for test keys.")
with open('../../data/dipsplus_raw/holdout_val_keys.txt', 'r') as f:
    lines = f.readlines()[1:]  #skip first line its just a comment
    test_keys = [line.strip() for line in lines]
val_keys = [key for key in val_keys if key not in test_keys]




#make datasets (save loaders for after model creation so that tokenizer has been defined)
train_dataset = DipsDataset(train_keys, s1_encodings, s2_encodings, targets)
val_dataset = DipsDataset(val_keys, s1_encodings, s2_encodings, targets)
test_dataset = DipsDataset(test_keys, s1_encodings, s2_encodings, targets)


##Create Model##
model = ContactMapPredictor(encoder_module = ProtBertEncoder(device='gpu'), num_linear_layers = FFN_DEPTH, dropout_rate=FFN_DROPOUT, ffn_width=FFN_WIDTH, activation_fn = FFN_ACTIVATION)
print(model)
model.to(device)


##Set Up Training##
loss_fn = MaskedBCEWithLogitsLoss(pos_weight=BCE_POS_WEIGHT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = MaskedBCEWithLogitsLoss(pos_weight=BCE_POS_WEIGHT).to(device)
collator = MyDataCollator(model.encoder.tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

##Train Model##
train_losses = []
val_losses = []
val_mccs = []
for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_mcc = evaluate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_mccs.append(val_mcc)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} -- Train Loss: {train_loss:.4f} -- Val Loss: {val_loss:.4f} -- Val MCC: {val_mcc:.4f}")

##Test Model On Test Set##
test_loss, test_mcc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f} -- Test MCC: {test_mcc:.4f}")

##Record Results##
#make outdir
torch.save(model.state_dict(), './protbert_full_dipsplus_model_holdouttest.pth')

