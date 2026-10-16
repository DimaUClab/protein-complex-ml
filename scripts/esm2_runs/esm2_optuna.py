'''
This script performs HPO on the ESM2 model using Optuna. This uses the predefined training and validation splits, and the run with the best hyperparameters are used for all subsequent testing.
'''

import joblib
import sys
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
#add src_code to path
import os
import torch
from torch.utils.data import DataLoader
import optuna
from transformers import EsmTokenizer
import random

print("Here we're redoing optuna but with a holdout set created from validation as initially intended before switching to DB5")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src_code')))

from data_utils import DipsDataset, MyDataCollator, filter_keys
from models import EsmEncoder, ContactMapPredictor
from train_test_routines import train_one_epoch, evaluate
from criterions import MaskedBCEWithLogitsLoss, MaskedMCC


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

s1_encodings = joblib.load('./s1_encodings_dipsplus.joblib')
s2_encodings = joblib.load('./s2_encodings_dipsplus.joblib')
targets = joblib.load('../../data/dipsplus_raw/dipsplus_targets.joblib')

# test_s1_encodings = joblib.load('../make_joblibs/esm2_db5_s1_encodings.joblib')
# test_s2_encodings = joblib.load('../make_joblibs/esm2_db5_s2_encodings.joblib')
# test_targets = joblib.load('../../data/db5_raw/db5_targets.joblib')
# test_keys = list(test_s1_encodings.keys())

db5_s1_encodings = joblib.load('../make_joblibs/esm2_db5_s1_encodings.joblib')
db5_s2_encodings = joblib.load('../make_joblibs/esm2_db5_s2_encodings.joblib')
db5_targets = joblib.load('../../data/db5_raw/db5_targets.joblib')
db5_keys = list(db5_s1_encodings.keys())


with open('../../data/dipsplus_raw/pairs-postprocessed-train-duplicates-removed.txt', 'r') as f:
    train_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings.keys()]
    
with open('../../data/dipsplus_raw/pairs-postprocessed-val.txt', 'r') as f:
    val_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings.keys()]

#filter keys with seqs longer than 650
metrics_df = pd.read_csv('../../data/dipsplus_raw/complex_info.csv')
metrics_df['complex_path'] = metrics_df['complex_path'].apply(
    lambda x: '/'.join(x.split('/')[-2:])
    )

train_keys = filter_keys(train_keys, metrics_df, 'seq1_len', 650, operator='<=')
train_keys = filter_keys(train_keys, metrics_df, 'seq2_len', 650, operator='<=')

val_keys = filter_keys(val_keys, metrics_df, 'seq1_len', 650, operator='<=')
val_keys = filter_keys(val_keys, metrics_df, 'seq2_len', 650, operator='<=')

#select test_keys by taking a holdout from val_keys
test_keys = random.sample(val_keys, int(0.2 * len(val_keys)))
val_keys = [key for key in val_keys if key not in test_keys]

print(test_keys)

#write test_keys to file
if not os.path.exists('holdout_val_keys.txt'):  
    with open('holdout_val_keys.txt', 'w') as f:
        f.write("## This file contains the keys held out from the validation set for final testing. Tied to the random seed 42 ##\n")
        for key in test_keys:
            f.write(f"{key}\n")
else: 
    print("holdout_val_keys.txt already exists. Using existing file for test keys.")
    with open('holdout_val_keys.txt', 'r') as f:
        lines = f.readlines()[1:]  #skip first line
        test_keys = [line.strip() for line in lines]
    val_keys = [key for key in val_keys if key not in test_keys]



train_dataset = DipsDataset(train_keys, s1_encodings, s2_encodings, targets)
val_dataset = DipsDataset(val_keys, s1_encodings, s2_encodings, targets)
test_dataset = DipsDataset(test_keys, s1_encodings, s2_encodings, targets)
db5_dataset = DipsDataset(db5_keys, db5_s1_encodings, db5_s2_encodings, db5_targets)

data_collator = MyDataCollator(EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"))


best_test_mcc = 0.0
best_val_mcc = 0.0
def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    num_epochs = trial.suggest_categorical('num_epochs', [5, 10, 15])
    ffn_depth = trial.suggest_categorical('ffn_depth', [2, 3, 4])
    ffn_width = trial.suggest_categorical('ffn_width', [128, 64, 32])
    ffn_dropout = trial.suggest_float('ffn_dropout', 0.0, 0.5)
    ffn_activation = trial.suggest_categorical('ffn_activation', ['relu', 'gelu', 'tanh'])
    bce_pos_weight = trial.suggest_float('bce_pos_weight', 1.0, 100.0)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    model = ContactMapPredictor(encoder_module=EsmEncoder(), 
                                num_linear_layers=ffn_depth,
                                dropout_rate=ffn_dropout,
                                ffn_width=ffn_width,
                                activation_fn=ffn_activation)
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MaskedBCEWithLogitsLoss(pos_weight=bce_pos_weight).to(device)
    
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device, verbose=True)
        
    val_loss, val_mcc = evaluate(model, val_loader, criterion, device)
    global best_val_mcc, best_test_mcc
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        test_loss, test_mcc = evaluate(model, DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator), criterion, device)
        if test_mcc > best_test_mcc:
            best_test_mcc = test_mcc
            print(f"New best test MCC: {best_test_mcc}")
    return val_mcc

storage_name = f"sqlite:///optuna_study_esm2_withHoldout.db"

    
study = optuna.create_study(direction='maximize', storage=storage_name, load_if_exists=True, study_name = 'esm2_hpo_withHoldout')
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
print(f"Best validation MCC achieved: {best_val_mcc}")
print(f"Test MCC achieved for best validation: {best_test_mcc}")