'''
I don't normally spit out the training MCC at the end of scripts, so this runs through all my models, loads them, and gets evaluation on the relevant training sets
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
from transformers import EsmTokenizer
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


def remove_keys(keys, keys_to_remove):
    return [key for key in keys if key not in keys_to_remove]

def load_model(model_path):
    model = ContactMapPredictor(encoder_module = ProtBertEncoder(device='gpu'), num_linear_layers = FFN_DEPTH, dropout_rate=FFN_DROPOUT, ffn_width=FFN_WIDTH, activation_fn = FFN_ACTIVATION)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

print("Loading data...")

##Load Data Encodings##
s1_encodings = joblib.load('./prot_bert_dipsplus_s1_encodings.joblib')
s2_encodings = joblib.load('./prot_bert_dipsplus_s2_encodings.joblib')
targets = joblib.load('../../data/dipsplus_raw/dipsplus_targets.joblib')

#Load keys
with open('../../data/dipsplus_raw/pairs-postprocessed-train-duplicates-removed.txt', 'r') as f:
    train_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings.keys()]
    print(f"Loaded {len(train_keys)} training keys")
    
with open('../../data/dipsplus_raw/pairs-postprocessed-val.txt', 'r') as f:
    val_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings.keys()]
    print(f"Loaded {len(val_keys)} validation keys")

#filter keys with seqs longer than 650
metrics_df = pd.read_csv('../../data/dipsplus_raw/complex_info.csv')
metrics_df['complex_path'] = metrics_df['complex_path'].apply(
    lambda x: '/'.join(x.split('/')[-2:])
    )

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


# the train and val keys will be reused for each model

rescounts = [20, 50, 80]
results = {}


'''
Below is the actual loop, It works by moving directly into the relevant folders for each model, and using the files from there to process.
'''
for res in rescounts:
    MAIN_RESULTS_DIR = f'/work/hdd/bcbv/jstump/protein_dataset_tests/main/scripts/protbert_runs/main_results/interfacearea{res}'
    os.chdir(MAIN_RESULTS_DIR)
    for method in ['targeted', 'uniform']:
        os.chdir(f'./{method}')
        for trial in range(5):
            print(f"Evaluating model: interfacearea {res}, method {method}, trial {trial}")
            os.chdir(f'./trial_{trial}')

            #remove keys
            with open('removed_train_keys.txt', 'r') as f:
                removed_train_keys = [line.strip() for line in f.readlines()]
                filtered_train_keys = remove_keys(train_keys, removed_train_keys)

            with open('removed_val_keys.txt', 'r') as f:
                removed_val_keys = [line.strip() for line in f.readlines()]
                filtered_val_keys = remove_keys(val_keys, removed_val_keys)

            #make those into datasets, loaders
            train_dataset = DipsDataset(filtered_train_keys, s1_encodings, s2_encodings, targets)
            val_dataset = DipsDataset(filtered_val_keys, s1_encodings, s2_encodings, targets)
            #load model
            model = load_model('./protbert_contact_map_predictor.pth')
            collator = MyDataCollator(model.encoder.tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)



            #get training MCC
            train_loss, train_mcc = evaluate(model, train_loader, MaskedBCEWithLogitsLoss(pos_weight=torch.tensor(BCE_POS_WEIGHT)).to(device), device)
            results[(res, method, trial)] = train_mcc

            print(f"Training MCC: {train_mcc}")
            os.chdir('../') #back to trial folder
        os.chdir('../') #back to method folder
    os.chdir('../') #back to interfacearea folder

#finally, do it all with the full model from optuna
os.chdir('/work/hdd/bcbv/jstump/protein_dataset_tests/main/scripts/protbert_runs/')

model = load_model('./protbert_full_dipsplus_model_holdouttest.pth')
train_dataset = DipsDataset(train_keys, s1_encodings, s2_encodings, targets)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MyDataCollator(EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")))
train_loss, train_mcc = evaluate(model, train_loader, MaskedBCEWithLogitsLoss(pos_weight=torch.tensor(BCE_POS_WEIGHT)).to(device), device)
results[('full', 'full', 'full')] = train_mcc



print("Results (training MCC):")
for key, value in results.items():
    print(f"{key}: {value}")

# Convert the dictionary to a list of records
data_records = []
for (res, method, trial), mcc in results.items():
    data_records.append({
        'Interfacearea': res,
        'Method': method,
        'Trial': trial,
        'Train_MCC': mcc
    })

# Create the DataFrame
df_results = pd.DataFrame(data_records)

# View the first few rows
print(df_results.head())  
df_results.to_csv('./training_mccs_interfacearea.csv', index = False)