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
from transformers import EsmTokenizer
import random 
import tokenization

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src_code')))

from data_utils import filter_random_range, filter_interpercentile_range, stratify_sample, DipsDataset, MyDataCollator, filter_keys
from models import AminoBertEncoder, ContactMapPredictor
from train_test_routines import train_one_epoch, evaluate
from criterions import MaskedBCEWithLogitsLoss, MaskedMCC


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#set hyperparams found from HPO#
BATCH_SIZE = 4
LEARNING_RATE = 2.02e-5
NUM_EPOCHS = 15
FFN_DEPTH = 4
FFN_WIDTH = 64
FFN_DROPOUT = 0.03
FFN_ACTIVATION = 'relu'
BCE_POS_WEIGHT = 7.25

##Handle Arguments##

parser = argparse.ArgumentParser(description="Configuration for complex info filtering and model training.")

# 1. The column in complex_info.csv to use for filtering
parser.add_argument(
    '--filter_column', 
    type=str, 
    default='residue count', 
    help="The column in complex_info.csv to use for filtering (e.g. 'residue count')."
)

# 2. The size of the percentage range (20, 50, 80)
parser.add_argument(
    '--range_size', 
    type=int, 
    choices=[20, 50, 80],  # Restricts input to exactly these values
    default=20, 
    help="The size of the percentage range (20, 50, or 80)."
)

# 3. Whether the column is actually being used for filtering
#    If --use_filtering is present, this becomes True. If omitted, it is False (uniform sampling).
parser.add_argument(
    '--use_filtering', 
    action='store_true', 
    help="Flag to enable filtering based on the column. If omitted, samples uniformly."
)

# 4. The path to store outputs
parser.add_argument(
    '--output_path', 
    type=str, 
    required=True, 
    help="Directory path to store outputs (models, logs, etc)."
)

# 5. Random seed
parser.add_argument(
    '--seed', 
    type=int, 
    default=42, 
    help="Random seed for reproducibility."
)

args = parser.parse_args()

#set random seeds
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
hf_set_seed(args.seed)


##Load Data Keys##
s1_encodings = joblib.load('./aminobert_dipsplus_raw_s1_encodings.joblib')
s2_encodings = joblib.load('./aminobert_dipsplus_raw_s2_encodings.joblib')
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


#perform random filtering on train and val; NOT test set
if args.use_filtering:
    # filtered_train_keys, start_percentile, end_percentile = filter_random_range(train_keys, metrics_df, args.filter_column, percentile_size = args.range_size)
    # filtered_val_keys = filter_interpercentile_range(val_keys, metrics_df, args.filter_column, start_percentile, end_percentile)

    #02/10/26: Random seed isn't doing what I want so I've saved the removed keys from the ESM runs and we will just reuse them here.
    with open(f'/work/hdd/bcbv/jstump/protein_dataset_tests/main/data/dipsplus_raw/removed_keys_by_trial/interfacearea{args.range_size}/targeted/trial_{args.seed}/removed_train_keys.txt', 'r') as f:
        removed_train_keys = [line.strip() for line in f.readlines()]
    with open(f'/work/hdd/bcbv/jstump/protein_dataset_tests/main/data/dipsplus_raw/removed_keys_by_trial/interfacearea{args.range_size}/targeted/trial_{args.seed}/removed_val_keys.txt', 'r') as f:
        removed_val_keys = [line.strip() for line in f.readlines()]
    filtered_train_keys = [key for key in train_keys if key not in removed_train_keys]
    filtered_val_keys = [key for key in val_keys if key not in removed_val_keys]

else:
    # filtered_train_keys = stratify_sample(train_keys, metrics_df, args.filter_column, args.range_size/100, num_bins = 10)
    # filtered_val_keys = stratify_sample(val_keys, metrics_df, args.filter_column, args.range_size/100, num_bins = 10)
    with open(f'/work/hdd/bcbv/jstump/protein_dataset_tests/main/data/dipsplus_raw/removed_keys_by_trial/interfacearea{args.range_size}/uniform/trial_{args.seed}/removed_train_keys.txt', 'r') as f:
        removed_train_keys = [line.strip() for line in f.readlines()]
    with open(f'/work/hdd/bcbv/jstump/protein_dataset_tests/main/data/dipsplus_raw/removed_keys_by_trial/interfacearea{args.range_size}/uniform/trial_{args.seed}/removed_val_keys.txt', 'r') as f:
        removed_val_keys = [line.strip() for line in f.readlines()]
    filtered_train_keys = [key for key in train_keys if key not in removed_train_keys]
    filtered_val_keys = [key for key in val_keys if key not in removed_val_keys]


#record what was removed
removed_train_keys = list(set(train_keys) - set(filtered_train_keys))
removed_val_keys = list(set(val_keys) - set(filtered_val_keys))

#make datasets (save loaders for after model creation so that tokenizer has been defined)
train_dataset = DipsDataset(filtered_train_keys, s1_encodings, s2_encodings, targets)
val_dataset = DipsDataset(filtered_val_keys, s1_encodings, s2_encodings, targets)
test_dataset = DipsDataset(test_keys, s1_encodings, s2_encodings, targets)


##Create Model##
model = ContactMapPredictor(encoder_module = AminoBertEncoder('./aminobert_model_files'), num_linear_layers = FFN_DEPTH, dropout_rate=FFN_DROPOUT, ffn_width=FFN_WIDTH, activation_fn = FFN_ACTIVATION)
print(model)
model.to(device)


##Set Up Training##
loss_fn = MaskedBCEWithLogitsLoss(pos_weight=BCE_POS_WEIGHT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = MaskedBCEWithLogitsLoss(pos_weight=BCE_POS_WEIGHT).to(device)
collator = MyDataCollator(tokenization.FullTokenizer(k=1, token_to_replace_with_mask='X'))

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

##Test Model On DB5##
test_loss, test_mcc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f} -- Test MCC: {test_mcc:.4f}")

##Record Results##
#make outdir
os.makedirs(args.output_path, exist_ok=True)

#save model
torch.save(model.state_dict(), os.path.join(args.output_path, 'aminobert_contact_map_predictor.pth'))
#save metrics
metrics_df = pd.DataFrame({
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_mcc': val_mccs
})
metrics_df.to_csv(os.path.join(args.output_path, 'training_metrics.csv'), index=False)

#write the removed keys to files
with open(os.path.join(args.output_path, 'removed_train_keys.txt'), 'w') as f:
    for key in removed_train_keys:
        f.write(f"{key}\n")
with open(os.path.join(args.output_path, 'removed_val_keys.txt'), 'w') as f:
    for key in removed_val_keys:
        f.write(f"{key}\n")

#save test results
with open(os.path.join(args.output_path, 'test_results.txt'), 'w') as f:
    f.write(f"Test Loss at Final Epoch: {test_loss:.4f}\n")
    f.write(f"Test MCC at Final Epoch: {test_mcc:.4f}\n")
