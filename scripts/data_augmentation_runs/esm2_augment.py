#this script runs, for a single trial, the data augmentation. It loads the ablated set and the generated proteins and trains a model on the combined set. The model is evaluated on the augmented validation set and the holdout test set. The results are saved to a file for later analysis. This script is meant to be run multiple times, once for each trial, with the appropriate arguments.

import joblib
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #using second gpu to keep first (default) one free for others
import torch
from torch.utils.data import DataLoader
import sys
import argparse
from transformers import set_seed as hf_set_seed
from transformers import EsmTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src_code')))

from data_utils import filter_random_range, filter_interpercentile_range, stratify_sample, DipsDataset, MyDataCollator, filter_keys
from models import EsmEncoder, ContactMapPredictor
from train_test_routines import train_one_epoch, evaluate
from criterions import MaskedBCEWithLogitsLoss, MaskedMCC


parser = argparse.ArgumentParser(description="configuration for data augmentation")

#whether its the interfacearea trials or the rescount ones
parser.add_argument(
    '--filter_column', 
    type=str, 
    default='rescount', 
    help="The feature the data was ablated by. rescount or interfacearea"
)

#uniform or targeted
parser.add_argument(
    '--ablation_type', 
    type=str, 
    default='targeted', 
    help="The type of ablation. uniform or targeted"
)

#size of the percentage range
parser.add_argument(
    '--range_size', 
    type=int, 
    choices=[20, 50, 80],  # Restricts input to exactly these values
    default=20, 
    help="The size of the percentage range (20, 50, or 80)."
)

# Random seed
parser.add_argument(
    '--trial_number', 
    type=int, 
    default=-1, 
    help="Trial number for reproducibility; expect values 0, 1, 2, 3, or 4."
)

# #output path for results. will mimic the original ablation run result structure
# parser.add_argument(
#     '--output_path', 
#     type=str, 
#     default='/raid/emily/protein_dataset_project/results/esm2_augment/', 
#     help="Output path for results."
# )

args = parser.parse_args()

#using second gpu to keep first (default) one free for others
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

#set hyperparams found from HPO#
BATCH_SIZE = 4
LEARNING_RATE = 8.16e-05
NUM_EPOCHS = 15
FFN_DEPTH = 3
FFN_WIDTH = 128
FFN_DROPOUT = 0.42
FFN_ACTIVATION = 'gelu'
BCE_POS_WEIGHT = 2.82

#set misc paths
S1_ENCODINGS_PATH = '../esm2_runs/s1_encodings_dipsplus.joblib'
S2_ENCODINGS_PATH = '../esm2_runs/s2_encodings_dipsplus.joblib'
TARGETS_PATH = '../../data/dipsplus_raw/dipsplus_targets.joblib'
GENERATED_ROOT = './larger_augments'
GENERATED_PATH = f'{GENERATED_ROOT}/esmc-val-{args.filter_column}{args.range_size}-{args.ablation_type}-generated_cm.joblib'
GENERATED_S1_ENCODINGS_PATH = f'./esm2_encodings/esm2_val_{args.filter_column}{args.range_size}_{args.ablation_type}_s1_encodings.joblib'
GENERATED_S2_ENCODINGS_PATH = f'./esm2_encodings/esm2_val_{args.filter_column}{args.range_size}_{args.ablation_type}_s2_encodings.joblib'
REMOVED_KEYS_FOLDER = f'../../data/dipsplus_raw/removed_keys_by_trial/{args.filter_column}{args.range_size}/{args.ablation_type}/' #holds folders like trial_0, trial_1. Each has a removed_train_keys.txt and a removed_val_keys.txt. These will be used to recreate the ablation conditions
OUTDIR = f'./augmentation_results/esm2/{args.filter_column}{args.range_size}/{args.ablation_type}/trial_{args.trial_number}/'

os.makedirs(OUTDIR, exist_ok=True)

##Load Data Keys##
s1_encodings = joblib.load(S1_ENCODINGS_PATH)
s2_encodings = joblib.load(S2_ENCODINGS_PATH)
targets = joblib.load(TARGETS_PATH)



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


with open(f'{REMOVED_KEYS_FOLDER}/trial_{args.trial_number}/removed_train_keys.txt', 'r') as f:
    removed_train_keys = [line.strip() for line in f.readlines()]
with open(f'{REMOVED_KEYS_FOLDER}/trial_{args.trial_number}/removed_val_keys.txt', 'r') as f:
    removed_val_keys = [line.strip() for line in f.readlines()]

filtered_train_keys = [key for key in train_keys if key not in removed_train_keys]
filtered_val_keys = [key for key in val_keys if key not in removed_val_keys]

#grab the generated proteins
generated_set = joblib.load(GENERATED_PATH) #this will be a list of dicts, each dict holds the 'sequence' and 'contact_map'. The former is both sequences combined as SEQ1|SEQ2. The latter is a contact map in sparse format.

#Processing the generated files into a format that works with existing data_utils
generated_keys = [f'val_generated_{i}' for i in range(len(generated_set))] #these keys match what's in the s1 and s2 encoding files
generated_targets = {k: d['contact_map'] for k, d in zip(generated_keys, generated_set)}
generated_s1_encodings = joblib.load(GENERATED_S1_ENCODINGS_PATH)
generated_s2_encodings = joblib.load(GENERATED_S2_ENCODINGS_PATH)


#combine the generated keys with the filtered keys
augmented_train_keys = filtered_train_keys + generated_keys

#combine the generated encodings and targets with the original ones
augmented_s1_encodings = s1_encodings | generated_s1_encodings
augmented_s2_encodings = s2_encodings | generated_s2_encodings
augmented_targets = targets | generated_targets


train_dataset = DipsDataset(augmented_train_keys, augmented_s1_encodings, augmented_s2_encodings, augmented_targets)
val_dataset = DipsDataset(filtered_val_keys, augmented_s1_encodings, augmented_s2_encodings, augmented_targets)
test_dataset = DipsDataset(test_keys, augmented_s1_encodings, augmented_s2_encodings, augmented_targets)

print("Total Train Set Size: {}".format(len(train_dataset)))
print("Total Val Set Size: {}".format(len(val_dataset)))
print("Total Test Set Size: {}".format(len(test_dataset)))

##Create Model##
model = ContactMapPredictor(encoder_module = EsmEncoder(), num_linear_layers = FFN_DEPTH, dropout_rate=FFN_DROPOUT, ffn_width=FFN_WIDTH, activation_fn = FFN_ACTIVATION)
print(model)
model.to(device)


##Set Up Training##
loss_fn = MaskedBCEWithLogitsLoss(pos_weight=BCE_POS_WEIGHT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = MaskedBCEWithLogitsLoss(pos_weight=BCE_POS_WEIGHT).to(device)
collator = MyDataCollator(EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

##Train Model##
train_losses = []
val_losses = []
val_mccs = []
for epoch in range(NUM_EPOCHS):
    print("Starting epoch {}/{}".format(epoch+1, NUM_EPOCHS))
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_mcc = evaluate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_mccs.append(val_mcc)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} -- Train Loss: {train_loss:.4f} -- Val Loss: {val_loss:.4f} -- Val MCC: {val_mcc:.4f}")

test_loss, test_mcc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f} -- Test MCC: {test_mcc:.4f}")

train_loss, train_mcc = evaluate(model, train_loader, criterion, device)


## Record Results ##
torch.save(model.state_dict(), os.path.join(OUTDIR, 'esm2_contact_map_predictor.pth'))
metrics_df = pd.DataFrame({
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_mcc': val_mccs
})
metrics_df.to_csv(os.path.join(OUTDIR, 'training_metrics.csv'), index=False)

#write the removed keys to files
#this is arguably unecessary since they already exist in a separate folder but it's a nice sanity check
with open(os.path.join(OUTDIR, 'removed_train_keys.txt'), 'w') as f:
    for key in removed_train_keys:
        f.write(f"{key}\n")
with open(os.path.join(OUTDIR, 'removed_val_keys.txt'), 'w') as f:
    for key in removed_val_keys:
        f.write(f"{key}\n")

#save test results
with open(os.path.join(OUTDIR, 'test_results.txt'), 'w') as f:
    f.write(f"Test Loss at Final Epoch: {test_loss:.4f}\n")
    f.write(f"Test MCC at Final Epoch: {test_mcc:.4f}\n")

with open(os.path.join(OUTDIR, 'train_results.txt'), 'w') as f:
    f.write(f"Train Loss at Final Epoch: {train_loss:.4f}\n")
    f.write(f"Train MCC at Final Epoch: {train_mcc:.4f}\n")

