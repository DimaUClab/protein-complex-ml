'''
DEPRECATED SCRIPT!!! Based on when db5 was the test set.

Runs training on complete set using optimized hyperparameters found via optuna, then evaluates on DB5.
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
from models import EsmEncoder, ContactMapPredictor
from train_test_routines import train_one_epoch, evaluate
from criterions import MaskedBCEWithLogitsLoss, MaskedMCC

def get_masked_confusion_matrix(model, loader, device, threshold=0.5):
    """
    Computes the confusion matrix over a dataset, respecting the padding mask.
    Returns:
        cm (np.array): [[TN, FP], 
                        [FN, TP]]
    """
    model.eval()
    
    # Initialize counters (using floats to prevent overflow on large datasets)
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in loader:
            # 1. Unpack Batch (matching your collator's keys)
            # Exclude 'labels' and 'mask' from inputs passed to model
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ['labels', 'mask']}
            targets = batch['labels'].to(device)
            mask = batch['mask'].to(device) # Your collator provides this
            
            # 2. Forward Pass
            logits = model(**inputs)
            preds = (torch.sigmoid(logits) > threshold).float()
            
            # 3. Apply Mask & Accumulate (Exact logic from your MaskedMCC)
            # We multiply by mask so that padded positions (mask=0) become 0 and don't count.
            TP += (preds * targets * mask).sum().item()
            TN += ((1 - preds) * (1 - targets) * mask).sum().item()
            FP += (preds * (1 - targets) * mask).sum().item()
            FN += ((1 - preds) * targets * mask).sum().item()
            
    # 4. Construct Matrix
    # Format: [[True Neg, False Pos], [False Neg, True Pos]]
    cm = np.array([[TN, FP], [FN, TP]])
    
    return cm

def print_cm_diagnostics(cm):
    """Helper to print the matrix nicely with derived metrics"""
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    
    print(f"Confusion Matrix:")
    print(f"      Pred 0    Pred 1")
    print(f"Act 0 {tn/total:<9} {fp/total:<9}")
    print(f"Act 1 {fn/total:<9} {tp/total:<9}")
    
    # calc simple metrics
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) + 1e-8)
    
    print(f"Stats -> Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | MCC: {mcc:.4f}")
    
    # The "Dummy Check"
    # If the model just guessed 0 for everything, what would accuracy be?
    print(f"Baseline (All 0s) Acc: {tn/total:.4f}")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#set hyperparams found from HPO#
BATCH_SIZE = 8
LEARNING_RATE = 7.13e-05
NUM_EPOCHS = 15
FFN_DEPTH = 2
FFN_WIDTH = 128
FFN_DROPOUT = 0.407
FFN_ACTIVATION = 'relu'
BCE_POS_WEIGHT = 5.48



##Load Data Keys##
s1_encodings = joblib.load('./s1_encodings_dipsplus.joblib')
s2_encodings = joblib.load('./s2_encodings_dipsplus.joblib')
targets = joblib.load('../../data/dipsplus_raw/dipsplus_targets.joblib')

test_s1_encodings = joblib.load('../make_joblibs/esm2_db5_s1_encodings.joblib')
test_s2_encodings = joblib.load('../make_joblibs/esm2_db5_s2_encodings.joblib')
test_targets = joblib.load('../../data/db5_raw/db5_targets.joblib')
test_keys = list(test_s1_encodings.keys())


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
#####SKIPS THE FILTERING DONE IN MAIN.PY#####


train_dataset = DipsDataset(train_keys, s1_encodings, s2_encodings, targets)
val_dataset = DipsDataset(val_keys, s1_encodings, s2_encodings, targets)
test_dataset = DipsDataset(test_keys, test_s1_encodings, test_s2_encodings, test_targets)

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
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_mcc = evaluate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_mccs.append(val_mcc)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} -- Train Loss: {train_loss:.4f} -- Val Loss: {val_loss:.4f} -- Val MCC: {val_mcc:.4f}")

##Test Model On DB5##
test_loss, test_mcc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f} -- Test MCC: {test_mcc:.4f}")


print("Training Set CM Results")
cm = get_masked_confusion_matrix(model, train_loader, device, threshold=0.5)
print_cm_diagnostics(cm)
print("Validation Set CM Results")
cm = get_masked_confusion_matrix(model, val_loader, device, threshold=0.5)
print_cm_diagnostics(cm)
print("Test Set (DB5) CM Results")
cm = get_masked_confusion_matrix(model, test_loader, device, threshold=0.5)
print_cm_diagnostics(cm)


#Save Model#
torch.save(model.state_dict(), './esm2_full_dipsplus_model.pth')
