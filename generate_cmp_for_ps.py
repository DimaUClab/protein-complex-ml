import numpy as np
import pandas as pd
from models import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import sys
import os
import argparse
from criterions import *
import tokenization
from data_utils import *
import joblib

model_dir = os.environ["PROJ_CM_MODELS"]
data_dir = os.environ["PROJ_DATASETS"]

BATCH_SIZE=4
DEVICE='cuda'
THREADS = 28
MAX_SEQ_LEN = 650

def evaluate(model, dataloader, criterion, device):
    '''
    Evaluates a model. Returns average loss and MCC.
    '''
    model.eval()
    total_loss = 0

    # Initialize and move metric to device
    val_mcc = MaskedMCC().to(device)
    val_mcc.reset() # Good practice to ensure clean state

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input1_ids=batch['input1_ids'],
                input1_attention_mask=batch['input1_attention_mask'],
                input2_ids=batch['input2_ids'],
                input2_attention_mask=batch['input2_attention_mask']
            )
            
            # Loss calculation
            loss = criterion(outputs, batch['labels'], batch['mask'])
            total_loss += loss.item()
            
            # Metric update: Use sigmoid because BCEWithLogitsLoss inputs raw logits
            val_mcc.update(torch.sigmoid(outputs), batch['labels'], batch['mask'])
    
    # Compute final metric and move back to CPU for logging/Optuna
    final_mcc = val_mcc.compute().item()
    avg_loss = total_loss / len(dataloader)
    
    print(f"Validation - Loss: {avg_loss:.4f}, MCC: {final_mcc:.4f}")
    return avg_loss, final_mcc

def predict(model, dataloader):
    '''
    Evaluates a model. Returns predicted contact map.
    '''
    model.eval()
    total_loss = 0
    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            probs = torch.sigmoid(
                        model(
                            input1_ids=batch['input1_ids'],
                            input1_attention_mask=batch['input1_attention_mask'],
                            input2_ids=batch['input2_ids'],
                            input2_attention_mask=batch['input2_attention_mask']
                        )
                    ).cpu().numpy()
            
            #outputs.append(probs)
            probs_pad_bottom = 650 - probs.shape[1]
            probs_pad_right = 650 - probs.shape[2]
            
            outputs.append(np.pad(probs, ((0, 0), (0, probs_pad_bottom), (0, probs_pad_right))))
    
    return outputs


def load_aminobert_model(fname):
    # BATCH_SIZE = 4
    # LEARNING_RATE = 2.02e-5
    # NUM_EPOCHS = 15
    # FFN_DEPTH = 4
    # FFN_WIDTH = 64
    # FFN_DROPOUT = 0.03
    # FFN_ACTIVATION = 'relu'
    # BCE_POS_WEIGHT = 7.25
    aminobert_cmp = ContactMapPredictor(AminoBertEncoder(os.path.join(model_dir, 'aminobert_things', 'aminobert_model_files')), ffn_width=64, num_linear_layers=4, dropout_rate=0.03, activation_fn='relu')
    aminobert_cmp_weights = torch.load(fname)
    aminobert_cmp.load_state_dict(aminobert_cmp_weights)
    return aminobert_cmp


def load_esm2_model(fname):
    # BATCH_SIZE = 4
    # LEARNING_RATE = 8.16e-05
    # NUM_EPOCHS = 15
    # FFN_DEPTH = 3
    # FFN_WIDTH = 128
    # FFN_DROPOUT = 0.42
    # FFN_ACTIVATION = 'gelu'
    # BCE_POS_WEIGHT = 2.82
    esm2_cmp = ContactMapPredictor(EsmEncoder(), ffn_width=128, num_linear_layers=3, dropout_rate=0.42, activation_fn='gelu')
    esm2_cmp_weights = torch.load(fname, weights_only=True)
    del esm2_cmp_weights["encoder.encoder.embeddings.position_embeddings.weight"]
    esm2_cmp.load_state_dict(esm2_cmp_weights)
    return esm2_cmp

def load_protbert_model(fname):
    # BATCH_SIZE = 4
    # LEARNING_RATE = 1.16e-6
    # NUM_EPOCHS = 15
    # FFN_DEPTH = 2
    # FFN_WIDTH = 128
    # FFN_DROPOUT = 0.0049
    # FFN_ACTIVATION = 'tanh'
    # BCE_POS_WEIGHT = 98.93
    protbert_cmp = ContactMapPredictor(ProtBertEncoder(), ffn_width=128, num_linear_layers=2, activation_fn='tanh', dropout_rate=0.0049)
    protbert_cmp_weights = torch.load(fname)
    protbert_cmp.load_state_dict(protbert_cmp_weights)
    return protbert_cmp


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', '-p', type=str, required=True)
    parser.add_argument('--index', '-i', type=int, required=True)
    #parser.add_argument('--trial', '-t', type=int, required=True)
    #parser.add_argument('--model', '-m', type=str, required=True)

    args = parser.parse_args()
    print(args)
    
    # Set CUDA device
    if DEVICE == "cpu":
        device = torch.device("cpu")
    elif DEVICE == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            print("cuda not available")
            device = torch.device("cpu")
    else:
        print("using default device")
        device = torch.device("cpu")
    
    # print computation device we got
    torch.set_num_threads(THREADS)
    print(f"Using device: {device}")

    with open(os.path.join(model_dir, 'holdout_val_keys.txt'), 'r') as f:
        lines = f.readlines()[1:]  #skip first line its just a comment
        holdout_keys = [line.strip() for line in lines]
    
    

    #with open(args.parameters, 'r') as f:
    #    lines = f.readlines()
    #    parameters = [line.strip().split(' ') for line in lines]
    
    #parameter_set = parameters[args.index]
    parameter_sets = pd.read_csv(args.parameters, index_col=0, header=0, sep='\t')
    parameters = parameter_sets.loc[args.index, 'parameters'].split(' ')
    args.model = parameters[0]
    args.reduction = f"{parameters[1]}/{parameters[2]}"
    args.trial = parameters[3]
    targets = joblib.load(os.path.join(model_dir, "dipsplus_targets.joblib"))
    
    s1_encodings_esm2 = joblib.load(os.path.join(model_dir, 'esm_s1_encodings_dipsplus.joblib'))
    s2_encodings_esm2 = joblib.load(os.path.join(model_dir, 'esm_s2_encodings_dipsplus.joblib'))
    if args.model == 'aminobert':
        s1_encodings_aminobert = joblib.load(os.path.join(model_dir, 'aminobert_dipsplus_raw_s1_encodings.joblib'))
        s2_encodings_aminobert = joblib.load(os.path.join(model_dir, 'aminobert_dipsplus_raw_s2_encodings.joblib'))
        holdout_dataset_aminobert = DipsDataset(holdout_keys, s1_encodings_aminobert, s2_encodings_aminobert, targets)
        #criterion = MaskedBCEWithLogitsLoss(pos_weight=7.25).to(device)
        collator_aminobert = MyDataCollator(tokenization.FullTokenizer(k=1, token_to_replace_with_mask='X'))
        holdout_loader_aminobert = DataLoader(holdout_dataset_aminobert, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_aminobert)
    elif args.model == 'esm2':
        holdout_dataset_esm2 = DipsDataset(holdout_keys, s1_encodings_esm2, s2_encodings_esm2, targets)
        #criterion_esm2 = MaskedBCEWithLogitsLoss(pos_weight=2.82).to(device)
        collator_esm2 = MyDataCollator(EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"))
        holdout_loader_esm2 = DataLoader(holdout_dataset_esm2, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_esm2)
    elif args.model == 'protbert':
        s1_encodings_protbert = joblib.load(f'{model_dir}/prot_bert_dipsplus_s1_encodings.joblib')
        s2_encodings_protbert = joblib.load(f'{model_dir}/prot_bert_dipsplus_s2_encodings.joblib')
        holdout_dataset_protbert = DipsDataset(holdout_keys, s1_encodings_protbert, s2_encodings_protbert, targets)
        protbert_dummy = ProtBertEncoder(device='gpu')
        #criterion_protbert = MaskedBCEWithLogitsLoss(pos_weight=98.93).to(device)
        collator_protbert = MyDataCollator(protbert_dummy.tokenizer)
        holdout_loader_protbert = DataLoader(holdout_dataset_protbert, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_protbert)
    
    #dips = pd.read_csv('dips_plus-one_hot.csv', header=0, index_col=0, sep=' ')
    dips_meta = pd.read_csv(os.path.join(data_dir, "dips_meta.dat"), header=0, index_col=0, sep=' ')

    # key lists for indexing into dips dataframe
    with open(f'{model_dir}/holdout_val_keys.txt', 'r') as f:
        lines = f.readlines()[1:]
        holdout_keys = [line.strip() for line in lines]

    with open(os.path.join(model_dir, "reductions_aminobert", args.reduction, args.trial, "removed_train_keys.txt"), 'r') as f:
        removed_train_keys = [line.strip() for line in f.readlines()]

    with open(os.path.join(model_dir, "reductions_aminobert", args.reduction, args.trial, "removed_val_keys.txt"), 'r') as f:
        removed_val_keys = [line.strip() for line in f.readlines()]

    with open(os.path.join(model_dir, "pairs-postprocessed-train-duplicates-removed.txt"), 'r') as f:
        all_train_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings_esm2.keys()]

    with open(os.path.join(model_dir, "pairs-postprocessed-val.txt"), 'r') as f:
        all_val_keys = [line.strip() for line in f.readlines() if line.strip() in s1_encodings_esm2.keys()]
    
    train_keys = []
    for key in all_train_keys:
        if key in removed_train_keys:
            continue
        if dips_meta.loc[key, 'l_len'] > MAX_SEQ_LEN or dips_meta.loc[key, 'r_len'] > MAX_SEQ_LEN:
            continue
        train_keys.append(key)

    val_keys = []
    for key in all_val_keys:
        if key in removed_val_keys:
            continue
        if key in holdout_keys:
            continue
        if dips_meta.loc[key, 'l_len'] > MAX_SEQ_LEN or dips_meta.loc[key, 'r_len'] > MAX_SEQ_LEN:
            continue
        val_keys.append(key)

    predictions = {}
    if args.model == 'aminobert':
        fname_aminobert = os.path.join(model_dir, "reductions_aminobert", args.reduction, args.trial, "aminobert_contact_map_predictor.pth")
        model_aminobert = load_aminobert_model(fname_aminobert).to(device)
        predictions['hold-out'] = predict(model_aminobert, holdout_loader_aminobert)
        predictions['hold-out'] = np.concatenate(predictions['hold-out'], axis=0)
        del holdout_loader_aminobert
        del holdout_dataset_aminobert
        torch.cuda.empty_cache()
        
        val_dataset_aminobert = DipsDataset(val_keys, s1_encodings_aminobert, s2_encodings_aminobert, targets)
        val_loader_aminobert = DataLoader(val_dataset_aminobert, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_aminobert)
        predictions['val'] = predict(model_aminobert, val_loader_aminobert)
        predictions['val'] = np.concatenate(predictions['val'], axis=0)
        del val_loader_aminobert
        del val_dataset_aminobert
        torch.cuda.empty_cache()

        
        train_dataset_aminobert = DipsDataset(train_keys, s1_encodings_aminobert, s2_encodings_aminobert, targets)
        train_loader_aminobert = DataLoader(train_dataset_aminobert, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_aminobert)
        predictions['train'] = predict(model_aminobert, train_loader_aminobert)
        predictions['train'] = np.concatenate(predictions['train'], axis=0)
        del train_loader_aminobert
        del train_dataset_aminobert
        torch.cuda.empty_cache()
        
        
        predictions['keys'] = {'hold-out':holdout_keys, 'val':val_keys, 'train':train_keys}
    elif args.model == 'esm2':
        fname_esm2 = os.path.join(model_dir, "reductions_esm2", args.reduction, args.trial, "esm2_contact_map_predictor.pth")
        model_esm2 = load_esm2_model(fname_esm2).to(device)
        predictions['hold-out'] = predict(model_esm2, holdout_loader_esm2)
        predictions['hold-out'] = np.concatenate(predictions['hold-out'], axis=0)
        del holdout_loader_esm2
        del holdout_dataset_esm2
        torch.cuda.empty_cache()
        
        val_dataset_esm2 = DipsDataset(val_keys, s1_encodings_esm2, s2_encodings_esm2, targets)
        val_loader_esm2 = DataLoader(val_dataset_esm2, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_esm2)
        predictions['val'] = predict(model_esm2, val_loader_esm2)
        predictions['val'] = np.concatenate(predictions['val'], axis=0)
        del val_loader_esm2
        del val_dataset_esm2
        torch.cuda.empty_cache()
        
        train_dataset_esm2 = DipsDataset(train_keys, s1_encodings_esm2, s2_encodings_esm2, targets)
        train_loader_esm2 = DataLoader(train_dataset_esm2, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_esm2)
        predictions['train'] = predict(model_esm2, train_loader_esm2)
        predictions['train'] = np.concatenate(predictions['train'], axis=0)
        del train_loader_esm2
        del train_dataset_esm2
        torch.cuda.empty_cache()

        
        predictions['keys'] = {'hold-out':holdout_keys, 'val':val_keys, 'train':train_keys}
    elif args.model == 'protbert':
        fname_protbert = os.path.join(model_dir, "reductions_protbert", args.reduction, args.trial, "protbert_contact_map_predictor.pth")
        model_protbert = load_protbert_model(fname_protbert).to(device)
        predictions['hold-out'] = predict(model_protbert, holdout_loader_protbert)
        predictions['hold-out'] = np.concatenate(predictions['hold-out'], axis=0)
        del holdout_loader_protbert
        del holdout_dataset_protbert
        torch.cuda.empty_cache()
        
        val_dataset_protbert = DipsDataset(val_keys, s1_encodings_protbert, s2_encodings_protbert, targets)
        val_loader_protbert = DataLoader(val_dataset_protbert, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_protbert)
        predictions['val'] = predict(model_protbert, val_loader_protbert)
        predictions['val'] = np.concatenate(predictions['val'], axis=0)
        del val_loader_protbert
        del val_dataset_protbert
        torch.cuda.empty_cache()
        
        
        train_dataset_protbert = DipsDataset(train_keys, s1_encodings_protbert, s2_encodings_protbert, targets)
        train_loader_protbert = DataLoader(train_dataset_protbert, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator_protbert)
        predictions['train'] = predict(model_protbert, train_loader_protbert)
        predictions['train'] = np.concatenate(predictions['train'], axis=0)
        del train_loader_protbert
        del train_dataset_protbert
        torch.cuda.empty_cache()
        
        predictions['keys'] = {'hold-out':holdout_keys, 'val':val_keys, 'train':train_keys}
    inner_fname = '-'.join(args.reduction.split('/'))
    fname = os.path.join(data_dir, "intermediates", "contact_map_predictions", f"{args.model}-{inner_fname}-{args.trial}-predictions-ps{args.index}.npz")
    with open(os.path.join(data_dir, "intermediates", "contact_map_predictions", f"{args.model}-{inner_fname}-{args.trial}-train_keys-ps{args.index}.txt"), "w") as f:
        for line in train_keys:
            f.write(f"{line}\n")
    with open(os.path.join(data_dir, "intermediates", "contact_map_predictions", f"{args.model}-{inner_fname}-{args.trial}-val_keys-ps{args.index}.txt"), "w") as f:
        for line in val_keys:
            f.write(f"{line}\n")
    np.savez_compressed(fname, holdout=predictions['hold-out'], val=predictions['val'], train=predictions['train'])
    #joblib.dump(predictions, fname, compress=('zlib', 3))
    #print(predictions['train'])
    
