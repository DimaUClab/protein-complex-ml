'''
The esm2_main.py script inadverdantly passed the validation set when it came time to evaluate on DB5. This script loads all the models trained on the different reduced sets and calculates their performance on DB5.
'''


from json import encoder
import joblib
import sys
import os
import torch 
from transformers import EsmTokenizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src_code')))

from data_utils import DipsDataset, MyDataCollator
from models import EsmEncoder, ContactMapPredictor
from train_test_routines import evaluate
from criterions import MaskedBCEWithLogitsLoss

#necessary hyperparams
BCE_POS_WEIGHT = 5.48
BATCH_SIZE = 8
FFN_DEPTH = 2
FFN_WIDTH = 128
FFN_DROPOUT = 0.407
FFN_ACTIVATION = 'relu'

#load data
test_s1_encodings = joblib.load('../make_joblibs/esm2_db5_s1_encodings.joblib')
test_s2_encodings = joblib.load('../make_joblibs/esm2_db5_s2_encodings.joblib')
test_targets = joblib.load('../../data/db5_raw/db5_targets.joblib')
test_keys = list(test_s1_encodings.keys())

test_dataset = DipsDataset(test_keys, test_s1_encodings, test_s2_encodings, test_targets)
collator = MyDataCollator(EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"))
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=8, 
    shuffle=False, 
    collate_fn=collator,
)
predictor = ContactMapPredictor(
                    encoder_module = EsmEncoder(), 
                    num_linear_layers = FFN_DEPTH, dropout_rate=FFN_DROPOUT, 
                    ffn_width=FFN_WIDTH, 
                    activation_fn = FFN_ACTIVATION)
results = {}
for root, dirs, files in os.walk('./main_results/'):
    for f in files:
        if f == 'esm2_contact_map_predictor.pth':
            model_path = os.path.join(root, f)
            print(f"Evaluating model: {model_path}")
            try:
                #load model
                predictor.load_state_dict(torch.load(model_path, map_location=device))
                predictor = predictor.to(device)
                predictor.eval()
                
                #define criterions
                bce_criterion = MaskedBCEWithLogitsLoss(pos_weight=torch.tensor(BCE_POS_WEIGHT)).to(device)
                
                #evaluate
                test_bce_loss, test_mcc = evaluate(
                    predictor, 
                    test_loader, 
                    bce_criterion, 
                    device
                )
                
                print(f"Test BCE Loss: {test_bce_loss:.4f}, Test MCC: {test_mcc:.4f}")
                results[model_path] = {
                    'test_bce_loss': test_bce_loss,
                    'test_mcc': test_mcc
                }
            except Exception as e:
                print(f"Error evaluating model {model_path}: {e}")
                results[model_path] = 'error'
                continue
            
#save results
print(results)
import pandas as pd
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv('./db5_esm2_model_evaluations.csv')
print("Saved DB5 evaluation results to db5_esm2_model_evaluations.csv") 