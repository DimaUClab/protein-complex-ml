import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from transformers import EsmModel, EsmTokenizer, DataCollatorWithPadding, BertModel, BertTokenizer


#All encoder models must output a tensor of shape (B, L, D) where B=batch size, L=sequence length, D=embedding dimension. Tokens such as BOS and EOS should be stripped by the encoder.

class EsmEncoder(torch.nn.Module):
    """
    Wrapper for ESM-2 models.
    """
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.encoder = EsmModel.from_pretrained(model_name)
        
        # This is the crucial attribute our Predictor head needs
        self.output_dim = self.encoder.config.hidden_size

        self.tokenizer = EsmTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # Run the encoder
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Return *only* the sequence embeddings, slicing off the
        # beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens.
        return outputs.last_hidden_state[..., 1:-1, :]



class ProtBertEncoder(torch.nn.Module):
    '''
    Wrapper for ProtBert model.
    '''

    def __init__(self, model_name="Rostlab/prot_bert", device='cpu'):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        
        # This is the crucial attribute our Predictor head needs
        self.output_dim = self.encoder.config.hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    def forward(self, input_ids, attention_mask):
        # Run the encoder
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Return *only* the sequence embeddings, slicing off the
        # beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens
        return outputs.last_hidden_state[..., 1:-1, :]


class AminoBertEncoder(torch.nn.Module):
    '''
    Wrapper for AminoBert model.
    '''

    def __init__(self, model_path, device='cpu'):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_path)
        self.output_dim = self.encoder.config.hidden_size
        #i think i can just skip the tokenizer? i don't actually use it at any point.

    def forward(self, input_ids, attention_mask):
        # Run the encoder
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Return *only* the sequence embeddings, slicing off the
        # beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens
        return outputs.last_hidden_state[..., 1:-1, :]

class ContactMapPredictor(torch.nn.Module):
    def __init__(self, encoder_module, num_linear_layers=2, dropout_rate = 0.1, ffn_width=120, activation_fn='relu'):
        """
        Initializes the contact map prediction head.
        
        Args:
            encoder_module (torch.nn.Module): 
                An instantiated encoder module (like EsmEncoder below).
                This module MUST have an attribute: `self.output_dim`
            
            num_linear_layers (int): 
                Number of hidden layers in the FFN head.
        """
        super().__init__()
        
        # --- 1. Dependency Injection ---
        # Store the encoder module that was passed in
        self.encoder = encoder_module
        
        # --- 2. Get Encoder Dimension ---
        # We ask the encoder module for its output dimension
        if not hasattr(self.encoder, 'output_dim'):
            raise ValueError("Encoder module must have an 'output_dim' attribute.")
            
        encoder_dim = self.encoder.output_dim
        
        # --- 3. Build the FFN Head ---
        # The FFN's input size is determined by the encoder's dim
        ffn_input_dim = encoder_dim * 2 # We concatenate two proteins
        self.ffn = self._build_ffn(ffn_input_dim, num_linear_layers, ffn_width, dropout_rate, activation_fn)


    def _build_ffn(self, ffn_input_dim, num_linear_layers, ffn_width, dropout_rate, activation_fn='relu'):
        
        if activation_fn == 'relu':
            activation = torch.nn.ReLU
        elif activation_fn == 'gelu':
            activation = torch.nn.GELU
        elif activation_fn == 'tanh':
            activation = torch.nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")   
        
        """Helper function to build the FFN."""
        ffn_layers = []
        
        hidden_dim = ffn_width 
        
        ffn_layers.append(torch.nn.Linear(ffn_input_dim, hidden_dim))
        ffn_layers.append(activation())
        ffn_layers.append(torch.nn.Dropout(dropout_rate))
        
        for _ in range(num_linear_layers):
            ffn_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            ffn_layers.append(activation())
            ffn_layers.append(torch.nn.Dropout(dropout_rate))
            
        # Final layer to predict one value (the contact probability)
        ffn_layers.append(torch.nn.Linear(hidden_dim, 1))
        
        return torch.nn.Sequential(*ffn_layers)

    def forward(self, input1_ids, input1_attention_mask, input2_ids, input2_attention_mask):
        """
        The forward pass is now much cleaner.
        """
        
        # 1. Get embeddings from the injected encoder.
        #    The encoder itself handles token-slicing.
        E1 = self.encoder(input_ids=input1_ids, attention_mask=input1_attention_mask)
        E2 = self.encoder(input_ids=input2_ids, attention_mask=input2_attention_mask)
        
        # 2. Tiling logic 
        E1_tiled = E1.unsqueeze(2) 
        E2_tiled = E2.unsqueeze(1)
        L1 = E1_tiled.shape[1]
        L2 = E2_tiled.shape[2]

        # 3. Concatenate features
        concatenated_features = torch.cat([E1_tiled.repeat(1, 1, L2, 1), E2_tiled.repeat(1, L1, 1, 1)], dim=-1)

        # 4. Pass through FFN head
        #    Output shape: (B, L1, L2, 1)
        output = self.ffn(concatenated_features)
        
        # Squeeze to (B, L1, L2)
        final_map = output.squeeze(-1)

        return final_map