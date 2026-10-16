'''
Contains the datasets objects, as well as the scripts used to stratify.
'''

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from transformers import EsmModel, EsmTokenizer, DataCollatorWithPadding
import dill

def get_complex_dill(path):
    '''Loads in a complex object stored with dill.'''
    with open(path, 'rb') as f:
        complex_obj = dill.load(f)
    return complex_obj

class DipsDataset(Dataset):
    '''
    Dataset for the DIPS Plus data. Construction is based on the keys (from the pairs-postprocessed text files), and dictionaries that map those keys to precomputed encodings and target contact maps. The encodings are the tokenizer outputs for each sequence in the pair. 
    
    The ordering is based on the keys list provided during construction. When indexing, that list is used to retrieve a key, and then that key is used to grab the encodings and targets.
    '''
    
    def __init__(self, keys, s1_encodings, s2_encodings, targets):
        self.s1_encodings = s1_encodings
        self.s2_encodings = s2_encodings
        self.targets = targets
        self.keys = keys

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            # Handle a single index or a slice
            key = self.keys[idx]
            s1 = self.s1_encodings[key]
            s2 = self.s2_encodings[key]
            target = torch.from_numpy(self.targets[key].toarray().astype(np.float32))

            return {
                'input1_ids': s1['input_ids'].squeeze(0),
                'input1_attention_mask': s1['attention_mask'].squeeze(0),
                'input2_ids': s2['input_ids'].squeeze(0),
                'input2_attention_mask': s2['attention_mask'].squeeze(0),
                'target': target
            }
        elif isinstance(idx, list):
            # Handle a list of indices (batch processing)
            batch = []
            for i in idx:
                batch.append(self.__getitem__(i))
            return batch
        else:
            raise TypeError("Invalid argument type.")

class MyDataCollator(DataCollatorWithPadding):
    '''Performs dynamic padding on inputs and targets. Padding the inputs is straightforward, but padding the target contact maps requires custom logic to ensure all targets in a batch have the same dimensions. The targets are padded with zeros, and a mask is created to indicate what's actually the contact map and what's just been padded
    
    Thus this collator outputs not just inputs and targets, but an additional mask tensor.
    '''
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Separate the features for each input stream and the targets
        input1_ids = [f['input1_ids'] for f in features]
        input1_attention_mask = [f['input1_attention_mask'] for f in features]
        input2_ids = [f['input2_ids'] for f in features]
        input2_attention_mask = [f['input2_attention_mask'] for f in features]
        targets = [f['target'] for f in features]

        # Pad the input sequences using the tokenizer's pad method
        batch_input1 = self.tokenizer.pad(
            {'input_ids': input1_ids, 'attention_mask': input1_attention_mask},
            return_tensors='pt'
        )
        batch_input2 = self.tokenizer.pad(
            {'input_ids': input2_ids, 'attention_mask': input2_attention_mask},
            return_tensors='pt'
        )


        # 1. Determine the maximum dimensions of the target tensors in the batch.
        #    This assumes your targets are 2D tensors (e.g., contact maps).
        max_height = max(t.shape[0] for t in targets)
        max_width = max(t.shape[1] for t in targets)

        # 2. Pad each target tensor to the maximum dimensions and collect them in a list. Also create a mask.
        masks = []
        padded_targets = []
        for t in targets:
            # Get the current shape
            h, w = t.shape
            # Calculate the padding needed for right and bottom sides
            padding = (0, max_width - w, 0, max_height - h)
            # Apply padding. 'constant' padding fills with a value, default is 0.
            padded_t = torch.nn.functional.pad(t, padding, "constant", 0)
            padded_targets.append(padded_t)

            mask = torch.zeros((max_height, max_width), dtype=torch.bool)
            mask[:h, :w] = 1
            masks.append(mask)
        masks = torch.stack(masks)
            
        # 3. Stack the list of padded tensors into a single batch tensor.
        batch_targets = torch.stack(padded_targets)


        return {
            'input1_ids': batch_input1['input_ids'],
            'input1_attention_mask': batch_input1['attention_mask'],
            'input2_ids': batch_input2['input_ids'],
            'input2_attention_mask': batch_input2['attention_mask'],
            'labels': batch_targets,  # Use the newly created padded tensor
            'mask': masks
        }



def filter_keys(keys, metrics_df, condition_col, condition_val, operator='=='):
    """
    Filters a list of keys based on a condition in the metrics DataFrame.

    Args:
        keys (list): The list of keys to filter (e.g., train_paths).
        metrics_df (pd.DataFrame): DataFrame with metrics.
        condition_col (str): The column in metrics_df to filter on.
        condition_val: The value to match in the condition column.
        operator (str): The comparison operator ('==', '!=', '<', '>', '<=', '>=').

    Returns:
        list: A filtered list of keys.

    Example:
        # Filter for interface_res_count <= 200
        filtered_keys = filter_keys(keys,  metrics_df, 'interface_res_count', 200, operator='<=')
    """
    print(f"Filtering keys where '{condition_col}' {operator} {condition_val}...")
    
    # Apply the specified operator
    if operator == '==':
        filtered_df = metrics_df[metrics_df[condition_col] == condition_val]
    elif operator == '!=':
        filtered_df = metrics_df[metrics_df[condition_col] != condition_val]
    elif operator == '<':
        filtered_df = metrics_df[metrics_df[condition_col] < condition_val]
    elif operator == '>':
        filtered_df = metrics_df[metrics_df[condition_col] > condition_val]
    elif operator == '<=':
        filtered_df = metrics_df[metrics_df[condition_col] <= condition_val]
    elif operator == '>=':
        filtered_df = metrics_df[metrics_df[condition_col] >= condition_val]
    else:
        raise ValueError("Unsupported operator. Use '==', '!=', '<', '>', '<=', '>='.")
    
    # Use a set for efficient lookup
    print(filtered_df)
    valid_paths = set(filtered_df['complex_path'])
    
    # Filter original keys
    filtered_keys_list = [k for k in keys if k in valid_paths]
    print(f"Original keys: {len(keys)}, Filtered keys: {len(filtered_keys_list)}")
    
    return filtered_keys_list

def filter_into_range(keys, metrics_df, condition_col, min_val, max_val):
    """
    Filters a list of keys based on a range condition in the metrics DataFrame.

    Args:
        keys (list): The list of keys to filter (e.g., train_paths).
        metrics_df (pd.DataFrame): DataFrame with metrics.
        condition_col (str): The column in metrics_df to filter on.
        min_val: The minimum value for the range (inclusive).
        max_val: The maximum value for the range (inclusive).

    Returns:
        list: A filtered list of keys.

    Example:
        # Filter for interface_res_count between 50 and 200
        filtered_keys = filter_into_range(keys,  metrics_df, 'interface_res_count', 50, 200)
    """
    print(f"Filtering keys where '{condition_col}' is between {min_val} and {max_val}...")
    
    # Apply the range condition
    filtered_df = metrics_df[(metrics_df[condition_col] >= min_val) & (metrics_df[condition_col] <= max_val)]
    
    # Use a set for efficient lookup
    valid_paths = set(filtered_df['complex_path'])
    
    # Filter original keys
    filtered_keys_list = [k for k in keys if k in valid_paths]
    print(f"Original keys: {len(keys)}, Filtered keys: {len(filtered_keys_list)}")
    
    return filtered_keys_list

def filter_interpercentile_range(keys, metrics_df, condition_col, lower_percentile, upper_percentile):
    """
    Filters a list of keys based on interpercentile range in the metrics DataFrame.

    Args:
        keys (list): The list of keys to filter (e.g., train_paths).
        metrics_df (pd.DataFrame): DataFrame with metrics.
        condition_col (str): The column in metrics_df to filter on.
        lower_percentile (float): The lower percentile (0-100).
        upper_percentile (float): The upper percentile (0-100).

    Returns:
        list: A filtered list of keys.
    """
    print(f"Filtering keys where '{condition_col}' is between the {lower_percentile}th and {upper_percentile}th percentiles...")
    
    # Calculate the lower and upper bounds based on percentiles
    lower_bound = metrics_df[condition_col].quantile(lower_percentile / 100)
    upper_bound = metrics_df[condition_col].quantile(upper_percentile / 100)
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

    # Apply the range condition
    filtered_df = metrics_df[(metrics_df[condition_col] >= lower_bound) & (metrics_df[condition_col] <= upper_bound)]

    # Use a set for efficient lookup
    valid_paths = set(filtered_df['complex_path'])

    # Filter original keys
    filtered_keys_list = [k for k in keys if k in valid_paths]
    print(f"Original keys: {len(keys)}, Filtered keys: {len(filtered_keys_list)}")

    return filtered_keys_list

def filter_random_range(keys, metrics_df, condition_col, percentile_size = 50):
    """
    Filters a list of keys based on a random range of a specified quantile size in the metrics DataFrame.

    Args:
        keys (list): The list of keys to filter (e.g., train_paths).
        metrics_df (pd.DataFrame): DataFrame with metrics.
        condition_col (str): The column in metrics_df to filter on.
        quantile_size (float): The size of the quantile range (0-100).
    Returns:
        filtered_list: A filtered list of keys.
        start_percentile: the start_percentile selected
        end_percentiole: the end_percentile selected
    """

    #first get the allowed start val of the quantile range; we can go from [0, quant] to [1-quant, 1]
    if percentile_size <= 0 or percentile_size > 100:
        raise ValueError("percentile_size must be between 0 and 100")
    max_start = 100.0 - percentile_size

    start_percentile = np.random.uniform(0, max_start)
    end_percentile = start_percentile + percentile_size

    print(f"Filtering keys where '{condition_col}' is between the {start_percentile:.2f}th and {end_percentile:.2f}th percentiles...")
    filtered_list = filter_interpercentile_range(keys, metrics_df, condition_col, start_percentile, end_percentile)

    return filtered_list, start_percentile, end_percentile


def get_size_of_feature_range(metrics_df, condition_col, min_val, max_val):
    """
    Get the number of entries in the metrics DataFrame that fall within a specified range.

    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics.
        condition_col (str): The column in metrics_df to filter on.
        min_val: The minimum value for the range (inclusive).
        max_val: The maximum value for the range (inclusive).

    Returns:
        int: The count of entries within the specified range.

    Example:
        # Get count of entries where interface_res_count is between 50 and 200
        count = get_size_of_feature_range(metrics_df, 'interface_res_count', 50, 200)
    """
    filtered_df = metrics_df[(metrics_df[condition_col] >= min_val) & (metrics_df[condition_col] <= max_val)]
    count = len(filtered_df)
    print(f"Number of entries where '{condition_col}' is between {min_val} and {max_val}: {count}")
    return count

def stratify_sample(keys, metrics_df, stratify_col, end_size, num_bins = 10):
    """
    Stratified sampling of keys based on a specified column in the metrics DataFrame. Bin the stratify_col as classification labels, use stratified kfold from sklearn

    Args:
        keys (list): The list of keys to sample from (e.g., train_paths).
        metrics_df (pd.DataFrame): DataFrame with metrics.
        stratify_col (str): The column in metrics_df to stratify by.
        end_size (int): The desired size of the sampled list.
    """

    #error check, deal with if end_size is a fraction and not a raw number
    if type(end_size) is not int and type(end_size) is not float:
        raise ValueError("end_size must be an integer or float")

    if type(end_size) is float:
        if end_size <= 0 or end_size > 1:
            raise ValueError("If end_size is a float, it must be between 0 and 1")
        end_size = int(len(keys) * end_size)

    temp_df = metrics_df[metrics_df['complex_path'].isin(keys)].copy()

    # First bin the stratify_col into categories
    est = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile')
    temp_df['binned'] = est.fit_transform(temp_df[stratify_col].values.reshape(-1, 1)).astype(int)

    # Now perform stratified sampling using StratifiedKFold
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=end_size)

    for trn, _ in splitter.split(temp_df, temp_df['binned']):
        sampled_df = temp_df.iloc[trn]

    sampled_keys = [k for k in keys if k in set(sampled_df['complex_path'])]

    print(f"Original keys: {len(keys)}, Sampled keys: {len(sampled_keys)}")
    return sampled_keys

