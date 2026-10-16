models.py
    Contains all code for the models, assuming a pytorch base. Models have some sort of encoder foot, and then a fixed head.

    EsmEncoder - Wraps the esm2_t6_8M_UR50D model. Outputs the last_hidden_state embeddings for each token in the sequence, ignoring the first <CLS> and last <EOS> ones.

    ContactMapPredictor - takes an encoder module and adds to it the appropriate head. This head first creates the contact map with concatenated embeddings, and then applys a ffn to each element. 

data_utils.py
    Contains custom objects to handle the dipsplus dataset, including the processing and padding.

     get_complex_dill - just a faster way to obtain the dictionary from a dill file

     DipsDataset - A Torch Dataset, meaning it has __len__ and __getitem__ methods. Built off of a list of keys (e.g. sz/1sz0.pdb_0.dill) that is used internally to index from dictionaries of pre-tokenized inputs and prebuilt sparse target contact maps (in CSR format). Each element of the dataset then consists of input_ids and attention_mask for sequence1, input_ids and attention_mask for sequence2, and the target matrix converted back to a full matrix. 

     MyDataCollator - a custom collator that pads inputs and targets. The sequence inputs are padded by model tokenizer, while the contact map is padded manually. Output is the padded inputs/outputs, as well as a mask for the contact map.

     filter_keys - filters a list of keys based on the value of some column of an associated df. Only supports binary comparisons (x<y).

     filter_into_range - filters a list of keys such that all values of an associated df column fall within a specified range of values 

     filter_interpercentile_range - filters a list of keys such that all values of an associated df column fall within two percentiles (e.g. 10th and 30th percentile)

     filer_random_range - filters a list of keys such that all values of an associated df column fall withn a randomly selected interpercentile range of a specified size. Returns both the filtered key list and the range used, such that the same range may be used for the validation set

     get_size_of_feature_range - counts the number of rows in a df such that the value of some column falls within a specified range of values

     stratify_sample - uses StratifiedKFold to sample from a list of keys such that the sample adequately covers the full range of some column in an associated df








criterions.py 
    Contains custom objects for calculating loss and other metrics on our contact maps. All use the mask created by the custom data collator to correctly ignore anything that's just padding.

    MaskedBCEWithLogitsLoss - given logits, target map, and a mask, get average BCE across the map

    MaskedMCC - given predicted probabilities (logits have gone thru the sigmoid), target map, and a mask, get the MCC treating each element of the contact map as a separate binary prediction.

    MaskedRecall - as MaskedMCC but calculates recall

    MaskedPrecision - as MaskedMCC but calculates precision 


train_test_routines.py
    Functions to carry out training and model evaluation.

    train_one_epoch - '''trains a model one epoch across an entire data loader. returns the average loss over the epoch.'''

    evaluate - evaluates a model across an entire data loader. returns the average loss over the epoch, as well as Matthews Correlation Coefficient (MCC).