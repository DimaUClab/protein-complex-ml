from transformers import BertForPreTraining, BertConfig, BertModel
from tokenization import FullTokenizer
import tokenization
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
from tqdm import tqdm

def generate_input_features_from_seq_list(seqs, labels, tokenizer, pad_to=1024, return_as_np_array=False):
    tseqs = [[tokenization.CLS_TOKEN] + tokenizer.tokenize(s) for s in seqs]
    input_mask = [[1]*len(ts) + [0]*(pad_to - len(ts)) for ts in tseqs]
    segment_ids = [[0]*pad_to for ts in tseqs]
    
    if pad_to is not None:
        for ts in tseqs:
            assert len(ts) <= pad_to
            ts += [tokenization.PAD_TOKEN]*(pad_to - len(ts))
            assert len(ts) == pad_to, ts
            
    input_ids = [tokenizer.convert_tokens_to_ids(tseq) for tseq in tseqs]
    
    if return_as_np_array:
        input_dict = {
            'input_ids': np.array(input_ids),
            'input_mask': np.array(input_mask),
            'segment_ids': np.array(segment_ids),
            'labels': np.array(labels)
        }
    else:
        input_dict = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'labels': labels
        }
            
    return input_dict

def fasta_read(fasta_file):
    headers = []
    seqs = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        headers.append(seq_record.id)
        seqs.append(str(seq_record.seq))

    return headers, seqs


def parse_fastas(data_file, prepend_m):
    headers, seqs = fasta_read(data_file)

    # Add a stop char to each sequence to be consistent
    # with how the model was trained.

    # Prepend an M. Again reflective of how the model
    # was trained.
    if prepend_m:
        for i in range(len(seqs)):
            if seqs[i][0] != 'M':
                seqs[i] = 'M' + seqs[i]

    # Remove sequences that are too long for the model
    seqs = [s if len(s) < 1023 else s[:1022] for s in seqs]
    # mask = np.array([len(s) for s in seqs]) <= 1023
    # print('Sequences being removed due to length:', np.sum(~mask))
    # print('Sequences being removed:', np.array(headers)[~mask], np.array(seqs)[~mask])
    seqs = [s + '*' for s in seqs]
    seqs = list(np.array(seqs))
    headers = list(np.array(headers))

    return seqs, headers

def get_bert_embed(input_dict, m, tok, device, normalize=True,\
        summary_method="MEAN", tqdm_bar=True, batch_size=64):
    '''
    This function is used to generate embedding vectors for proteins
    
    param:
        input_dict: input dictionary, should contain input_ids, input_mask, segment_ids, labels
        m: model
        tok: tokenizer
        device: device
        normalize: normalize the embeddings or not
        summary_method: method for generating embeddings from bert output, CLS for class token or MEAN for mean pooling. MEAN by default.
        tqdm_bar: progress bar
        batch_size: batch size for bert
    return:
        embeddings in numpy array with shape (phrase_list_length, embedding_dim)
    '''
    m = m.to(device)
    input_ids = input_dict['input_ids']
    attention_mask = input_dict['input_mask']
    # print(len(input_ids))
    m.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            attention_mask_gpu_0 = torch.LongTensor(attention_mask[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = m(input_gpu_0, attention_mask_gpu_0)[1]
                # print(embed.shape)
            if summary_method == "MEAN":
                res = m(input_gpu_0, attention_mask_gpu_0)[0]
                embed = torch.mean(m(input_gpu_0, attention_mask_gpu_0)[0], dim=1)
                # print(embed.shape)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 == 0:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
        if tqdm_bar:
            pbar.close()
    output_list.append(output.cpu().numpy())
    # print('end')
    return np.concatenate(output_list, axis=0)



if __name__ == "__main__":
    model = BertModel.from_pretrained("aminobert")

    tokenizer = FullTokenizer(k=1, token_to_replace_with_mask='X')
    data_file = 'test.fasta'
    seqs, headers = parse_fastas(data_file=data_file, prepend_m=True)
    input_dict = generate_input_features_from_seq_list(seqs, labels=None, tokenizer=tokenizer, pad_to=1024, return_as_np_array=True)
    embedding = get_bert_embed(input_dict, model, tokenizer, device='cuda:0', normalize=True, summary_method="MEAN", tqdm_bar=True, batch_size=64)
    print(embedding.shape)
    
    

