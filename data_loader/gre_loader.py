import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset, which provides the label text
    """
    def __init__(self, path, tokenizer, label_tokenizer):
        '''
        encoder: we need its tokenizer here.
        '''
        self.tokenizer = tokenizer
        with open(path, 'r') as f:
            json_data = json.load(f)
        self.data = []   # a list of sample(dict with key 'relationid')
        for rel in json_data.keys():
            items = json_data[rel]
            for item in items:
                item['relationid'] = rel
            self.data.extend(items)
        self.label_tokenizer = label_tokenizer   # input a relation id string ang return tokens of label
        

    # def __getraw__(self, item):
    #     word, pos1, pos2, mask = self.encoder.tokenizer(item['tokens'],
    #         item['h'][2][0],
    #         item['t'][2][0])
    #     return word, pos1, pos2, mask

    def __getitem__(self, index):
        # 只用返回一个样本，返回文本label和labelid
        item = random.choice(self.data)
        word, pos1, pos2, mask = self.tokenizer(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        label_token = self.label_tokenizer(item['relationid'])
        
        return word, pos1, pos2, mask, label_token


    def __len__(self):
        return 1000000000

def collate_fn(data):
    data.sort(key=lambda x:len(x[0]), reverse=True)
    word, pos1, pos2, mask, label_token = zip(*data)
    word = [torch.tensor(word, dtype=torch.int32) for i in word]
    pos1 = [torch.tensor(pos1, dtype=torch.int32) for i in pos1]
    pos2 = [torch.tensor(pos2, dtype=torch.int32) for i in pos2]
    mask = [torch.tensor(mask, dtype=torch.int32) for i in mask]
    label_len = [len(label_token) for i in label_token]
    label_token = [torch.tensor(label_token, dtype=torch.int32) for i in label_token]

    word = pad_sequence(word, batch_first=True, padding_value=400001)
    pos1 = pad_sequence(pos1, batch_first=True, padding_value=400001)
    pos2 = pad_sequence(pos2, batch_first=True, padding_value=400001)
    mask = pad_sequence(mask, batch_first=True, padding_value=400001)
    label_token = pad_sequence(label_token, batch_first=True, padding_value=400001)

    return word, pos1, pos2, mask, label_token, label_len
    
    # 解决不同长度的串的问题

def get_data_loader(dataset, batch_size, num_workers=1):
    data_loader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    return data_loader


# d = FewRelDataset(r'data\val_wiki.json', None, None)
# # print(d.data[0])
        






















# import pandas as pd

# import json
# path = "data/props_raw.json"
# def get_json(path=path):
#     with open(path) as f:
#         return json.load(f)
        
# a = get_json(path)
# a = {i['id']:i['label'] for i in a}

# words = ' '.join(a.values()).lower().split(' ')

# # dic = get_json(r'pretrain\glove\glove_word2id.json')
# # words = [dic.get(i, 'None') for i in words]
# # print(pd.value_counts(words))
# # print(len(words))
# print(set(words).__len__())
