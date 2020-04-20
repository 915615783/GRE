import json
import numpy as np

class Tokenizer():
    def __init__(self, word2id_path, is_pad_cut_to_max_length=False, max_length=128):
        with open(word2id_path, 'r') as f:
            self.word2id = json.load(f)
        self.is_pad_cut_to_max_length = is_pad_cut_to_max_length
        self.max_length = max_length

    def __call__(self, tokens, h, t):
        '''h and t indicate the position of entity. it is a list of int'''
        # token ->  index
        indexed_tokens = [self.word2id[token.lower()] if (token.lower() in self.word2id) else self.word2id['[UNK]'] for token in tokens]

        # padding or cutting
        tokens_len = len(indexed_tokens)
        if self.is_pad_cut_to_max_length:
            if tokens_len < self.max_length:
                indexed_tokens += [self.word2id['[PAD]']] * (self.max_length - tokens_len) 
            else:
                indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        indexed_tokens_len = len(indexed_tokens)
        pos1 = np.zeros((indexed_tokens_len), dtype=np.int32)
        pos2 = np.zeros((indexed_tokens_len), dtype=np.int32)
        h = min(indexed_tokens_len, h[0])
        t = min(indexed_tokens_len, t[0])
        for i in range(indexed_tokens_len):
            pos1[i] = i - h + self.max_length
            pos2[i] = i - t + self.max_length

        # mask
        mask = np.zeros((indexed_tokens_len), dtype=np.int32)
        mask[:tokens_len] = 1
        return indexed_tokens, pos1, pos2, mask

# to = Tokenizer(r'pretrain\glove\glove_word2id.json', True)
# print(to(['wo', 'ri', 'ni', 'ma'], [0,1], [2]))
        

        