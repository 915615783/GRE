import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

class LstmEncoder(nn.Module):
    def __init__(self, hidden_size, num_embeddings, embedding_dim, num_pos_embedding=128*2, pos_embedding_dim=10, 
        word_mat_path=None):
        super().__init__()
        if word_mat_path != None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(np.load(word_mat_path)), padding_idx=400001)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=400001)

        self.pos1_embedding = nn.Embedding(num_pos_embedding, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(num_pos_embedding, pos_embedding_dim, padding_idx=0)

        self.encoder = nn.LSTM(embedding_dim + 2*pos_embedding_dim, hidden_size, batch_first=True)

    def forward(self, word, pos1, pos2, mask):
        '''
        word: (b, max_l)
        pos: (b, max_l)
        mask: (b, max_l)
        '''
        w = self.embedding(word)   # (b, max_l, embedding_dim) 
        p1 = self.pos1_embedding(pos1)   # (b, max_l, pos_dim)
        p2 = self.pos2_embedding(pos2)

        emb = torch.cat([w, p1, p2], -1)   # (b, max_l, lstm_input_dim)

        emb = pack_padded_sequence(emb, mask.sum(-1), batch_first=True)
        output, (h_n, c_n) = self.encoder(emb)   # h and c: (layers*directions, batch, hidden)

        output = pad_packed_sequence(output, batch_first=True)   # (b, max_l, h)
        h_n = h_n.squeeze(0) # (b, h)
        c_n = c_n.squeeze(0)
        return output, (h_n, c_n)





        