import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

class LstmDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, word_mat_path=None):
        '''
        num_emgbeddings also means the output size of softmax
        '''
        super().__init__()
        if word_mat_path != None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(np.load(word_mat_path)), padding_idx=400001)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=400001)
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, num_embeddings + 1)   # +1: '[EOS]'
        self.sos_input = torch.tensor(np.random.randn(1, embedding_dim))


    def forward(self, h, c, max_length, is_teacher_force=None, teacher_input=None):
        '''
        这个是给定label长度，用于训练，predict需要可变长度
        h: (b, h)
        c: (b, h)
        max_length (int): max length of the label text, include [EOS]
        teacher_input : padded tensor
        '''
        output = []
        input_x = torch.cat([self.sos_input]*(h.size(0)), 0)   # (b, embedding_dim)
        for i in range(max_length):
            h, c = self.lstm_cell(input_x, (h, c))
            output.append(self.linear(h))
            input_x = output[-1][:, :-1].topk(1, dim=-1)   # remove the eos dim, (b, 1)
            input_x = input_x.squeeze(-1)  # (b)
            input_x = self.embedding(input_x)  # b, embedding_dim

        output = torch.stack(output, 1)   # (b, max_l, num_embeddings + 1)
        return output




        
