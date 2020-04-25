import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import global_var

class LstmDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, word_mat_path=None):
        '''
        num_emgbeddings also means the output size of softmax
        '''
        super().__init__()
        if word_mat_path != None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(np.load(word_mat_path), dtype=torch.float32), freeze=False)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, num_embeddings + 1)   # +1: '[EOS]'
        self.sos_input = nn.Parameter(torch.tensor(np.random.randn(1, embedding_dim), dtype=torch.float32), ) 


    def forward(self, h, c, max_length, is_teacher_force=False, teacher_input=None):
        '''
        这个是给定label长度，用于训练，predict需要可变长度
        h: (b, h)
        c: (b, h)
        max_length (int): max length of the label text, include [EOS]
        teacher_input : padded tensor(b, max_l)
        '''
        # clip the eos 
        teacher_input = torch.clamp(teacher_input, 0, self.embedding.weight.size(0)-1)

        output = []
        input_x = torch.cat([self.sos_input]*(h.size(0)), 0)   # (b, embedding_dim)
        # input_x = c
        for i in range(max_length):
            h, c = self.lstm_cell(input_x, (h, c))
            output.append(self.linear(h))
            if (not is_teacher_force):
                # input_x = c
                _, input_x = output[-1][:, :-1].topk(1, dim=-1)   # remove the eos dim, (b, 1)
                input_x = input_x.squeeze(-1)  # (b)
                input_x = self.embedding(input_x)  # b, embedding_dim
                input_x = F.relu(input_x)
            else:
                input_x = teacher_input[:, i]  # (b)
                input_x = self.embedding(input_x)


        output = torch.stack(output, 1)   # (b, max_l, num_embeddings + 1)
        return output




        
