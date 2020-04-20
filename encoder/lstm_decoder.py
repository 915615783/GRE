import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

class LstmDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, ):