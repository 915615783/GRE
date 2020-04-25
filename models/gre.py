import torch
from torch import nn
from encoder.lstm_decoder import LstmDecoder
from encoder.lstm_encoder import LstmEncoder
import torch.nn.functional as F 
import numpy as np
from logger import logger

class Gre(nn.Module):
    '''
    Input the batch data. Output the loss, metric and some output tensor.
    '''
    def __init__(self, hidden_size, num_embeddings, embedding_dim, word_mat_path, label_pad_index, is_teacher_force=False):
        '''
        label_pad_index (int): ignore index when calculating loss 
        '''
        super().__init__()
        self.is_teacher_force = is_teacher_force
        logger.info('Is Teacher Force : %s'% is_teacher_force)
        self.encoder = LstmEncoder(hidden_size, num_embeddings, embedding_dim, word_mat_path=word_mat_path)
        self.decoder = LstmDecoder(num_embeddings, embedding_dim, hidden_size, word_mat_path=word_mat_path)
        
        # print(id(self.decoder.embedding), id(self.encoder.embedding))
        self.decoder.embedding = self.encoder.embedding
        # print(id(self.decoder.embedding), id(self.encoder.embedding))

        self.label_pad_index = label_pad_index
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.label_pad_index)

    def forward(self, word, pos1, pos2, mask, label_token, label_len):
        '''
        label_token: (b, max_l)
        '''
        encoder_out, (h_n, c_n) = self.encoder(word, pos1, pos2, mask)
        decoder_out = self.decoder(h_n, c_n, label_token.size(1), self.is_teacher_force, label_token)
        # nllloss ignore_index，padding需要是一个唯一的值，但是前面label——token的padding好像用了不唯一的0
        # 改了
        loss = self.loss_func(decoder_out.view(-1, decoder_out.size(-1)), label_token.view(-1))
        return decoder_out, loss   # (b, max_length, num_embeddings + 1)
