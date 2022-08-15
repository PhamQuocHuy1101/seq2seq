from unicodedata import bidirectional
import torch
import torch.nn as nn

import architecture.utils as utils

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layer, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layer, 
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.apply(utils.init_seq2seq)

    def forward(self, X, *arg):
        '''
            Args:
                X: [batch, n_step]
            Return:
                out: [batch, n_step, hidden]
                state: [h, c]
                h: [2*n_layer, batch, H_out]
                c: [2*n_layer, batch, H_cell]
        '''
        embs = self.embedding(X.type(torch.long)) # [batch, n_step, emb]
        out, state = self.rnn(embs, *arg)

        return out, state
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layer, dropout = 0.1, use_bid = True):
        super(Decoder, self).__init__()
        self.num_layer = num_layer
        self.n_direct = 2 if use_bid else 1

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layer, 
                            dropout=dropout, bidirectional=use_bid, batch_first=True)
        self.cls = nn.Linear(self.n_direct*hidden_size if use_bid else hidden_size, vocab_size)
        self.apply(utils.init_seq2seq)
        
    def init_state(self, last_enc_state):
        init_state = last_enc_state.repeat(self.n_direct * self.num_layer, 1, 1)
        dec_state = (torch.ones_like(init_state), init_state)
        return dec_state

    def forward(self, X, state):
        embs = self.embedding(X)
        out, state = self.rnn(embs, state)
        out = self.cls(out)
        return out, state
