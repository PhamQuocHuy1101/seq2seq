import torch
import torch.nn as nn

from architecture.block import Encoder, Decoder

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, *args):
        _, enc_state = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_state[1][-1], *args) # last endcoder state
        dec_outputs, dec_state = self.decoder(dec_X, dec_state)
        return dec_outputs, dec_state