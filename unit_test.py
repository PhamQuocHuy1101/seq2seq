import torch
from architecture import Encoder, Decoder, EncoderDecoder

vocab_size = 26
embedding_size = 32
hidden_size = 64
num_encode_layer = 3
num_decode_layer = 2

encoder = Encoder(vocab_size, embedding_size, hidden_size, num_encode_layer, 0.1)
decoder = Decoder(vocab_size+4, embedding_size, hidden_size, num_decode_layer, 0.1)

inputs = torch.tensor([[0, 1, 2, 3, 5], [0, 3, 5, 3, 7]])
out_enc, state_enc = encoder(inputs)
print("Encode")
print(out_enc.shape, state_enc[0].shape, state_enc[1].shape)

# init_state = state_enc[1][-1].repeat(2*num_decode_layer, 1, 1)
inputs2= torch.tensor([[0], [0]])
dec_state = decoder.init_state(state_enc[1][-1])
out_dec, state_dec = decoder(inputs2, dec_state)
print("Decode")
print(out_dec.shape, state_dec[0].shape, state_dec[1].shape)

seq2seq = EncoderDecoder(encoder, decoder)
inputs2= torch.tensor([[0, 1, 2, 3, 5, 2, 1], [0, 3, 2, 5, 3, 7, 1]])
out, state = seq2seq(inputs,inputs2)
print("Seq2seq")
print(out.shape, state[0].shape, state[1].shape)
