import os
import argparse
from easyConfig import setup_config
import random
import numpy as np
import torch
from architecture import EncoderDecoder
from dataset.dataset import Vocab, TrainingData
import loader
from dataset import utils


def test(model, input, max_length, device):
    # dataset
    src, seq = input
    src = torch.tensor(src, dtype=torch.long, device = device).unsqueeze(0)
    seq = torch.tensor(seq, dtype=torch.long, device = device).unsqueeze(0)
    out = [seq]
    print(seq.shape)

    _, enc_state = model.encoder(src)
    dec_state = model.decoder.init_state(enc_state[1][-1])
    for _ in range(max_length):
        dec_outputs, _ = model.decoder(out[-1], dec_state)
        Y = dec_outputs.softmax(dim=-1).argmax(dim=-1)
        out.append(torch.cat((seq, Y[0][-1].view(1, 1)), dim = -1))
        # print(Y.shape)
        # out.append(Y[0][-1].view(1, 1))
        # print(Y.shape)
    return out[-1].cpu().tolist()[0]
    return torch.cat(out, dim=1).cpu().squeeze().tolist()



def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark= False

if __name__ == '__main__':
    config = setup_config('config', 'config', False)
    seed_everywhere(config.seed)
    config.model.encoder.vocab_size = len(config.vocab.en) + 3
    config.model.decoder.vocab_size = len(config.vocab.fra) + 3
    
    device = torch.device(config.device)

    lang_1 = Vocab(config.vocab.en)
    lang_2 = Vocab(config.vocab.fra)

    encoder = loader.load_model('Encoder', config.model.encoder)
    decoder = loader.load_model('Decoder', config.model.decoder)
    model = EncoderDecoder(encoder, decoder)
    model.to(device = device)
    
    checkpoint = torch.load(config.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # print(config.text)

    input = utils.format(config.text, lang_1, lang_2, config.data.max_length)
    with torch.no_grad():
        out = test(model, input, config.data.max_length, device)

        # for item in out:
            # print(item)
        print(out)
        sentence = ''.join([lang_2[i] for i in out])
        print(sentence)
