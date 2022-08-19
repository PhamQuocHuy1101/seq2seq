import os
import time
import random
from easyConfig import setup_config

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm

from architecture import EncoderDecoder
from dataset.dataset import Vocab, TrainingData
import loader

def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark= False

def read_text(file):
    with open(file, 'r') as f:
        data = f.read().splitlines()
        return data

def collate_fn(batch):
    src, tar = [], []
    for b in batch:
        src.append(b[0])
        tar.append(b[1])
    return torch.tensor(src, dtype=torch.long), torch.tensor(tar, dtype=torch.long)

def compute_loss(out, Y, padding_token):
    l = F.cross_entropy(out.flatten(0, -2), Y.flatten(), reduction='none')
    mask = ( Y.flatten() != padding_token).type(torch.float32)
    return (l * mask).sum() / mask.sum()

def compute_accuracy(out, Y, padding_token):
    pred = torch.softmax(out, dim = -1).argmax(dim = -1)
    mark = (Y == pred).type(torch.float32)
    mask = (Y != padding_token).type(torch.float32)
    char_acc = (mark*mask).sum()
    seq_acc = ((mark*mask).sum(dim = 1) == mask.sum(dim = 1)).sum()
    return (char_acc, mask.sum()), (seq_acc, len(mask))

def test(model, src, device, max_length):
    seq = torch.tensor([[1]], dtype=torch.long, device=device)
    out = seq
    _, enc_state = model.encoder(src.unsqueeze(0))
    dec_state = model.decoder.init_state(enc_state[1][-1])
    for _ in range(max_length-1):
        dec_outputs, _ = model.decoder(out, dec_state)
        Y = dec_outputs.softmax(dim=-1).argmax(dim=-1)
        out = torch.cat((seq, Y), dim=-1)
    
    return out.cpu().squeeze(0)

def train(config):
    device = torch.device(config.device)
    # dataset
    lang_1 = Vocab(config.vocab.en)
    lang_2 = Vocab(config.vocab.fra)
    train_lines = read_text(config.data.train)
    val_lines = read_text(config.data.val)
    train_data = TrainingData(lang_1, lang_2, train_lines, config.data.max_length)
    train_loader = DataLoader(train_data, batch_size = config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = TrainingData(lang_1, lang_2, val_lines, config.data.max_length)
    val_loader = DataLoader(val_data, batch_size = config.val_batch_size, shuffle=False, collate_fn=collate_fn)
    print("Load data")

    encoder = loader.load_model('Encoder', config.model.encoder)
    decoder = loader.load_model('Decoder', config.model.decoder)
    model = EncoderDecoder(encoder, decoder)
    model.to(device = device)
    print("Load model")

    optimizer = optim.AdamW(params = model.parameters(), **config.optim.args)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=config.n_epoch, **config.optim.scheduler)

    best_char, best_seq = 0.0, 0.0
    for epoch in tqdm(range(config.n_epoch), leave=True):
        # print(f"----- Epoch {epoch}/{config.n_epoch} -----")
        train_loss = 0.0
        model.train()
        for src, tar in tqdm(train_loader, leave=False):
            src = src.to(device = device)
            tar = tar.to(device = device)
            
            optimizer.zero_grad()
            out, _ = model(src, tar[:, :-1])
            loss = compute_loss(out, tar[:, 1:], lang_2.pad_token)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # break
        train_loss /= len(train_loader)

        val_loss = 0.0
        acc_char, acc_seq = 0.0, 0.0
        n_char, n_seq = 0, 0
        p, t = None, None
        with torch.no_grad():
            model.eval()
            for src, tar in tqdm(val_loader, leave=False):
                src = src.to(device = device)
                tar = tar.to(device = device)
                out, _ = model(src, tar[:, :-1])
                loss = compute_loss(out, tar[:, 1:], lang_2.pad_token)
                val_loss += loss.item()
                (acc_char_item, n_char_item), (acc_seq_item, n_seq_item) = compute_accuracy(out.cpu(), tar[:, 1:].cpu(), lang_2.pad_token)
                acc_char += acc_char_item
                n_char += n_char_item
                acc_seq += acc_seq_item
                n_seq += n_seq_item
                # for i, s in enumerate(src):
                #     out = test(model, s, device, config.data.max_length)
                #     mark = (out == tar[i].cpu()).type(torch.float32)
                #     mask = (tar[i].cpu() != lang_2.pad_token).type(torch.float32)
                #     acc_seq += ((mark*mask).sum() == mask.sum()).sum()
                #     n_seq += 1
                    # print(acc_seq)
                # break
                p = torch.softmax(out, dim = -1).argmax(dim = -1)
                t = tar
                # break
            p = p.cpu().tolist()
            t = t.cpu().tolist()
            # p = [''.join([lang_2[i] for i in j]) for j in p]
            # t = [''.join([lang_2[i] for i in j]) for j in t]

            print(''.join([lang_2[i] for i in p[2]])[:30], '===', ''.join([lang_2[i] for i in t[2]])[:30])
            val_loss /= len(val_loader)
            acc_char /= n_char
            acc_seq /= n_seq
            acc_seq = acc_seq.item()

            if acc_char > best_char or acc_seq > best_seq:
                best_char = acc_char
                best_seq = acc_seq
                torch.save({
                    'model': model.state_dict(),
                    'best_char': best_char,
                    'best_seq': best_seq
                }, f'{best_char:.5f}_{best_seq:.5f}_{config.checkpoint}')
            print("\nTrain loss: ", train_loss)
            print(f"Val loss: {val_loss:.5f}, charactor acc {acc_char:.5f}, seqence acc {acc_seq:.5f}")
        scheduler.step()
        # break
        
        

if __name__ == '__main__':
    config = setup_config('config', 'config', True)
    config.model.encoder.vocab_size = len(config.vocab.en) + 3
    config.model.decoder.vocab_size = len(config.vocab.fra) + 3

    seed_everywhere(config.seed)
    train(config)
