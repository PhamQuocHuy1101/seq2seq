import torch
from torch.utils.data import Dataset

from dataset import utils

class Vocab:
    def __init__(self, raw):
        raw = ['<pad>', '<sos>', '<eos>'] + list(raw)
        self.char2token = {c:t for t, c in enumerate(raw)}
        self.token2char = {t:c for t, c in enumerate(raw)}
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2

    def __getitem__(self, idx):
        return self.char2token.get(idx, None) if type(idx) == str else self.token2char.get(idx, None)

class TrainingData(Dataset):
    def __init__(self, src_vocab, tar_vocab, sentences, max_length):
        super(TrainingData, self).__init__()
        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab
        self.max_length = max_length
        self.lines = [self.__build_sentence(line) for line in sentences]

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        return self.lines[index]

    def __build_sentence(self, raw):
        src, tar = raw.split('\t')
        src = [self.src_vocab[s] for s in list(src)] + [self.src_vocab.end_token]
        tar = [self.tar_vocab.start_token] + [self.tar_vocab[s] for s in list(tar)] + [self.tar_vocab.end_token]
        return (utils.normalize_sentence(src, self.src_vocab.pad_token, self.max_length),
                utils.normalize_sentence(tar, self.tar_vocab.pad_token, self.max_length))

