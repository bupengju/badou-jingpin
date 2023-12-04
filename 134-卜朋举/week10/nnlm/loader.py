import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CorpusDataset(Dataset):
    def __init__(self, path, vocab, win_sz, sample_length):
        self.path = path
        self.vocab = vocab
        self.win_sz = win_sz
        self.sample_length = sample_length
        self.data = []
        self.load()

    def load(self):
        sentences = ""
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                sentences += line
        
        for _ in range(self.sample_length):
            x, y = build_sample(sentences, self.win_sz)
            self.data.append((x, y))

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def load_data(path, vocab, win_sz, sample_length, batch_size, shuffle=True):
    def collate_fn(batch):
        x, y = zip(*batch)
        batch_x, batch_y = [], []
        for i in range(len(x)):
            batch_x.append([vocab.get(word, vocab["<unk>"]) for word in x[i]])
            batch_y.append(vocab.get(y[i], vocab["<unk>"]))
        x = torch.LongTensor(batch_x)
        y = torch.LongTensor(batch_y)
        return x, y
    dataset = CorpusDataset(path, vocab, win_sz, sample_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader


def build_sample(sentences, win_sz):
    i = random.randint(0, len(sentences) - win_sz - 1)
    inputs = sentences[i:i + win_sz]
    targets = sentences[i + win_sz]
    return inputs, targets


def build_vocab(path):
    vocab = {"<pad>": 0, "<unk>": 1} 
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            vocab[line] = idx + 2
    return vocab
