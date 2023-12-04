import collections

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from preprocessing import normalize_string


class TranslationDataset(Dataset):

    def __init__(self, data_path, src_vocab, trg_vocab):
        super(TranslationDataset, self).__init__()
        self.data_path = data_path
        self.data = []
        self.load()

    def load(self, ):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx > 500:
                    break
                src_line, trg_line = line.strip().split("\t")[:2]
                src_words, trg_words = normalize_string(src_line), normalize_string(trg_line)
                self.data.append((src_words, trg_words))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def padding(words, max_length, pad_idx, sos_idx, eos_idx):
    words = words[:max_length-2]
    words = [sos_idx] + words + [eos_idx]
    pad_length = max_length - len(words)
    words = words + [pad_idx] * pad_length
    return words


def load_data(data_path, src_vocab, trg_vocab, max_length, batch_size, shuffle=True):
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_item, trg_item in batch:
            src_item = [src_vocab.get(item, src_vocab["<unk>"]) for item in src_item.strip().split(" ")]
            src_item = padding(src_item, max_length, src_vocab["<pad>"], src_vocab["<sos>"], src_vocab["<eos>"])
            trg_item = [trg_vocab.get(item, trg_vocab["<unk>"]) for item in trg_item.strip().split(" ")]
            trg_item = padding(trg_item, max_length + 1, trg_vocab["<pad>"], trg_vocab["<sos>"], trg_vocab["<eos>"])
            src_batch.append(src_item)
            trg_batch.append(trg_item)
        src_batch, trg_batch = torch.LongTensor(src_batch), torch.LongTensor(trg_batch)
        return src_batch, trg_batch
    
    dataset = TranslationDataset(data_path, src_vocab, trg_vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_vocab(vocab_path):
    vocab = collections.defaultdict(int)
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, char in enumerate(f):
            char = char.strip()
            vocab[char] = idx
    return vocab
