import json

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx + 1
    vocab["[PAD]"] = 0

    return vocab


def sentence_to_seq(sentence, vocab):
    input_id = []
    for text in sentence:
        input_id.append(vocab.get(text, vocab["[UNK]"]))
    return input_id


class DataGen(Dataset):

    def __init__(self, data_path, schema_path, vocab, max_lens):
        super(DataGen, self).__init__()
        self.data_path = data_path
        self.schema_path = schema_path
        self.vocab = vocab
        self.max_lens = max_lens
        self.sentences = []
        self.schema = self.load_schema()
        self.data = self.load()

    def load_schema(self):
        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return schema

    def load(self):
        data = []
        with open(self.data_path, encoding="utf-8") as f:
            sentences = f.read().split("\n\n")
            for sentence in sentences:
                chars = []
                labels = []
                for line in sentence.split("\n"):
                    if line == "":
                        continue
                    char, label = line.split()
                    chars.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(chars))
                # print(chars)
                input_ids = sentence_to_seq(chars, self.vocab)
                input_ids = self.padding(input_ids)
                labels = self.padding(labels, -1)
                data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return data

    def padding(self, seq, pad_token=0):
        seq = seq[:self.max_lens]
        seq += [pad_token] * (self.max_lens - len(seq))
        return seq

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def load_dataset(data_path, schema_path, vocab, max_lens, batch_size, shuffle):
    return DataLoader(DataGen(data_path, schema_path, vocab, max_lens), batch_size, shuffle)
