import collections
import json
import random
from pathlib import Path

import jieba
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DataGen(Dataset):

    def __init__(self, data_path, cfg):
        self.cfg = cfg
        self.data_path = data_path
        self.schema = load_schema(cfg["schema_path"])
        self.cfg["class_num"] = len(self.schema)
        self.vocab = load_vocab(cfg["vocab_path"])
        self.cfg["vocab_size"] = len(self.vocab)
        self.data_type = None
        self.data, self.kwb = self.load()

    def load(self):
        data = []
        kwb = collections.defaultdict(list)
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        seq = self.sentence_to_seq(question)
                        seq = torch.LongTensor(seq)
                        kwb[self.schema[label]].append(seq)
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    seq = self.sentence_to_seq(question)
                    seq = torch.LongTensor(seq)
                    label = torch.LongTensor([self.schema[label]])
                    data.append([seq, label])
        return data, kwb

    def sentence_to_seq(self, sentence):
        seq = []
        if Path(self.cfg["vocab_path"]).stem == "words":
            for word in jieba.lcut(sentence):
                seq.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in sentence:
                seq.append(self.vocab.get(char, self.vocab["[UNK]"]))

        seq = self.padding(seq)
        return seq

    def padding(self, seq):
        seq = seq[:self.cfg["maxlength"]]
        seq += [0] * (self.cfg["maxlength"] - len(seq))
        return seq

    def __getitem__(self, item):
        if self.data_type == "train":
            return self.random_sample()
        else:
            return self.data[item]

    def __len__(self):
        if self.data_type == "train":
            return self.cfg["epoch_data_size"]
        else:
            assert self.data_type.lower() == "test"
            return len(self.data)

    def random_sample(self):
        sq_idx = list(self.kwb.keys())
        p, n = random.sample(sq_idx, 2)
        if len(self.kwb[p]) == 1:
            s1 = s2 = self.kwb[p][0]
        else:
            s1, s2 = random.sample(self.kwb[p], 2)

        s3 = random.choice(self.kwb[n])
        return [s1, s2, s3]


def load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.loads(f.read())
    return schema


def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx + 1

    return vocab


def load_data(data_path, config, shuffle):
    dataset = DataGen(data_path, config)
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return data_loader
