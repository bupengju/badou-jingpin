import json
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
        self.data = self.load()

    def load(self):
        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                if isinstance(line, dict):
                    questions = line["questions"]
                    target = line["target"]
                    label = torch.LongTensor([self.schema[target]])
                    for question in questions:
                        seq = self.sentence_to_seq(question)
                        seq = torch.LongTensor(seq)
                        data.append([seq, label])
                else:
                    question = line[0]
                    seq = self.sentence_to_seq(question)
                    seq = torch.LongTensor(seq)
                    label = torch.LongTensor([self.schema[line[1]]])
                    data.append([seq, label])
        return data

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
        return self.data[item]

    def __len__(self):
        return len(self.data)


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
