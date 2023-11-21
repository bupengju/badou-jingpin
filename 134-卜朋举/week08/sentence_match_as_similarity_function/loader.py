import collections
import json
import random

import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer

transformers.logging.set_verbosity_error()


class DataGen(Dataset):

    def __init__(self, data_path, cfg):
        self.cfg = cfg
        self.data_path = data_path
        self.schema = load_schema(cfg["schema_path"])
        # self.cfg["class_num"] = len(self.schema)
        self.tokenizer = load_vocab(cfg["vocab_path"])
        self.cfg["vocab_size"] = len(self.tokenizer.vocab)
        self.maxlength = self.cfg["maxlength"]
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
                        kwb[self.schema[label]].append(question)
                else:
                    assert isinstance(line, list)
                    self.data_type = "test"
                    question, label = line
                    label = torch.LongTensor([self.schema[label]])
                    data.append([question, label])
        return data, kwb

    def random_train_sample(self):
        standard_question_idx = list(self.kwb.keys())
        if random.random() <= self.cfg["positive_sample_rate"]:
            p = random.choice(standard_question_idx)
            if len(self.kwb[p]) < 2:
                return self.random_train_sample()
            else:
                s1, s2 = random.sample(self.kwb[p], 2)
                seq = self.sentence_to_seq(s1, s2)
                seq = torch.LongTensor(seq)
                return [seq, torch.LongTensor([1])]
        else:
            p, n = random.sample(standard_question_idx, 2)
            s1 = random.choice(self.kwb[p])
            s2 = random.choice(self.kwb[n])
            seq = self.sentence_to_seq(s1, s2)
            seq = torch.LongTensor(seq)
            return [seq, torch.LongTensor([0])]

    def sentence_to_seq(self, sentence1, sentence2):
        seq = self.tokenizer.encode(
            sentence1, sentence2, truncation="longest_first",
            max_length=self.maxlength, padding="max_length")
        return seq

    def __getitem__(self, item):
        if self.data_type == "train":
            return self.random_train_sample()
        else:
            assert self.data_type == "test", self.data_type
            return self.data[item]

    def __len__(self):
        if self.data_type == "train":
            return self.cfg["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)


def load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.loads(f.read())
    return schema


def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)

    return tokenizer


def load_data(data_path, config, shuffle):
    dataset = DataGen(data_path, config)
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return data_loader
