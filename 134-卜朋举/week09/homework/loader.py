import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer


class DataGen(Dataset):

    def __init__(self, data_path, schema, max_length):
        super(DataGen, self).__init__()
        self.data_path = data_path
        self.schema = schema
        self.max_length = max_length
        self.data = []
        self.load()

    def load(self):
        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(line) > self.max_length:
                    for i in range(len(line) // self.max_length):
                        data.append(self.process_sentence(
                            line[i * self.max_length:(i + 1) * self.max_length]))
                else:
                    data.append(self.process_sentence(line))
        self.data = data

    def process_sentence(self, sentence):
        sentence_without_sign = []
        label = []

        for char in sentence:
            if char in self.schema:
                label.append(self.schema[char])
            else:
                sentence_without_sign.append(char)
                label.append(0)
        return sentence_without_sign, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def load_data(data_path, vocab_path, schema_path, batch_size, max_length, shuffle=True):
    """
    Load data from data_path and return a DataLoader object.

    :param data_path: str, path to the data file.
    :param vocab_path: str, path to the vocabulary file.
    :param batch_size: int, batch size for DataLoader.
    :param max_length: int, maximum length of each sequence.
    :param shuffle: bool, whether to shuffle the data or not. Default is True.

    :return: DataLoader object.
    """
    tokenizer = BertTokenizer.from_pretrained(vocab_path)

    def collate_fn(batch):
        data = [b[0] for b in batch]
        label = [b[1] for b in batch]

        out = tokenizer.batch_encode_plus(
            data, max_length=max_length, padding="max_length", truncation=True,
            is_split_into_words=True, return_tensors="pt"
        )

        for i in range(len(label)):
            label[i] = [0] + label[i] + [0] * max_length
            label[i] = label[i][:max_length]
        label = torch.LongTensor(label)
        return out, label

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    dataset = DataGen(data_path, schema, max_length)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader
