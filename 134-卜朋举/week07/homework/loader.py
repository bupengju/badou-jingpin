import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class DataGen(Dataset):

    def __init__(self, data_path, cfg):
        super(DataGen, self).__init__()
        self.cfg = cfg
        self.data_path = data_path
        self.idx_to_label = {
            0: "家居", 1: "房产", 2: "股票", 3: "社会", 4: "文化",
            5: "国际", 6: "教育", 7: "军事", 8: "彩票", 9: "旅游",
            10: "体育", 11: "科技", 12: "汽车", 13: "健康",
            14: "娱乐", 15: "财经", 16: "时尚", 17: "游戏"
        }
        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        # self.cfg["class_num"] = len(self.label_to_idx)
        self.cfg["class_num"] = 2
        if cfg["model_type"].lower() == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(cfg["pretrain_model_path"])
        self.vocab = load_vocab(cfg["vocab_path"])
        self.cfg["vocab_size"] = len(self.vocab)
        self.data = self.load()

    def load(self):
        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                tag = line["tag"]
                # label = self.label_to_idx[tag]
                label = int(tag)
                title = line["title"]
                if self.cfg["model_type"].lower() == "bert":
                    seq = self.tokenizer.encode(title, max_length=self.cfg["max_length"], padding='max_length',
                                                truncation=True)
                else:
                    seq = self.sentence_to_seq(title)
                seq = torch.LongTensor(seq)
                label = torch.LongTensor([label])
                data.append([seq, label])

        return data

    def sentence_to_seq(self, sentence):
        seq = []
        for char in sentence:
            seq.append(self.vocab.get(char, self.vocab["[UNK]"]))
        seq = self.padding(seq)
        return seq

    def padding(self, seq):
        seq = seq[:self.cfg["max_length"]]
        seq += [0] * (self.cfg["max_length"] - len(seq))
        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx + 1

    return vocab


def load_data(data_path, cfg, shuffle):
    dataset = DataGen(data_path, cfg)
    return DataLoader(dataset, cfg["batch_size"], shuffle=shuffle)


def train_test_split(cfg):
    data_path = cfg["data_path"]
    train_path = cfg["train_data_path"]
    test_path = cfg["test_data_path"]
    train_ratio = cfg["train_ratio"]

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as data_file, \
            open(train_path, "w", encoding="utf-8") as train_file, \
            open(test_path, "w", encoding="utf-8") as test_file:

        for idx, line in enumerate(data_file):
            if idx == 0:
                continue

            line = {"tag": line.strip()[0], "title": line.strip()[2:]}
            line = json.dumps(line, ensure_ascii=False)

            rand_num = random.random()
            if rand_num < train_ratio:
                train_file.write(line+"\n")
            else:
                test_file.write(line+"\n")


if __name__ == '__main__':
    import collections
    data_path = r"../data/homework/train_comment.json"
    class_count = collections.defaultdict(int)
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line.strip())
            class_count[line["tag"]] += 1
    print(class_count)
