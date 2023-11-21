import jieba
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            vocab[line.strip()] = i + 1  # 0 for padding
    vocab["<PAD>"] = -100
    vocab["<UNK>"] = len(vocab)
    return vocab


def sentence_to_seq(sentence, vocab):
    seq = [vocab.get(char, vocab["<UNK>"]) for char in sentence]
    return seq


def sentence_to_label(sentence):
    words = jieba.lcut(sentence)
    pointer = 0
    label = [0] * len(sentence)
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label


class DataGen(Dataset):

    def __init__(self, vocab, corpus_path, max_length):
        super(DataGen, self).__init__()
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.vocab = vocab
        self.data = []
        self.load()

    def load(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 1000:
                    break

                seq = sentence_to_seq(line, self.vocab)
                label = sentence_to_label(line)
                seq, label = self.padding(seq, label)
                seq = torch.LongTensor(seq)
                label = torch.LongTensor(label)
                self.data.append([seq, label])

    def padding(self, seq, label):
        seq = seq[:self.max_length]
        seq += [0] * (self.max_length - len(seq))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))  # 不参与损失计算
        return seq, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def load_dataset(vocab, corpus_path, max_length, batch_size, shuffle):
    return DataLoader(DataGen(vocab, corpus_path, max_length), batch_size, shuffle)
