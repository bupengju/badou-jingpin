import json
from pathlib import Path

import jieba
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class TorchModel(nn.Module):

    def __init__(self, input_size, hidden_size, vocab_size, num_hidden_layer):
        super(TorchModel, self).__init__()
        self.embed = nn.Embedding(vocab_size + 1, input_size, max_norm=1)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_hidden_layer)
        self.classifier = nn.Linear(hidden_size, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embed(x)  # batch_size x seq_length --> batch_size x seq_length x embed_size
        x, _ = self.rnn(x)  # batch_size x seq_length x embed_size --> batch_size x seq_length x hidden_size
        x = self.classifier(x)  # batch_size x seq_length x hidden_size --> batch_size x seq_length x 2

        if y is not None:
            return self.loss(x.view(-1, 2), y.view(-1))  # batch_size x seq_length x 2 --> batch_size * seq_length x 2
        else:
            return x


class TorchDataset(Dataset):

    def __init__(self, corpus_path, vocab, max_length):
        super(TorchDataset, self).__init__()
        self.corpus_path = corpus_path
        self.vocab = vocab
        self.max_length = max_length
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                seq = self.sentence_to_sequence(line.strip(), self.vocab)
                label = self.sequence_to_label(line.strip())
                seq, label = self.padding(seq, label)
                seq = torch.LongTensor(seq)
                label = torch.LongTensor(label)
                data.append([seq, label])

                if idx > 5000:
                    break
        return data

    def sentence_to_sequence(self, sentence, vocab):
        sequence = [vocab.get(c, vocab["unk"]) for c in sentence]
        return sequence

    def sequence_to_label(self, sentence):
        label = [0] * len(sentence)
        words = jieba.lcut(sentence)
        point = 0
        for word in words:
            point += len(word)
            label[point - 1] = 1
        return label

    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return sequence, label

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def build_vocab(chars_path):
    vocab = {}
    with open(chars_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx + 1
    vocab["unk"] = len(vocab) + 1
    return vocab


def main(chars_path, corpus_path, output_root):
    epoch = 20
    batch_size = 32
    lr = 0.01
    input_size = 64
    hidden_size = 128
    num_rnn_layer = 2
    max_length = 20
    output_root = Path(output_root)

    vocab = build_vocab(chars_path)
    vocab_path = output_root.joinpath("vocab.json").as_posix()
    json.dump(vocab, open(vocab_path, "w", encoding="utf-8"))
    train_dataset = TorchDataset(corpus_path, vocab, max_length)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    net = TorchModel(input_size, hidden_size, len(vocab), num_rnn_layer)

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    for e in range(epoch):
        net.train()
        watch_loss = []
        for x, y in train_dataloader:
            optim.zero_grad()
            loss = net(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("Epoch: %03d, loss: %.6f" % (e + 1, np.mean(watch_loss).item()))
    torch.save(net.state_dict(), output_root.joinpath("model.pth").as_posix())


def predict(model_path, vocab_path, input_sentence):
    input_size = 64
    hidden_size = 128
    num_rnn_layer = 2
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    net = TorchModel(input_size, hidden_size, len(vocab), num_rnn_layer)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    for cc in input_sentence:
        x = [vocab.get(c, vocab["unk"]) for c in cc]
        with torch.no_grad():
            x = torch.LongTensor(x)
            x = torch.unsqueeze(x, 0)
            x = net(x)
            pred = torch.squeeze(torch.argmax(x, dim=-1))
            for idx, p in enumerate(pred):
                if pred[idx] == 1:
                    print(cc[idx], end=" ")
                else:
                    print(cc[idx], end="")
            print()


if __name__ == '__main__':
    main("./data/chars.txt", "./data/corpus.txt", "./out")
    input_sentences = [
        "同时国内有望出台新汽车刺激方案",
        "沪胶后市有望延续强势",
        "经过两个交易日的强势调整后",
        "昨日上海天然橡胶期货价格再度大幅上扬"
    ]
    predict("./out/model.pth", "./out/vocab.json", input_sentences)
