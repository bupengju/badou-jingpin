import json

import jieba
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertModel


# 在中文分词任务上尝试使用bert预训练模型。


class TorchModel(nn.Module):

    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained("./data/bert-base-chinese", return_dict=False)
        self.fc = nn.Linear(input_size, 2)
        self.act = nn.Softmax()
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):
        seq_output, pooler_output = self.bert(x)
        out = self.act(self.fc(seq_output))
        if y is not None:
            return self.loss(out.view(-1, 2), y.view(-1))
        else:
            return out


class TorchDataset(Dataset):

    def __init__(self, vocab_path, corpus_path, max_len, cut_line_num=5000):
        super(TorchDataset, self).__init__()
        self.vocab_path = vocab_path
        self.corpus_path = corpus_path
        self.max_len = max_len
        self.cut_line_num = cut_line_num
        self.data = self.load_data()

    def load_vocab(self, ):
        vocab = {"<UNK>": 0}
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                vocab[line.strip()] = idx + 1

        return vocab

    def load_data(self):
        vocab = self.load_vocab()
        result = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                sentence = line.strip()
                seq = self.sentence_to_seq(sentence, vocab)
                label = self.sentence_to_label(sentence)
                pad_seq, pad_label = self.padding(seq, label)
                pad_seq = torch.LongTensor(pad_seq)
                pad_label = torch.LongTensor(pad_label)
                result.append([pad_seq, pad_label])

                if idx >= self.cut_line_num:
                    break
        return result

    def sentence_to_seq(self, sentence, vocab):
        seq = [vocab.get(word, vocab["<UNK>"]) for word in sentence]
        return seq

    def sentence_to_label(self, sentence):
        cut = [0] * len(sentence)
        words = jieba.lcut(sentence)
        index = 0
        for word in words:
            index += len(word)
            cut[index - 1] = 1
        return cut

    def padding(self, seq, label):
        seq = seq[:self.max_len]
        seq += [0] * (self.max_len - len(seq))

        label = label[:self.max_len]
        label += [-100] * (self.max_len - len(label))

        return seq, label

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def train(vocab_path, corpus_path):
    epoch = 20
    lr = 0.01
    batch_size = 32
    input_size = 768
    max_len = 5
    train_sample_len = 2000

    dataset = TorchDataset(vocab_path, corpus_path, max_len, train_sample_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset.load_vocab()
    json.dump(dataset.load_vocab(), open("./out/vocab.json", "w"))

    net = TorchModel(input_size)

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    for e in range(epoch):
        net.train()
        watch_loss = []
        for x, y in dataloader:
            optim.zero_grad()
            loss = net(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("Epoch: %d, loss: %.4f" % (e + 1, np.mean(watch_loss).item()))
    torch.save(net.state_dict(), "./out/model.pth")


def predict(weight_path, vocab_path, input_string):
    input_size = 768
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))

    for sentence in input_string:
        sentence = sentence.strip()
        seq = [vocab.get(char, vocab["<UNK>"]) for char in sentence]
        seq_tensor = torch.LongTensor([seq])

        with torch.no_grad():
            net = TorchModel(input_size)
            net.eval()
            net.load_state_dict(torch.load(weight_path))
            y_pred = net(seq_tensor)
            y_pred = torch.argmax(y_pred, dim=-1).squeeze()

            for idx, p in enumerate(y_pred.numpy()):
                if p == 1:
                    print(sentence[idx], end=" ")
                else:
                    print(sentence[idx], end="")
            print()


def main():
    # train("./data/chars.txt", "./data/corpus.txt")
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    predict("./out/model.pth", "./out/vocab.json", input_strings)


if __name__ == '__main__':
    main()
