import json

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel


class TorchModel(nn.Module):

    def __init__(self, input_dim):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained("./data/bert-base-chinese", return_dict=False)
        self.fc = nn.Linear(input_dim, 3)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        seq_output, pooler_output = self.bert(x)
        x = self.fc(pooler_output)
        out = self.softmax(x)

        if y is not None:
            return self.loss(out, y)
        else:
            return out


def build_vocab():
    strings = "abcdefghijklmnopqrstuvwxyz"
    vocab_dict = {}
    for idx, char in enumerate(strings):
        vocab_dict[char] = idx + 1
    vocab_dict["UNK"] = len(vocab_dict) + 1
    return vocab_dict


def build_sample(vocab, seq_len):
    x = np.random.choice(list(vocab.keys()), seq_len, replace=False)
    if set("abc") & set(x):
        y = 0
    elif set("xyz") & set(x):
        y = 1
    else:
        y = 2
    x = [vocab.get(c, vocab["UNK"]) for c in x]
    return x, y


def build_dataset(vocab, seq_len, sample_len):
    np.random.seed(2023)
    xs, ys = [], []
    for _ in range(sample_len):
        x, y = build_sample(vocab, seq_len)
        xs.append(x)
        ys.append(y)
    return torch.LongTensor(xs), torch.LongTensor(ys)


def train():
    epoch = 10
    batch_size = 8
    lr = 0.01
    input_size = 768
    seq_len = 6
    sample_len = 200

    vocab = build_vocab()

    net = TorchModel(input_size)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    for e in range(epoch):
        watch_loss = []
        net.train()
        for _ in range(sample_len // batch_size):
            x, y = build_dataset(vocab, seq_len, batch_size)
            optim.zero_grad()
            loss = net(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("Epoch: %d, loss: %.4f" % (e + 1, np.mean(watch_loss).item()), end=", ")
        evaluate(net, vocab, seq_len)

    torch.save(net.state_dict(), "./out/model.pth")
    json.dump(vocab, open("./out/vocab.json", "w", encoding="utf-8"))


def evaluate(model, vocab, seq_len):
    model.eval()
    x, y = build_dataset(vocab, seq_len, 10)

    with torch.no_grad():
        y_pred = model(x)
        acc = torch.sum(torch.argmax(y_pred) == y) / y_pred.shape[0]
    print("acc: %.6f" % float(acc))
    return acc


def predict(model_path, vocab_path):
    input_size = 768
    seq_len = 6
    sample_len = 5
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    idx_2_ch = {v: k for k, v in vocab.items()}
    x_test, y_test = build_dataset(vocab, seq_len, sample_len)

    net = TorchModel(input_size)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    for x, y in zip(x_test, y_test):
        # print(x, y)
        y_pred = net(x.unsqueeze(0))
        y_pred = torch.argmax(y_pred).numpy()
        strings = "".join([idx_2_ch[i] for i in x.squeeze().numpy()])
        print("input: %s, target: %d, predict: %d" % (strings, y.numpy(), y_pred))


def main():
    train()
    predict("./out/model.pth", "./out/vocab.json")


if __name__ == '__main__':
    main()
