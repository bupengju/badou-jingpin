from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class NNLM(nn.Module):

    def __init__(self, input_size, vocab):
        super(NNLM, self).__init__()
        self.embed = nn.Embedding(len(vocab) + 1, input_size, max_norm=1.)
        self.rnn = nn.RNN(input_size, input_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(input_size, len(vocab) + 1)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embed(x)  # n x seq_len x embed_size
        x, _ = self.rnn(x)  # n x seq_len x embed_size
        x = x[:, -1, :]  # n x embed_size
        if self.training:
            x = self.dropout(x)
        out = self.fc(x)  # n x (vocab_size + 1)
        if y is not None:
            return self.loss(out, y)
        else:
            return torch.softmax(out, dim=-1)


def build_vocab(path):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            char = line[:-1]
            # print(line, "--", char, "--", len(char), "--", idx+1)
            vocab[char] = idx + 1
        vocab["\n"] = 1

    return vocab


def build_sample(corpus, vocab, window_size):
    start = np.random.randint(0, len(vocab) - window_size - 1)
    end = start + window_size
    window = corpus[start: end]
    target = corpus[end]
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]
    y = vocab[target]
    return x, y


def build_dataset(sample_length, corpus, vocab, window_size):
    xs, ys = [], []
    for _ in range(sample_length):
        x, y = build_sample(corpus, vocab, window_size)
        xs.append(x)
        ys.append(y)
    return torch.LongTensor(xs), torch.LongTensor(ys)


def train(corpus_path, vocab_path, output):
    epoch = 20
    lr = 0.01
    batch_size = 32
    embed_size = 64
    train_sample_lens = 2000
    win_size = 6
    vocab = build_vocab(vocab_path)
    corpus = open(corpus_path, "r", encoding="utf-8").read()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NNLM(embed_size, vocab)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    for e in range(epoch):
        net.train()
        watch_loss = []
        for _ in range(train_sample_lens // batch_size):
            x, y = build_dataset(batch_size, corpus, vocab, win_size)
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            loss = net(x, y)
            watch_loss.append(loss.item())
            loss.backward()
            optim.step()
        print("Epoch: %d, loss: %.6f" % (e + 1, np.mean(watch_loss).item()))

    cate = Path(corpus_path).stem
    weight_path = Path(output).joinpath(cate+".pth").as_posix()
    with open(weight_path, "wb") as f:
        torch.save(net.state_dict(), f)


def main(corpus_root, vocab_path, output):
    for f in sorted(Path(corpus_root).glob("*.txt")):
        print(f.stem)
        train(f.as_posix(), vocab_path, output)


if __name__ == '__main__':
    main("./data/corpus", "./data/vocab.txt", "./out")
