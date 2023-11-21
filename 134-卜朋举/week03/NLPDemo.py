import json
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class LR(nn.Module):

    def __init__(self, sentence_lens, vocab_lens, embed_size):
        super(LR, self).__init__()
        self.embed = nn.Embedding(vocab_lens, embed_size, max_norm=1.0)
        self.pool = nn.AvgPool1d(sentence_lens)
        self.linear = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, x, y=None):
        x = self.embed(x)  # batch_size x seq_len --> batch_size x seq_len x embed_size
        x = torch.transpose(x, 1, 2)  # batch_size x embed_size x seq_len torch pooling 在最后一个轴执行, 因此需要转置
        x = self.pool(x)  # batch_size x embed_size
        x = torch.squeeze(x)
        x = self.linear(x)  # batch_size x embed_size --> batch_size x 1
        x = self.sigmoid(x)  # batch_size x 1

        if y is not None:
            return self.loss(x, y)
        else:
            return x


def build_vocab():
    s = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"UNK": 0}
    for idx, c in enumerate(s):
        vocab[c] = idx + 1
    return vocab


def build_sample(vocab, sentence_lens):
    # np.random.normal()
    p = np.random.normal(0, 1, size=len(vocab))
    p = np.exp(p) / np.sum(np.exp(p))
    # print(p)
    x = np.random.choice(list(vocab.keys()), sentence_lens, replace=False, p=p)
    y = 0
    # if set("abc") & set(x):
    if np.argmax(p) > 6:
        y = 1
    x = [vocab.get(i, vocab["UNK"]) for i in x]
    return x, y


def build_dataset(vocab, sentence_lens, n_sample):
    np.random.seed(2023)
    x, y = [], []
    for _ in range(n_sample):
        xi, yi = build_sample(vocab, sentence_lens)
        x.append(xi)
        y.append(yi)

    return torch.LongTensor(x), torch.FloatTensor(y)


def evaluate(model, vocab, sentence_lens, n_sample):
    model.eval()
    x, y = build_dataset(vocab, sentence_lens, n_sample)
    # print("positive count: %d, negative count: %d" % (y.numpy().sum(), (1 - y.numpy()).sum()))
    right, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_hat, y_true in zip(y_pred, y):
            if float(y_hat) < 0.5 and y_true == 0:
                right += 1
            elif float(y_hat) >= 0.5 and y_true == 1:
                right += 1
            else:
                wrong += 1
    # print("right pred: %d, acc: %f" % (right, right / (right + wrong)))
    return right / (right + wrong)


def predict(model_path, vocab_path, sentences, embed_size):
    vocab = json.load(open(vocab_path, "r"))
    model = LR(len(sentences[0]), len(vocab), embed_size)
    model.load_state_dict(torch.load(model_path))

    x = []
    for sentence in sentences:
        x.append([vocab.get(c, vocab["UNK"]) for c in sentence])
    x = torch.LongTensor(x)

    model.eval()
    with torch.no_grad():
        pred = model.forward(x)

    for idx, sentence in enumerate(sentences):
        print("sentence: %s, pred prob: %f, pred cate: %f" % (sentence, pred[idx], round(pred[idx].item())))


def main():
    epoch = 200
    batch_size = 10
    lr = 0.01

    n_sample_train = 300
    sentence_lens = 6
    embed_size = 5

    vocab = build_vocab()

    dataset = build_dataset(vocab, sentence_lens, n_sample_train)

    model = LR(sentence_lens, len(vocab), embed_size)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    for e in range(epoch):
        model.train()
        watch_loss = []
        for i in range(0, len(dataset), batch_size):
            optim.zero_grad()
            x = dataset[0][i:i+batch_size]
            y = dataset[1][i:i+batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("Epoch: %d, loss: %f" % (e + 1, np.mean(watch_loss).item()))
        acc = evaluate(model, vocab, sentence_lens, 60)
        log.append([acc, np.mean(watch_loss).item()])

    # plt.figure()
    plt.plot([i for i in range(len(log))], [row[0] for row in log], label="acc")
    plt.plot([i for i in range(len(log))], [row[1] for row in log], label="loss")
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "./out/model.pth")

    json.dump(vocab, open("./out/vocab.json", "w"))


if __name__ == '__main__':
    # vocab = build_vocab()
    # print(build_dataset(vocab, 6, 10)[1].sum())

    main()

    test_sentences = ["abcedf", "xhlfnc", "abcddd", "lllkkk", "mmmmnn"]
    predict("./out/model.pth", "./out/vocab.json", test_sentences, 5)
