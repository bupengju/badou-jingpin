import torch
import torch.nn as nn


class GetFirst(nn.Module):

    def __init__(self):
        super(GetFirst, self).__init__()

    def forward(self, x):
        return x[0]


class SentenceMatchNetwork(nn.Module):

    def __init__(self, cfg):
        super(SentenceMatchNetwork, self).__init__()
        vocab_size = cfg["vocab_size"]
        hidden_size = cfg["hidden_size"]
        max_length = cfg["maxlength"]
        # class_num = cfg["class_num"]
        self.embed = nn.Embedding(vocab_size, hidden_size, max_norm=1.)
        self.encoder = nn.Sequential(
            nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True),
            GetFirst(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.pooling = nn.AvgPool1d(max_length)
        self.fc = nn.Linear(hidden_size, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embed(x)
        x = self.encoder(x)
        x = self.pooling(x.transpose(1, 2)).squeeze()
        out = self.fc(x)

        if y is not None:
            return self.loss(out, y.squeeze())
        else:
            return torch.softmax(out, dim=-1)


def choose_optim(parameters, cfg):
    name = cfg["optim"].lower()
    lr = cfg["lr"]
    if name == "adam":
        optim = torch.optim.Adam(parameters, lr=lr)
    else:
        optim = torch.optim.SGD(parameters, lr=lr)
    return optim
