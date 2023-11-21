import torch
import torch.nn as nn


class TorchModel(nn.Module):

    def __init__(self, cfg):
        super(TorchModel, self).__init__()
        vocab_size = cfg["vocab_size"]
        hidden_size = cfg["hidden_size"]
        max_length = cfg["maxlength"]
        class_num = cfg["class_num"]
        self.embed = nn.Embedding(vocab_size, hidden_size, max_norm=1.)
        self.pooling = nn.AvgPool1d(max_length)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, class_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embed(x)
        x = self.pooling(x.transpose(1, 2)).squeeze()
        x = self.relu(self.linear(x))
        out = self.fc(x)

        if y is not None:
            return self.loss(out, y.squeeze())
        else:
            return out


def choose_optim(parameters, cfg):
    name = cfg["optim"].lower()
    lr = cfg["lr"]
    if name == "adam":
        optim = torch.optim.Adam(parameters, lr=lr)
    else:
        optim = torch.optim.SGD(parameters, lr=lr)
    return optim
