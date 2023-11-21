import torch
import torch.nn as nn


class SentenceEncoder(nn.Module):

    def __init__(self, cfg):
        super(SentenceEncoder, self).__init__()
        vocab_size = cfg["vocab_size"]
        hidden_size = cfg["hidden_size"] + 1
        self.embed = nn.Embedding(vocab_size, hidden_size, max_norm=1., padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        outputs, (ht, ct) = self.lstm(x)
        return ht.squeeze()


class SimNetwork(nn.Module):

    def __init__(self, cfg):
        super(SimNetwork, self).__init__()
        self.encoder = SentenceEncoder(cfg)
        self.loss = nn.CosineEmbeddingLoss()

    def cosine_distance(self, x1, x2):
        cosine_sim = torch.sum(torch.mul(x1, x2), dim=-1)
        return 1 - cosine_sim

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])

    def forward(self, x1, x2=None, x3=None):
        if x2 is not None and x3 is not None:
            x1 = self.encoder(x1)
            x2 = self.encoder(x2)
            x3 = self.encoder(x3)
            return self.cosine_triplet_loss(x1, x2, x3)
        else:
            return self.encoder(x1)


def choose_optim(parameters, cfg):
    name = cfg["optim"].lower()
    lr = cfg["lr"]
    if name == "adam":
        optim = torch.optim.Adam(parameters, lr=lr)
    else:
        optim = torch.optim.SGD(parameters, lr=lr)
    return optim
