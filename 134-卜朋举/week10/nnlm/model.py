import torch.nn as nn
import torch


class NNLM(nn.Module):

    def __init__(self, vocab_size, hidden_dim, num_layers, dropout):
        super(NNLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.classify = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        _, x = self.rnn(x)
        x = self.dropout(x)
        x = x[-1, :]
        x = self.classify(x)
        # print(x.shape, y.shape)
        if y is not None:
            loss = self.loss(x, y)
            return loss
        else:
            return x
