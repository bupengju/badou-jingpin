import torch
import torch.nn as nn


class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size, max_length, hidden_size):
        super(LanguageModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, max_norm=1)
        self.inner_projection_layer = nn.Linear(embed_size * max_length, hidden_size)
        self.outer_projection_layer = nn.Linear(hidden_size, hidden_size)
        self.x_projection_layer = nn.Linear(embed_size * max_length, hidden_size)
        self.projection_layer = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # batch_size x max_len x vocab_size --> batch_size x max_len x embed_size
        x = self.embed(x)
        # batch_size x max_len x embed_size --> batch_size x hidden_size
        x_proj = self.x_projection_layer(x.view(x.shape[0], -1))
        # batch_size x max_len x embed_size --> batch_size x hidden_size
        x = self.inner_projection_layer(x.view(x.shape[0], -1))
        x = torch.tanh(x)
        # batch_size x hidden_size --> batch_size x hidden_size
        x = self.outer_projection_layer(x)
        # batch_size x hidden_size --> batch_size x hidden_size
        x = x_proj + x
        out = self.projection_layer(x)

        if y is not None:
            return self.loss(out, y)
        else:
            return out


if __name__ == '__main__':
    vocab_size = 10
    embed_size = 16
    max_len = 4
    hidden_size = vocab_size

    context = torch.LongTensor([[1, 2, 3, 4]])

    net = LanguageModel(vocab_size, embed_size, max_len, hidden_size)
    pred = net(context)
    print(pred)

    print(net.state_dict()["embed.weight"])
