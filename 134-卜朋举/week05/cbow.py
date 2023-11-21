import torch.nn as nn
import torch


class CBOW(nn.Module):

    def __init__(self, vocab_size, embed_size, window_length):
        super(CBOW, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, max_norm=1)
        self.pool = nn.AvgPool1d(window_length)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embed(x)  # batch_size x seq_len x vocab_size --> batch_size x seq_len x embed_size
        x = self.pool(torch.transpose(x, 1, 2)).squeeze()  # batch_size x seq_len x embed_size --> batch_size x embed_size
        out = self.linear(x)  # batch_size x embed_size --> batch_size x vocab_size

        if y is not None:
            return self.loss(out, y)
        else:
            return out


if __name__ == '__main__':
    vocab_size = 10
    embed_size = 16
    window_size = 4

    net = CBOW(vocab_size, embed_size, window_size)
    inputs = torch.LongTensor([2, 3, 5, 8])
    inputs = torch.unsqueeze(inputs, 0)
    pred = net(inputs)
    print(pred)

    print(net.state_dict()["embed.weight"])
