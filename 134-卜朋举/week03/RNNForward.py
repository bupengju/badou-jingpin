import numpy as np
import torch
import torch.nn as nn


class TorchRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, bias=False, batch_first=True)

    def forward(self, x):
        return self.rnn(x)


class MyRNN(object):

    def __init__(self, ih, hh, hidden_size):
        self.ih = ih
        self.hh = hh
        self.hidden_size = hidden_size

    def forward(self, x):
        # ht = np.random.randn(self.hidden_size)
        ht = np.zeros(self.hidden_size)
        outputs = []
        for xt in x:
            ux = np.dot(xt, self.ih.T)  # seq_len x input_size --> seq_len x hidden_size
            wh = np.dot(ht, self.hh.T)  # seq_len x hidden_size --> seq_len x hidden_size
            ht = np.tanh(ux + wh)  # seq_len x hidden_size --> seq_len x hidden_size
            outputs.append(ht)

        return outputs, ht


if __name__ == '__main__':
    x = np.random.randn(15).reshape(5, 3)
    x_tensor = torch.FloatTensor(x)
    x_tensor = torch.unsqueeze(x_tensor, 0)

    input_size, hidden_size = 3, 6

    torch_model = TorchRNN(input_size, hidden_size)
    _, ht = torch_model(x_tensor)
    print(ht)

    state_dict = torch_model.state_dict()
    ih = state_dict["rnn.weight_ih_l0"].numpy()
    hh = state_dict["rnn.weight_hh_l0"].numpy()
    print(ih.shape)
    print(hh.shape)

    my_model = MyRNN(ih, hh, hidden_size)
    _, ht = my_model.forward(x)
    print(ht)
