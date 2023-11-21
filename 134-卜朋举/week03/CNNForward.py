import torch
import torch.nn as nn
import numpy as np


class TorchCNN(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size):
        super(TorchCNN, self).__init__()
        self.cnn = nn.Conv2d(in_channel, out_channel, kernel_size, bias=False)

    def forward(self, x):
        return self.cnn(x)


class MyCNN(object):

    def __init__(self, w, in_height, in_width, kernel_size):
        self.w = w
        self.in_height = in_height
        self.in_width = in_width
        self.kernel_size = kernel_size

    def forward(self, x):
        feature_maps = []
        for kw in self.w:
            kw = np.squeeze(kw)
            # h_out = (h_in + 2 * p - k) / s + 1, same as w_out
            h_out = self.in_height - self.kernel_size + 1
            w_out = self.in_width - self.kernel_size + 1
            feature_map = np.zeros((h_out, w_out))
            for i in range(h_out):
                for j in range(w_out):
                    window = x[i:i + self.kernel_size, j: j + self.kernel_size]
                    feature_map[i, j] = np.sum(kw * window)
            feature_maps.append(feature_map)
        return np.array(feature_maps)


def npy_cnn1d(x, w, k_size):
    seq_outputs = []
    out_lens = x.shape[1] - k_size + 1
    for i in range(out_lens):
        window = x[:, i: i + k_size]
        k_outputs = []
        for wi in w:
            k_outputs.append(np.sum(wi * window))
        seq_outputs.append(np.array(k_outputs))

    return np.array(seq_outputs).T


if __name__ == '__main__':
    x = np.random.randn(25).reshape((1, 5, 5))
    x_tensor = torch.FloatTensor(x)
    x_tensor = torch.unsqueeze(x_tensor, 0)

    in_channel = 1
    out_channel = 5
    k_size = 3

    torch_model = TorchCNN(in_channel, out_channel, k_size)
    state_dict = torch_model.state_dict()
    torch_w = state_dict["cnn.weight"]  # out_channel_size x in_channel_size x h_out x w_out
    torch_w = torch_w.squeeze().numpy()
    # h_out = (h_in + 2 * p - k ) / s + 1
    print(torch_w.shape)
    torch_out = torch_model(x_tensor)  # batch_size x out_channel_size x h_out x w_out
    print(torch_out)

    my_model = MyCNN(torch_w, x.shape[1], x.shape[2], k_size)
    output = my_model.forward(x.squeeze(0))
    print(output)

    # conv1d
    input_channel = 5
    output_channel = 7
    k_size = 3  #
    torch_cnn1d = nn.Conv1d(input_channel, output_channel, k_size, bias=False)
    state_dict = torch_cnn1d.state_dict()  # weight
    print(state_dict["weight"].shape)

    x = np.random.randn(50).reshape((5, 10))
    x_tensor = torch.FloatTensor(x)
    torch_out = torch_cnn1d(x_tensor)
    print(torch_out.shape)

    npy_output = npy_cnn1d(x, state_dict["weight"].numpy(), k_size)
    print(npy_output.shape)
