import numpy as np
import torch
import torch.nn as nn


def npy_conv1d(x, state_dict):
    w = state_dict["weight"].numpy()
    b = state_dict["bias"].numpy()

    result = []
    for i in range(x.shape[1] - k_size + 1):
        win = x[:, i:i+k_size]
        out = []
        for kernel in w:
            out.append(np.sum(kernel * win))
        result.append(np.array(out) + b)

    return np.array(result).T


if __name__ == '__main__':
    input_dim = 7
    hidden_size = 8
    k_size = 2
    torch_conv1d = nn.Conv1d(input_dim, hidden_size, k_size)
    state_dict = torch_conv1d.state_dict()
    for k, v in state_dict.items():
        print(k, v.shape)

    x = torch.rand((7, 4))  # embed_dim x max_lens

    print(torch_conv1d(x.squeeze(0)).shape)

    print(npy_conv1d(x.numpy(), state_dict).shape)
