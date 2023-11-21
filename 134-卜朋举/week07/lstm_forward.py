import numpy as np
import torch
import torch.nn as nn


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def npy_lstm(x, state_dict):
    weight_ih_l0 = state_dict["weight_ih_l0"].numpy()
    bias_ih_l0 = state_dict["bias_ih_l0"].numpy()
    weight_hh_l0 = state_dict["weight_hh_l0"].numpy()
    bias_hh_l0 = state_dict["bias_hh_l0"].numpy()
    h_size = weight_ih_l0.shape[0] // 4

    wii = weight_ih_l0[:h_size, :].T
    wif = weight_ih_l0[h_size:2 * h_size, :].T
    wic = weight_ih_l0[2 * h_size:3 * h_size, :].T
    wio = weight_ih_l0[3 * h_size:, :].T

    bii = bias_ih_l0[:h_size]
    bif = bias_ih_l0[h_size:2 * h_size]
    bic = bias_ih_l0[2 * h_size:3 * h_size]
    bio = bias_ih_l0[3 * h_size:]

    whi = weight_hh_l0[:h_size, :].T
    whf = weight_hh_l0[h_size:2 * h_size, :].T
    whc = weight_hh_l0[2 * h_size:3 * h_size, :].T
    who = weight_hh_l0[3 * h_size:, :].T

    bhi = bias_hh_l0[:h_size]
    bhf = bias_hh_l0[h_size:2 * h_size]
    bhc = bias_hh_l0[2 * h_size:3 * h_size]
    bho = bias_hh_l0[3 * h_size:]

    ct = np.zeros((1, hidden_size))
    ht = np.zeros((1, hidden_size))

    output = []
    for xt in x:
        xt = xt[np.newaxis, :]
        it = sigmoid(np.dot(xt, wii) + bii + np.dot(ht, whi) + bhi)
        ft = sigmoid(np.dot(xt, wif) + bif + np.dot(ht, whf) + bhf)
        ot = sigmoid(np.dot(xt, wio) + bio + np.dot(ht, who) + bho)
        c_tilda = np.tanh(np.dot(xt, wic) + bic + np.dot(ht, whc) + bhc)
        ct = ft * ct + it * c_tilda
        ht = ot * np.tanh(ct)
        output.append(ht)
    return np.array(output).squeeze(), (ht, ct)


def npy_gru(x, state_dict):
    weight_ih_l0 = state_dict["weight_ih_l0"].numpy()
    weight_hh_l0 = state_dict["weight_hh_l0"].numpy()
    bias_ih_l0 = state_dict["bias_ih_l0"].numpy()
    bias_hh_l0 = state_dict["bias_hh_l0"].numpy()
    h_size = weight_ih_l0.shape[0] // 3

    wir = weight_ih_l0[:h_size, :].T
    wiz = weight_ih_l0[h_size:2 * h_size, :].T
    wih = weight_ih_l0[2 * h_size:, :].T

    bir = bias_ih_l0[:h_size]
    biz = bias_ih_l0[h_size:2 * h_size]
    bih = bias_ih_l0[2 * h_size:]

    whr = weight_hh_l0[:h_size, :].T
    whz = weight_hh_l0[h_size:2 * h_size, :].T
    whh = weight_hh_l0[2 * h_size:, :].T

    bhr = bias_hh_l0[:h_size]
    bhz = bias_hh_l0[h_size:2 * h_size]
    bhh = bias_hh_l0[2 * h_size:]

    ht = np.zeros((1, h_size))
    output = []
    for xt in x:
        xt = xt[np.newaxis, :]
        rt = sigmoid(np.dot(xt, wir) + bir + np.dot(ht, whr) + bhr)
        zt = sigmoid(np.dot(xt, wiz) + biz + np.dot(ht, whz) + bhz)
        h_tilda = np.tanh(np.dot(xt, wih) + bih + np.dot(rt * ht, whh) + bhh)
        ht = zt * ht + (1 - zt) * h_tilda
        output.append(ht)
    output = np.array(output).squeeze()
    return output, ht


if __name__ == '__main__':
    length = 6
    input_dim = 12
    hidden_size = 7

    x = torch.randn((length, input_dim))

    print(x.shape)

    torch_lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
    for k, v in torch_lstm.state_dict().items():
        print(k, v.shape)
    out, (ht, ct) = torch_lstm(x)
    print(out.shape, ht.shape)
    output, (np_ht, np_ct) = npy_lstm(x.numpy(), torch_lstm.state_dict())
    print(output.shape, np_ht.shape)

    print("=" * 50)

    torch_gru = nn.GRU(input_dim, hidden_size, batch_first=True)
    for k, v in torch_gru.state_dict().items():
        print(k, v.shape)

    torch_outputs, torch_ht = torch_gru(x)
    print(torch_outputs.shape, torch_ht.shape)

    outputs, ht = npy_gru(x, torch_gru.state_dict())
    print(outputs.shape, ht.shape)
