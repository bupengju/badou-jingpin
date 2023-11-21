import torch
import torch.nn as nn
import numpy as np

x = torch.randn([2, 4, 5])

bn = nn.BatchNorm1d(4)
bn_out = bn(x)
print(bn_out.shape)

ln = nn.LayerNorm(5)
ln_out = ln(x)
print(ln_out.shape)

n_features = 5
eps = 1e-05
momentum = 0.1

gamma = bn.state_dict()["weight"].numpy()[:, np.newaxis]
beta = bn.state_dict()["bias"].numpy()[:, np.newaxis]
running_mean = np.zeros(n_features)
running_var = np.zeros(n_features)

mean = np.mean(x.numpy(), axis=1, keepdims=True)
var = np.var(x.numpy(), axis=1, keepdims=True)

running_mean = momentum * running_mean + (1 - momentum) * mean
running_var = momentum * running_var + (1 - momentum) * var

x_norm = (x.numpy() - mean) / np.sqrt(var + eps)

y = gamma * x_norm + beta
print(y[0])
print(bn_out[0])

