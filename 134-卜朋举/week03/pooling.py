import torch
import torch.nn as nn

pool = nn.MaxPool1d(5)

x = torch.randn([3, 4, 5])
print(x.shape)
x = pool(x)  # pooling操作默认对于输入张量的最后一维进行
print(x.shape)
