import torch
import torch.nn as nn


if __name__ == '__main__':
    m = nn.Sigmoid()
    bceloss = nn.BCELoss()
    inputs = torch.randn(5)
    target = torch.FloatTensor([1, 0, 1, 0, 0])
    print(bceloss(m(inputs), target))

    m2 = nn.Softmax(dim=-1)
    inputs = torch.randn(15).view(3, 5)
    print(m2(inputs))
    target = torch.LongTensor([1, 2, 4])
    multi_loss = nn.CrossEntropyLoss()
    print(multi_loss(m2(inputs), target))
