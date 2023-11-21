import numpy as np
import torch

from config import config
from evaluate import Evaluate
from loader import load_data
from model import SimNetwork
from model import choose_optim


# 尝试在文本匹配中使用trpletloss


def main(cfg):
    train_data = load_data(cfg["train_data_path"], cfg, True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = SimNetwork(cfg)
    net.to(device)
    optim = choose_optim(net.parameters(), cfg)

    evaluate = Evaluate(net, cfg)

    for e in range(cfg["epoch"]):
        net.train()
        watch_loss = []
        for batch_data in train_data:
            optim.zero_grad()
            x1, x2, y = [d.to(device) for d in batch_data]
            loss = net(x1, x2, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        acc = evaluate.eval()
        print("Epoch: %d, loss: %.4f, acc: %.4f" % (e + 1, np.mean(watch_loss).item(), acc))


if __name__ == '__main__':
    main(config)
