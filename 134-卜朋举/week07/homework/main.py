import random
import time
from pathlib import Path

import numpy as np
import torch

from config import config
from evaluate import Evaluate
from loader import load_data
from loader import train_test_split
from model import ModelHelper
from model import choose_optim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg):
    seed = cfg["seed"]
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(cfg["train_data_path"]).exists():
        train_test_split(cfg)

    train_data = load_data(cfg["train_data_path"], cfg, True)

    model = ModelHelper()
    model.to(device)

    optim = choose_optim(cfg, model)

    evaluate = Evaluate(model, cfg)

    for e in range(cfg["epoch"]):
        model.train()
        watch_loss = []
        for idx, batch_data in enumerate(train_data):
            optim.zero_grad()
            x, y = [data.to(device) for data in batch_data]
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        acc = evaluate.eval()
        print("Epoch: %d, loss: %.6f, acc: %.2f" % (e + 1, np.mean(watch_loss).item(), acc))

    model_path = Path(cfg["model_path"]).joinpath("weights", cfg["model_type"] + ".pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path.as_posix())

    start = time.time()
    acc = evaluate.eval100()
    end = time.time()
    print("acc: %.4f, %.2f seconds" % (acc, (end - start)))


if __name__ == '__main__':
    main(config)
