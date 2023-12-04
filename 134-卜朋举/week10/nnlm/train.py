import torch
import torch.nn as nn
import numpy as np

from config import config
from model import NNLM
from loader import build_vocab
from loader import load_data
from evaluate import evaluate


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab(config["vocab_path"])
    data_loader = load_data(config["data_path"], vocab, config["win_sz"], config["sample_length"], config["batch_size"])
    model = NNLM(len(vocab), config["hidden_dim"], config["num_layers"], config["dropout"]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    for epoch in range(config["epochs"]):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            # print("--> ", x.shape, y.shape)
            x = x.to(device)
            y = y.to(device)
            loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("Epoch: {}, Loss: {}".format(epoch + 1, np.mean(watch_loss)))
        evaluate(model, "如果这真的是梦")


if __name__ == "__main__":
    train()
