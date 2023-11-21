import numpy as np
import torch.cuda

from config import config
from evaluate import Evaluator
from loader import build_vocab
from loader import load_dataset
from model import TorchModel
from model import choose_optim


def main():
    vocab = build_vocab(config["vocab_path"])

    train_dataset = load_dataset(config["train_data_path"], config["schema_path"], vocab, config["max_length"],
                                 config["batch_size"], True)
    test_dataset = load_dataset(config["test_data_path"], config["schema_path"], vocab, config["max_length"],
                                config["batch_size"], False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = TorchModel(len(vocab), config["hidden_size"], config["num_layers"], config["class_num"], config["use_crf"])
    net.to(device)

    evaluator = Evaluator(net, test_dataset, config["batch_size"], config["use_crf"])

    optim = choose_optim(net.parameters(), config["lr"], config["optim"])
    for e in range(config["epoch"]):
        net.train()
        watch_loss = []
        for x, y in train_dataset:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            loss = net(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("Epoch: %03d, Loss: %.6f" % (e + 1, np.mean(watch_loss).item()))
    evaluator = Evaluator(net, test_dataset, config["batch_size"], config["use_crf"])
    evaluator.eval()
    torch.save(net.state_dict(), config["model_path"])


if __name__ == '__main__':
    main()
