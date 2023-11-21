import torch
from torch.optim import AdamW
from transformers import get_scheduler

from config import config
from evaluate import test
from loader import load_data
from model import Bert


def main():
    train_data = load_data(config["train_data_path"], config["vocab_path"],
                           config["schema_path"], config["batch_size"], config["max_length"])
    test_data = load_data(config["test_data_path"], config["vocab_path"],
                          config["schema_path"], config["batch_size"], config["max_length"], shuffle=False)

    model = Bert(config["class_num"], config["use_crf"])
    model.train()

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    scheduler = get_scheduler(
        "linear", optimizer, config["epoch"], len(train_data))

    for e in range(config["epoch"]):
        watch_loss = []
        for i, (x, y) in enumerate(train_data):
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            watch_loss.append(loss.item())

        watch_loss = sum(watch_loss) / len(watch_loss)
        watch_acc = test(model, test_data)
        print(f"epoch: {e}, loss: {watch_loss}, acc: {watch_acc}")

    torch.save(model.state_dict(), config["model_path"])


if __name__ == '__main__':
    main()
