import torch
from transformers import get_scheduler

from loader import load_data
from model import SentenceBert
from config import config


def train():
    train_data_loader = load_data(config["train_data_path"], config["pretrained_path"], config["batch_size"],
                                  config["max_length"], config["max_sentence_length"])
    test_data_loader = load_data(config["test_data_path"], config["pretrained_path"], config["batch_size"],
                                 config["max_length"], config["max_sentence_length"], shuffle=False)

    model = SentenceBert(config["class_num"], config["pretrained_path"],
                         config["hidden_size"], config["dropout_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(
        train_data_loader) * config["epoch"])

    for epoch in range(config["epoch"]):
        model.train()
        watch_loss = []
        for i, batch in enumerate(train_data_loader):
            inputs, labels = batch
            inputs = inputs.to(config["device"])
            labels = labels.to(config["device"])
            optimizer.zero_grad()
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            watch_loss.append(loss.item())
        model.eval()
        correct = 0
        total = 0
        watch_acc = []
        for i, batch in enumerate(test_data_loader):
            inputs, labels = batch
            inputs = inputs.to(config["device"])
            labels = labels.to(config["device"])
            with torch.no_grad():
                logits = model(inputs)
                predict = torch.argmax(logits, dim=-1)
                correct += torch.sum(torch.eq(predict, labels)).item()
                total += len(labels)
                watch_acc.append(correct / total)
        print("Epoch %d, loss %.4f, acc %.4f" % (epoch, sum(
            watch_loss) / len(watch_loss), sum(watch_acc) / len(watch_acc)))
        torch.save(model.state_dict(), config["model_path"])


if __name__ == "__main__":
    train()
