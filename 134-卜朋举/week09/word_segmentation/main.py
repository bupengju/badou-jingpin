import torch

from config import config
from loader import build_vocab
from loader import load_dataset
from model import TorchModel


def inference(sentence):
    vocab = build_vocab(config["vocab_path"])
    net = TorchModel(config)
    net.load_state_dict(torch.load(config["model_path"]))
    net.eval()

    seq = [vocab.get(char, vocab["<UNK>"]) for char in sentence]

    with torch.no_grad():
        pred = net(torch.LongTensor([seq]))
        pred = torch.argmax(pred, -1).squeeze()

    result = []
    for char, label in zip(sentence, pred.numpy()):
        result.append(char)
        if label == 1:
            result.append(" ")
    print("".join(result))


def train():
    vocab = build_vocab(config["vocab_path"])
    config["vocab_size"] = len(vocab)
    train_dataset = load_dataset(vocab, config["corpus_path"], config["max_length"], config["batch_size"], True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = TorchModel(config)
    net.to(device)

    optim = torch.optim.Adam(net.parameters(), lr=config["lr"])

    for epoch in range(config["epoch"]):
        net.train()
        watch_loss = []
        for x, y in train_dataset:
            optim.zero_grad()
            loss = net(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print("Epoch: %03d, loss: %.6f" % (epoch + 1, sum(watch_loss)))

    torch.save(net.state_dict(), config["model_path"])


if __name__ == '__main__':
    train()
    inference("昨日上海天然橡胶期货价格再度大幅上扬")
