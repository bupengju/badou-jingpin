import numpy as np
import torch

from config import config
from loader import build_vocab


def evaluate(model, pre_words):
    vocab = build_vocab(config["vocab_path"])
    idx2word = {idx: word for word, idx in vocab.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            x = [vocab.get(word, vocab["<unk>"]) for word in pre_words]
            x = torch.LongTensor(x).unsqueeze(0).to(device)
            y = model(x)
            if config["sample_type"] == "argmax":
                y = torch.argmax(y, dim=-1)
            else:
                p = torch.softmax(y, dim=-1).squeeze().cpu().numpy()
                # print(y.shape, len(p))
                # print(p.shape)
                y = np.random.choice(list(range(len(p))), p=p)
            pre_words += idx2word.get(y.item(), vocab["<unk>"])
        
        print(pre_words)
