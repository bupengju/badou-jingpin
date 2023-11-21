import numpy as np
import torch

from loader import load_data


class Evaluate(object):

    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model
        self.data = load_data(cfg["test_data_path"], cfg, False)

    def eval(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        acc = []
        for idx, batch_data in enumerate(self.data):
            x, y = [data.to(device) for data in batch_data]
            with torch.no_grad():
                pred = self.model(x)
                pred = pred.numpy()
                y = y.numpy()
                acc.append(np.mean(np.argmax(pred, axis=1) == y.squeeze()))
        acc = np.mean(acc)
        return acc

    def eval100(self):
        dataset = load_data(self.cfg["test_data_path"], self.cfg, True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        acc = []
        for idx, batch_data in enumerate(dataset):
            if idx > 99:
                break
            x, y = [data.to(device) for data in batch_data]
            with torch.no_grad():
                pred = self.model(x)
                pred = pred.numpy()
                y = y.numpy()
                acc.append(np.mean(np.argmax(pred, axis=1) == y.squeeze()))
        acc = np.mean(acc)
        return acc
