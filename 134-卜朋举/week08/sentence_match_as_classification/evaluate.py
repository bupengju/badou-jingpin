import numpy as np
import torch

from loader import load_data


class Evaluate(object):

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.valid_data = load_data(cfg["valid_data_path"], cfg, False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def eval(self):
        self.model.eval()
        acc = []
        with torch.no_grad():
            for idx, batch_data in enumerate(self.valid_data):
                x, y = [d.to(self.device) for d in batch_data]
                pred = self.model(x)
                pred = pred.numpy()
                y = y.numpy()
                acc.append(np.mean(np.argmax(pred, axis=1) == y.squeeze()))
        return np.mean(acc)
