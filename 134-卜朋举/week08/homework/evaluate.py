import numpy as np
import torch

from loader import load_data


class Evaluate(object):

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.valid_data = load_data(cfg["valid_data_path"], cfg, False)
        self.train_data = load_data(cfg["train_data_path"], cfg, True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def kwb_to_vec(self):
        self.q_idx_to_sq_idx = {}
        self.questions = []
        for sq_idx, question_ids in self.train_data.dataset.kwb.items():
            for question_id in question_ids:
                self.q_idx_to_sq_idx[len(self.questions)] = sq_idx
                self.questions.append(question_id)

        with torch.no_grad():
            q_matrix = torch.stack(self.questions, dim=0)
            q_matrix.to(self.device)
            kwb_vecs = self.model(q_matrix)
            self.kwb_vecs = torch.nn.functional.normalize(kwb_vecs, dim=-1)

    def eval(self):
        self.model.eval()
        self.kwb_to_vec()
        acc = []
        with torch.no_grad():
            for idx, batch_data in enumerate(self.valid_data):
                x, y = [d.to(self.device) for d in batch_data]
                pred_vec = self.model(x)
                sim_res = torch.mm(pred_vec, self.kwb_vecs.T)
                hit_index = [self.q_idx_to_sq_idx[i] for i in torch.argmax(sim_res, dim=-1).numpy()]
                acc.append(np.mean(np.array(hit_index) == y.numpy().squeeze()))
        return np.mean(acc)
