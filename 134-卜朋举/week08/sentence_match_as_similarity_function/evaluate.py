import numpy as np
import torch

from loader import load_data


class Evaluate(object):

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.valid_data = load_data(cfg["valid_data_path"], cfg, False)
        self.train_data = load_data(cfg["train_data_path"], cfg, True)
        # self.tokenizer = self.train_data.dataset.tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def kwb_to_vec(self):
        self.q_idx_to_sq_idx = {}
        self.questions = []
        for sq_idx, questions in self.train_data.dataset.kwb.items():
            for question in questions:
                self.q_idx_to_sq_idx[len(self.questions)] = sq_idx
                self.questions.append(question)

    def eval(self):
        self.model.eval()
        self.kwb_to_vec()
        acc = []
        with torch.no_grad():
            for batch_data in self.valid_data:
                test_questions, labels = batch_data
                preds = []
                for test_question in test_questions:
                    input_ids = []

                    for question in self.questions:
                        input_ids.append(self.train_data.dataset.sentence_to_seq(test_question, question))

                    input_ids = torch.LongTensor(input_ids)
                    input_ids.to(self.device)
                    pred = self.model(input_ids)[:, 1]  # 如果改为x[:,0]则是两句话不匹配的概率
                    preds.append(self.q_idx_to_sq_idx[np.argmax(pred.numpy()).item()])
                acc.append(np.mean(np.array(preds) == labels.squeeze().numpy()))
        return np.mean(acc)
