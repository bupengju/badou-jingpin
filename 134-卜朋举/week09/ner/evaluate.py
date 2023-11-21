import re
from collections import defaultdict

import numpy as np
import torch.cuda


class Evaluator(object):

    def __init__(self, model, dataset, batch_size, use_crf):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_crf = use_crf

    def eval(self, ):
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON": defaultdict(int),
            "ORGANIZATION": defaultdict(int)
        }
        self.model.eval()
        for idx, batch_data in enumerate(self.dataset):
            sentence = self.dataset.dataset.sentences[idx * self.batch_size:(idx + 1) * self.batch_size]
            if torch.cuda.is_available():
                batch_data = [data.cuda() for data in batch_data]
            x, y = batch_data
            with torch.no_grad():
                pred = self.model(x)
            self.write_stats(y, pred, sentence)
        self.show_stats()

    def write_stats(self, label, pred, sentence):
        if not self.use_crf:
            pred = torch.argmax(pred, dim=-1)

        for gt, pred_result, sent in zip(label, pred, sentence):
            if not self.use_crf:
                pred_result = pred_result.cpu().detach().tolist()
            gt = gt.cpu().detach().tolist()
            true_enti = self.decode(sent, gt)
            print(gt)
            print(pred_result)
            pred_enti = self.decode(sent, pred_result)
            # print(true_enti, pred_enti)

            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_enti[key] if ent in gt[key]])
                self.stats_dict[key]["样本实体数"] += len(true_enti[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_enti[key])

    def show_stats(self):
        f1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            f1 = (2 * precision * recall) / (precision + recall + 1e-5)
            f1_scores.append(f1)
            print("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, f1))
        print("Macro-F1: %f" % np.mean(f1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        print("Micro-F1 %f" % micro_f1)

    def decode(self, sentence, label):
        label = "".join([str(x) for x in label[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", label):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", label):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", label):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", label):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
