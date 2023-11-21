from pathlib import Path

import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel


class Bert(nn.Module):

    def __init__(self, class_num, use_crf=False):
        super(Bert, self).__init__()
        self.model_dir = Path(__file__).parent.parent.joinpath("data", "bert-base-chinese")
        self.encoder = BertModel.from_pretrained(self.model_dir, return_dict=False)
        self.rnn = nn.GRU(self.encoder.config.hidden_size, self.encoder.config.hidden_size // 2,
                          batch_first=True, bidirectional=True)
        self.classify = nn.Linear(self.encoder.config.hidden_size, class_num)
        self.loss = nn.CrossEntropyLoss()
        self.crf = CRF(class_num, batch_first=True)
        self.use_crf = use_crf

    def forward(self, x, y=None):
        seq_output, pooler_output = self.encoder(**x)
        outputs, hidden_out = self.rnn(seq_output)
        logits = self.classify(outputs)

        if y is not None:
            logits, y = self.bert_decode(logits, x["attention_mask"], y)
            if self.use_crf:
                loss = -self.crf(logits, y, mask=None, reduction="mean")
            else:
                loss = self.loss(logits, y)
            return loss
        else:
            logits = self.bert_decode(logits, x["attention_mask"])
            if self.use_crf:
                logits = self.crf.decode(logits, mask=None)
            else:
                logits = logits
            return logits

    def bert_decode(self, pred, attention_mask, y=None):
        pred = pred.view(-1, pred.shape[-1])
        attention_mask = attention_mask.view(-1)
        selected = attention_mask == 1
        pred = pred[selected]
        if y is not None:
            y = y.view(-1)[selected]
            return pred, y
        else:
            return pred
