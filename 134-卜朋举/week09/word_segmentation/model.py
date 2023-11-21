import torch.nn as nn
from transformers import BertModel


class TorchModel(nn.Module):

    def __init__(self, cfg):
        super(TorchModel, self).__init__()
        self.model_type = cfg["model_type"]
        self.max_lens = cfg["num_layer"]
        self.hidden_size = cfg["hidden_size"]
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["embed_size"], padding_idx=0, max_norm=1)
        if self.model_type.lower() == "rnn":
            self.encoder = nn.RNN(cfg["embed_size"], cfg["hidden_size"], cfg["num_layer"], batch_first=True)
        elif self.model_type.lower() == "lstm":
            self.encoder = nn.LSTM(cfg["embed_size"], cfg["hidden_size"], cfg["num_layer"], batch_first=True)
        elif self.model_type.lower() == "gru":
            self.encoder = nn.GRU(cfg["embed_size"], cfg["hidden_size"], cfg["num_layer"], batch_first=True)
        elif self.model_type.lower() == "bert":
            self.encoder = BertModel.from_pretrained("../data/bert-base-chinese", return_dict=False)
            cfg["hidden_size"] = self.encoder.config.hidden_size
        else:
            raise "model_type 参数错误!"

        self.classifier = nn.Linear(cfg["hidden_size"], cfg["class_num"])
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):
        if self.model_type.lower() == "bert":
            seq_output, pooler_output = self.encoder(x)
        else:
            x = self.embed(x)
            seq_output, _ = self.encoder(x)

        out = self.classifier(seq_output)

        if y is None:
            return out
        else:
            # (batch_size * sen_len, class_num),   (batch_size * sen_len, 1) 注意此处的shape
            return self.loss(out.view(-1, 2), y.view(-1))
