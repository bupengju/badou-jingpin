import torch
import torch.nn as nn
from transformers import BertModel

from config import config


class ModelHelper(nn.Module):

    def __init__(self):
        super(ModelHelper, self).__init__()
        vocab_size = config["vocab_size"] + 1
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        class_num = config["class_num"]
        model_type = config["model_type"]
        self.use_bert = False
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0, max_norm=1.)
        if model_type.lower() == "fasttext":
            self.encoder = lambda x: x
        elif model_type.lower() == "textrnn":
            self.encoder = TextRNN(config)
        elif model_type.lower() == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type.lower() == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type.lower() == "textcnn":
            self.encoder = TextCNN(config)
        elif model_type.lower() == "gatedcnn":
            self.encoder = GatedCNN(config)
        elif model_type.lower() == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type.lower() == "bertlstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type.lower() == "bertcnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type.lower() == "bertmidlayer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size
        else:
            pass
        self.pooling_style = config["pooling_style"]
        self.fc = nn.Linear(hidden_size, class_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embed(x)
            x = self.encoder(x)

        if isinstance(x, tuple):
            x = x[0]

        if self.pooling_style == "max":
            pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            pooling_layer = nn.AvgPool1d(x.shape[1])

        x = pooling_layer(x.transpose(1, 2)).squeeze()
        out = self.fc(x)

        if y is not None:
            return self.loss(out, y.squeeze())
        else:
            return out


class TextRNN(nn.Module):

    def __init__(self, cfg):
        super(TextRNN, self).__init__()

        self.bi_lstm = nn.LSTM(cfg["hidden_size"], cfg["hidden_size"], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(2 * cfg["hidden_size"], cfg["hidden_size"])

    def forward(self, x):
        outputs, (hn, cn) = self.bi_lstm(x)
        outputs = self.dropout(outputs) if self.training else outputs
        outputs, (hn, cn) = self.lstm(outputs)
        return outputs, (hn, cn)


class TextCNN(nn.Module):

    def __init__(self, cfg):
        super(TextCNN, self).__init__()
        pad = int((cfg["kernel_size"] - 1) / 2)
        self.cnn = nn.Conv1d(cfg["hidden_size"], cfg["hidden_size"], cfg["kernel_size"], padding=pad)

    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class GatedCNN(nn.Module):

    def __init__(self, cfg):
        super(GatedCNN, self).__init__()
        self.conv_a = TextCNN(cfg)
        self.conv_b = TextCNN(cfg)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(x)
        b = self.sigmoid(b)
        return a * b


class TextRCNN(nn.Module):

    def __init__(self, cfg):
        super(TextRCNN, self).__init__()
        self.bi_lstm = nn.LSTM(cfg["hidden_size"], cfg["hidden_size"], batch_first=True)
        pad = int((cfg["kernel_size"] - 1) / 2)
        self.cnn = nn.Conv1d(2 * cfg["hidden_size"], cfg["hidden_size"], cfg["kernel_size"], padding=pad)

    def forward(self, x):
        outputs, (hn, cn) = self.bi_lstm(x)
        out = self.cnn(outputs.transpose(1, 2)).transpose(1, 2)
        return out


class BertLSTM(nn.Module):

    def __init__(self, cfg):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(cfg["pretrain_model_path"], return_dict=False)
        hidden_size = self.bert.config.hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        seq_out, pooler_out = self.bert(x)
        outputs, (hn, cn) = self.lstm(seq_out)
        return outputs, (hn, cn)


class BertCNN(nn.Module):

    def __init__(self, cfg):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(cfg["pretrain_model_path"], return_dict=False)
        hidden_size = self.bert.config.hidden_size
        pad = int((cfg["kernel_size"] - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, cfg["kernel_size"], padding=pad)

    def forward(self, x):
        seq_out, pooler_out = self.bert(x)
        seq_out = self.cnn(seq_out.transpose(1, 2)).transpose(1, 2)
        return seq_out


class BertMidLayer(nn.Module):

    def __init__(self, cfg):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(cfg["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states


def choose_optim(cfg, model):
    optim = cfg["optim"]
    lr = cfg["lr"]

    if optim.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        return torch.optim.SGD(model.parameters(), lr=lr)
