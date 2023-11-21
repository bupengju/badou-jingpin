import torch.nn as nn
from transformers import BertModel


class SentenceBert(nn.Module):

    def __init__(self, class_num, bert_path, hidden_size, dropout_rate):
        super(SentenceBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size, class_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        """
        :param x: torch.LongTensor, shape: [batch_size, seq_len]
        :param y: torch.LongTensor, shape: [batch_size,]
        :return: loss or logits
        """
        mask = x.gt(0)
        x = self.bert(x, attention_mask=mask)[0]
        x = self.dropout(x)
        # 取出句子的第一个token的输出作为句子的表示
        x = self.rnn(x)[0][:, 0, :]
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return x
