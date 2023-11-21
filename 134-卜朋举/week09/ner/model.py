import torch
import torch.nn as nn
from torchcrf import CRF


class TorchModel(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers, class_num, use_crf=True, padding_idx=0):
        super(TorchModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx, max_norm=1)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(2 * hidden_size, class_num)
        self.crf = CRF(class_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.use_crf = use_crf

    def forward(self, x, y=None):
        x = self.embed(x)
        seq_out, ht = self.encoder(x)
        out = self.classify(seq_out)

        if y is not None:
            if self.use_crf:
                mask = y.gt(-1)  # 忽略label中的padding
                return - self.crf(out, y, mask, reduction="mean")
            else:
                # 注意此处loss计算中tensor的shape与文本分类中的区别
                return self.loss(out.view(-1, out.shape[-1]), y.view(-1))
        else:
            if self.use_crf:
                return self.crf.decode(out)
            else:
                return out


def choose_optim(model_params, lr, optim_type):
    if optim_type.lower() == "adam":
        optim = torch.optim.Adam(model_params, lr)
    else:
        optim = torch.optim.SGD(model_params, lr)

    return optim
