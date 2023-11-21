import math

import torch
import torch.nn as nn
from torch.nn import Transformer


def generate_square_subsequent_mask(sz, device=torch.device('cpu')):
    mask = torch.triu(torch.ones((sz, sz), device=device), diagonal=1)
    mask = mask.masked_fill(mask==1, float('-inf'))
    return mask


def create_masks(src, trg, device=torch.device('cpu')):
    src_seq_len = src.shape[0]
    trg_seq_len = trg.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    trg_mask = torch.zeros((trg_seq_len, trg_seq_len), device=device).type(torch.bool)
    src_padding_mask = (src == 0)  # .transpose(0, 1)
    trg_padding_mask = (trg == 0)  # .transpose(0, 1)
    return src_mask, trg_mask, src_padding_mask, trg_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) #pe: [max_len, emb_size]
        pe = pe.unsqueeze(0).transpose(0, 1) #pe: [max_len, 1, emb_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):

    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size, dim_feedforward, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg) * math.sqrt(self.emb_size))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)
    
    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size)), memory, tgt_mask)
