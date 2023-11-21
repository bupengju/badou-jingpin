import math

import numpy as np
import torch
from transformers import BertModel


class BertByHand(object):

    def __init__(self, state_dict):
        self.num_att_headers = 12
        self.hidden_size = 768
        self.num_layers = 1
        self.state_dict = state_dict
        self.init(state_dict)

    def init(self, state_dict):
        # embedding
        self.word_embed = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embed = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embed = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embed_layer_norm_w = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embed_layer_norm_b = state_dict["embeddings.LayerNorm.bias"].numpy()

        # transform
        self.transform_weights = []
        for i in range(self.num_layers):
            qw = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            qb = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            kw = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            kb = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            vw = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            vb = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            att_output_w = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            att_output_b = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            att_ln_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            att_ln_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            inter_w = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            inter_b = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_w = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_b = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_ln_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_ln_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transform_weights.append(
                [qw, qb, kw, kb, vw, vb, att_output_w, att_output_b,
                 att_ln_w, att_ln_b, inter_w, inter_b, output_w,
                 output_b, ff_ln_w, ff_ln_b
                 ]
            )

        self.pooler_dense_w = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_b = state_dict["pooler.dense.bias"].numpy()

    def embed_forward(self, x):
        we = self.take_embed(self.word_embed, x)  # max_len --> max_len * hidden_size
        pe = self.take_embed(self.position_embed, list(range(len(x))))  # max_len --> max_len * hidden_size
        te = self.take_embed(self.token_type_embed, [0] * len(x))
        embed = we + pe + te
        embed = self.layer_norm(embed, self.embed_layer_norm_w, self.embed_layer_norm_b)
        return embed

    def take_embed(self, embed, x):
        return np.take(embed, x, axis=0)

    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    def single_transformer_layer_forward(self, x, layer_idx):
        weights = self.transform_weights[layer_idx]
        qw, qb, kw, kb, vw, vb, att_output_w, att_output_b, \
        att_ln_w, att_ln_b, inter_w, inter_b, output_w, \
        output_b, ff_ln_w, ff_ln_b = weights
        att_output = self.self_attention(
            x, qw, qb, kw, kb, vw, vb, att_output_w, att_output_b, self.num_att_headers, self.hidden_size
        )
        x = self.layer_norm(x + att_output, att_ln_w, att_ln_b)
        ff_x = self.feed_forward(x, inter_w, inter_b, output_w, output_b)
        x = self.layer_norm(x + ff_x, ff_ln_w, ff_ln_b)
        return x

    def layer_normal(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    def forward(self, x):
        x = self.embed_forward(x)
        seq_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(seq_output[0])
        return seq_output, pooler_output

    def self_attention(self, x, qw, qb, kw, kb, vw, vb, att_output_w, att_output_b, num_att_headers, hidden_size):
        q = np.dot(x, qw.T) + qb
        k = np.dot(x, kw.T) + kb
        v = np.dot(x, vw.T) + vb

        att_head_size = hidden_size // num_att_headers
        q = self.transpose_multi_head(q, att_head_size, num_att_headers)
        k = self.transpose_multi_head(k, att_head_size, num_att_headers)
        v = self.transpose_multi_head(v, att_head_size, num_att_headers)
        qk = np.matmul(q, k.swapaxes(1, 2))  # num_att_header, max_len, max_len
        qk /= np.sqrt(att_head_size)
        qk = self.sofmax(qk)
        qkv = np.matmul(qk, v)
        qkv = qkv.swapaxes(1, 0).reshape(-1, hidden_size)
        att = np.dot(qkv, att_output_w.T) + att_output_b
        return att

    def transpose_multi_head(self, x, att_head_size, num_att_head):
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_att_head, att_head_size)
        x = x.swapaxes(1, 0)  # num_att_head, max_len, att_head_size)
        return x

    def sofmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def feed_forward(self, x, inter_w, inter_b, output_w, output_b):
        x = np.dot(x, inter_w.T) + inter_b
        x = self.gelu(x)
        x = np.dot(x, output_w.T) + output_b
        return x

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_w.T) + self.pooler_dense_b
        x = np.tanh(x)
        return x


if __name__ == '__main__':
    bert = BertModel.from_pretrained("./data/bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()
    print(state_dict.keys())
    bert.eval()
    x_tensor = torch.LongTensor([[2450, 15486, 15167, 2110]])
    seq_output, pool_output = bert(x_tensor)
    print(seq_output.shape)
    print(pool_output.shape)

    bert_by_hand = BertByHand(state_dict)
    seq_output, pool_output = bert_by_hand.forward(x_tensor.numpy().squeeze())
    print(seq_output.shape)
    print(pool_output.shape)
