import numpy as np
import torch
import torch.nn as nn

if __name__ == '__main__':

    num_embed = 6
    embed_size = 5
    embedding = nn.Embedding(num_embed, embed_size)
    print(embedding.weight)

    s = "abcdef"
    vocab = dict(zip(s, list(range(len(s)))))
    print(vocab)

    sentences = []
    for _ in range(3):
        list_s = list(s)
        np.random.shuffle(list_s)
        sentences.append([vocab[i] for i in list_s])
    print(sentences)

    x_tensor = torch.LongTensor(sentences)
    print(embedding(x_tensor))
    print(embedding(x_tensor).shape)
