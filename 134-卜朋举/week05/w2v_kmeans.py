import collections

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


def load_sentences(corpus_path):
    sentences = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            sentences.append(jieba.lcut(line.strip()))
    return sentences


def sentences_to_vec(sentences, model_path):
    vectors = []
    model = Word2Vec.load(model_path)
    for sentence in sentences:
        vector = np.zeros(model.vector_size)
        count = len(sentence)
        for word in sentence:
            try:
                vector += model.wv[word]
            except:
                count -= 1
        vectors.append(vector / count)
    return np.array(vectors)


def main(title_path, model_path):
    sentences = load_sentences(title_path)
    vectors = sentences_to_vec(sentences, model_path)
    n_cluster = int(np.sqrt(len(sentences)))
    kmeans = KMeans(n_cluster)
    kmeans.fit(vectors)

    sentences_label_dict = collections.defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentences_label_dict[label].append("".join(sentence))

    for label, sentences in sentences_label_dict.items():
        print("Cluster label: %s" % label)
        for i in range(min(5, len(sentences))):
            print(sentences[i])
        print("*" * 50)


if __name__ == '__main__':
    main("./data/titles.txt", "./out/w2v.model")
