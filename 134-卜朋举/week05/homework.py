import collections

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


# 在基于kmeans的聚类中，增加类内相似度的计算


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


def cosine_dist(x1, x2):
    dist = np.dot(x1, x2) / (np.sqrt(np.sum(np.square(x1))) * np.sqrt(np.sum(np.square(x2))))
    return dist


def main(title_path, model_path):
    sentences = load_sentences(title_path)
    vectors = sentences_to_vec(sentences, model_path)
    n_cluster = int(np.sqrt(len(sentences)))
    kmeans = KMeans(n_cluster, n_init="auto")
    kmeans.fit(vectors)

    inner_dist = collections.defaultdict(list)
    for label, vector in zip(kmeans.labels_, vectors):
        center = kmeans.cluster_centers_[label]
        inner_dist[label].append(cosine_dist(vector, center))
    inner_dist = {label: np.mean(val) for label, val in inner_dist.items()}
    inner_dist = sorted(inner_dist.items(), key=lambda x: x[1], reverse=True)

    sentences_label_dict = collections.defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentences_label_dict[label].append("".join(sentence))

    for label, dist in inner_dist:
        print("Cluster label: %s" % label)
        sentences = sentences_label_dict[label]
        for i in range(min(5, len(sentences))):
            print(sentences[i])
        print("*" * 50)


if __name__ == '__main__':
    main("./data/titles.txt", "./out/w2v.model")
