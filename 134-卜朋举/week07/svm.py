import json

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import classification_report

LABELS = {
    "健康": 0, "军事": 1, "房产": 2, "社会": 3, "国际": 4, "旅游": 5,
    "彩票": 6, "时尚": 7, "文化": 8, "汽车": 9, "体育": 10, "家居": 11,
    "教育": 12, "娱乐": 13, "科技": 14, "股票": 15, "游戏": 16, "财经": 17
}


def load_data(data_path):
    sentences, label = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            class_name = line["tag"]
            label.append(class_name)
            sentence = line["title"]
            sentences.append(jieba.lcut(sentence))

    return sentences, label


def train_wv(sentences):
    model = Word2Vec(sentences, vector_size=32)
    model.save("./out/w2v.model")


def sentences_to_seq(w2v_model_path, sentences):
    model = Word2Vec.load(w2v_model_path)
    result = []
    for sentence in sentences:
        vec = []
        for word in sentence:
            try:
                vec.append(model.wv[word])
            except:
                vec.append(np.zeros(model.vector_size))
        result.append(np.array(vec).mean(axis=0))
    return np.array(result)


def label_to_idx(label):
    idx = [LABELS.get(c) for c in label]
    return np.array(idx)


def train_svm(seqs, labels):
    model = SVC()
    model.fit(seqs, labels)
    return model


def main(total_data_path, train_data_path, test_data_path):

    sentences, labels = load_data(total_data_path)
    train_wv(sentences)

    train_sentences, train_labels = load_data(train_data_path)
    train_seq = sentences_to_seq("./out/w2v.model", train_sentences)
    train_labels = label_to_idx(train_labels)

    model = train_svm(train_seq, train_labels)

    test_sentences, test_labels = load_data(test_data_path)
    test_seq = sentences_to_seq("./out/w2v.model", test_sentences)
    test_labels = label_to_idx(test_labels)

    pred = model.predict(test_seq)
    print(classification_report(test_labels, pred))


if __name__ == '__main__':
    main("./data/tag_news.json", "./data/train_tag_news.json", "./data/valid_tag_news.json")
