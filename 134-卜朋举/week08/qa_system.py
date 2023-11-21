import collections
import json
from pathlib import Path

import jieba
import numpy as np
from gensim.models import Word2Vec

from bm25 import BM25
from similarity_function import similarity_base_on_ed
from similarity_function import similarity_base_on_jd


class QASystem(object):

    def __init__(self, questions_path, method):
        self.target_to_questions = self.load_questions(questions_path)
        self.method = method.lower()
        if self.method == "bm25":
            self.bm25_model = self.load_bm25()
        elif self.method == "word2vec":
            self.w2v_model = self.load_w2v()
            self.target_to_vectors = self.target_to_vectors()
        else:
            pass

    def load_questions(self, questions_path):
        target_to_questions = {}
        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                target_to_questions[line["target"]] = line["questions"]
        return target_to_questions

    def load_bm25(self):
        corpus = collections.defaultdict(list)
        for target, questions in self.target_to_questions.items():
            for question in questions:
                corpus[target].extend(jieba.lcut(question))
        return BM25(corpus)

    def load_w2v(self):
        if Path("./out/w2v.model").exists():
            w2v = Word2Vec.load("./out/w2v.model")
        else:
            corpus = []
            for question in self.target_to_questions.values():
                corpus.append(jieba.lcut(question))
            w2v = Word2Vec(corpus, vector_size=32, min_count=1)
            w2v.save("./out/w2v.model")

        return w2v

    def target_to_vectors(self):
        result = {}
        for target, questions in self.target_to_questions.items():
            vec = []
            for question in questions:
                vec.append(self.sentence_to_vec(question))
            result[target] = np.array(vec)
        return result

    def sentence_to_vec(self, sentence):
        words = jieba.lcut(sentence)
        vec = np.zeros(self.w2v_model.wv.vector_size)
        count = 0
        for word in words:
            if word in self.w2v_model.wv:
                count += 1
                vec += self.w2v_model.wv[word]

        if count == 0:
            return vec
        else:
            vec = np.array(vec) / count
            vec = vec / np.sqrt(np.sum(np.square(vec)))
            return vec

    def query(self, q):
        result = []
        if self.method == "edit_distance":
            for target, questions in self.target_to_questions.items():
                scores = [similarity_base_on_ed(q, question) for question in questions]
                score = max(scores)
                result.append([target, score])
        elif self.method == "jaccard_distance":
            for target, questions in self.target_to_questions.items():
                scores = [similarity_base_on_jd(q, question) for question in questions]
                score = max(scores)
                result.append([target, score])
        elif self.method == "bm25":
            words = jieba.lcut(q)
            result = self.bm25_model.get_scores(words)
        elif self.method == "word2vec":
            q_vec = self.sentence_to_vec(q)
            for target, vecs in self.target_to_vectors.items():
                score = np.mean(np.dot(q_vec, vecs.T))
                result.append([target, score])
        else:
            raise "method error: unknown method!"
        result = sorted(result, key=lambda x: x[1], reverse=True)[:3]
        return result


def main():
    qa = QASystem("./data/train.json", "bm25")
    result = qa.query("我想重置下固话密码")
    print(result)


if __name__ == '__main__':
    main()
