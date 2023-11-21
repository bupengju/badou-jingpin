import collections
import math
from pathlib import Path

import jieba


class BM25(object):

    EPSILON = 0.25
    PARAM_K1 = 1.5
    PARAM_B = 0.6

    def __init__(self, corpus):
        self.corpus_size = 0
        self.word_freq_per_doc = collections.defaultdict(dict)
        self.every_single_doc_len = collections.defaultdict(int)
        self.total_doc_len = 0
        self.doc_contained_word = collections.defaultdict(set)
        self.idf = collections.defaultdict(float)
        self.init(corpus)
        self.calc_idf()

    def init(self, corpus):
        for idx, doc in corpus.items():
            self.corpus_size += 1
            self.every_single_doc_len[idx] = len(doc)
            self.total_doc_len += len(doc)

            freq = collections.defaultdict(int)
            for word in doc:
                freq[word] += 1
                self.doc_contained_word[word].add(idx)
            self.word_freq_per_doc[idx] = freq

    def calc_idf(self):
        idf_sum = 0
        negative_idf = []
        for word, docs in self.doc_contained_word.items():
            doc_len_contained_word = len(docs)
            idf = math.log(self.corpus_size - doc_len_contained_word + 0.5)
            idf -= math.log(doc_len_contained_word + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idf.append(word)
        avg_idf = idf_sum / len(self.idf)
        eps = BM25.EPSILON * avg_idf
        for word in negative_idf:
            self.idf[word] = eps

    @property
    def avgdl(self):
        return self.total_doc_len / self.corpus_size

    def get_score(self, query, doc_index):
        k1 = BM25.PARAM_K1
        b = BM25.PARAM_B
        score = 0
        word_freq_on_cur_doc = self.word_freq_per_doc[doc_index]
        for word in query:
            if word not in word_freq_on_cur_doc:
                continue
            word_freq = word_freq_on_cur_doc[word]
            k = k1 * (1-b+b*self.every_single_doc_len[doc_index]/self.avgdl)
            score += self.idf[word] * word_freq * (k1 + 1) / (word_freq + k)
        return [doc_index, score]

    def get_scores(self, query):
        scores = [self.get_score(query, idx) for idx in self.every_single_doc_len.keys()]
        return scores


def load_data(corpus_path):
    corpus = collections.defaultdict(list)
    for f in sorted(Path(corpus_path).glob("*.txt")):
        cate = f.stem

        with open(f.as_posix(), "r", encoding="utf-8") as file:
            for line in file:
                words = jieba.lcut(line.strip())
                corpus[cate].extend(words)
    return corpus


def main():
    query = jieba.lcut("耶伦正式成为美国首位女财长")
    corpus = load_data("./data/category_corpus")
    bm25 = BM25(corpus)
    scores = bm25.get_scores(query)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    print(scores)


if __name__ == '__main__':
    main()
