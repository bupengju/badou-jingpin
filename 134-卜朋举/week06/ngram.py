import collections
import math


class NGram(object):

    def __init__(self, corpus_path, n=3):
        self.n = n
        self.sentences = self.load_corpus(corpus_path)
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.sep = "|"
        self.unk_prob = 1e-05
        self.fix_backoff_prop = 0.4
        self.ngram_count_dict = {i + 1: collections.defaultdict(int) for i in range(n)}
        self.ngram_prop_dict = {i + 1: collections.defaultdict(float) for i in range(n)}
        self.calc_ngram_count()
        self.calc_ngram_prop()

    def load_corpus(self, corpus_path):
        sentences = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                sentences.append(line.strip().split())
        return sentences

    def calc_ngram_count(self):
        for sentence in self.sentences:
            sentence = [self.sos] + sentence + [self.eos]
            for window_size in range(1, self.n + 1):
                for idx in range(len(sentence)):
                    if len(sentence[idx: idx + window_size]) != window_size:
                        continue
                    ngram = self.sep.join(sentence[idx: idx + window_size])
                    self.ngram_count_dict[window_size][ngram] += 1
        self.ngram_count_dict[0] = sum(self.ngram_count_dict[1].values())

    def calc_ngram_prop(self):
        for window_size in range(1, self.n + 1):
            for ngram, count in self.ngram_count_dict[window_size].items():
                if window_size > 1:
                    ngram_splits = ngram.split(self.sep)
                    ngram_prefix = self.sep.join(ngram_splits[:-1])
                    ngram_prefix_count = self.ngram_count_dict[window_size - 1][ngram_prefix]
                else:
                    ngram_prefix_count = self.ngram_count_dict[0]
                self.ngram_prop_dict[window_size][ngram] = count / ngram_prefix_count

    def get_ngram_prop(self, ngram):
        n = len(ngram.split(self.sep))
        if ngram in self.ngram_prop_dict[n]:
            return self.ngram_prop_dict[n][ngram]
        elif n == 1:
            return self.unk_prob
        else:
            ngram = self.sep.join(ngram.split(self.sep)[:-1])
            return self.fix_backoff_prop * self.get_ngram_prop(ngram)

    def calc_sentence_ppl(self, sentence):
        sentence = sentence.split(" ")
        sentence = [self.sos] + sentence + [self.eos]
        sentence_prop = 0.
        for idx in range(len(sentence)):
            ngram = self.sep.join(sentence[max(0, idx - self.n + 1): idx + 1])
            sentence_prop += math.log(self.get_ngram_prop(ngram))
        return 2 ** (-sentence_prop / len(sentence))


def main(corpus_path):
    lm = NGram(corpus_path, 3)
    print("词总数:", lm.ngram_count_dict[0])
    print(lm.ngram_prop_dict)
    print(lm.calc_sentence_ppl("e e e e e"))


if __name__ == '__main__':
    main("./data/sample.txt")
