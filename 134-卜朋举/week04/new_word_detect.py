import collections
import math


class NewWordDetect(object):

    def __init__(self, corpus_path, window_size):
        self.window_size = window_size
        self.word_count = collections.defaultdict(int)
        self.left_neighbour = collections.defaultdict(dict)
        self.right_neighbour = collections.defaultdict(dict)
        self.load_corpus(corpus_path)
        self.calc_pmi()
        self.calc_entropy()
        self.calc_word_values()

    def load_corpus(self, corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                sentence = line.strip()
                for word_length in range(1, self.window_size):
                    self.ngram_count(sentence, word_length)

    def ngram_count(self, sentence, word_length):
        for i in range(len(sentence) - word_length + 1):
            word = sentence[i: i + word_length]
            self.word_count[word] += 1

            if i - 1 >= 0:
                char = sentence[i - 1]
                self.left_neighbour[word][char] = self.left_neighbour[word].get(char, 0) + 1

            if i + word_length < len(sentence):
                char = sentence[i + word_length]
                self.right_neighbour[word][char] = self.right_neighbour[word].get(char, 0) + 1

    def calc_word_count_by_length(self):
        self.word_count_by_length = collections.defaultdict(int)
        for word, count in self.word_count.items():
            self.word_count_by_length[len(word)] += count

    def calc_pmi(self):
        self.calc_word_count_by_length()
        self.pmi = {}
        for word, count in self.word_count.items():
            p_word = count / self.word_count_by_length[len(word)]
            p_char = 1
            for char in word:
                p_char *= self.word_count[char] / self.word_count_by_length[1]
            self.pmi[word] = math.log10(p_word / p_char) / len(word)

    def calc_entropy_by_word_count_dict(self, word_count_dict):
        total = sum(word_count_dict.values())
        entropy = sum([-(c / total) * math.log10(c / total) for c in word_count_dict.values()])
        return entropy

    def calc_entropy(self):
        self.word_left_entropy = {}
        self.word_right_entropy = {}

        for word, count_dict in self.left_neighbour.items():
            self.word_left_entropy[word] = self.calc_entropy_by_word_count_dict(count_dict)

        for word, count_dict in self.right_neighbour.items():
            self.word_right_entropy[word] = self.calc_entropy_by_word_count_dict(count_dict)

    def calc_word_values(self):
        self.word_values = {}
        for word in self.pmi.keys():
            if len(word) < 2 or "," in word:
                continue
            pmi = self.pmi.get(word, 1e-03)
            left_en = self.word_left_entropy.get(word, 1e-03)
            right_en = self.word_right_entropy.get(word, 1e-03)
            self.word_values[word] = pmi * left_en * right_en


if __name__ == '__main__':
    nwd = NewWordDetect("./data/sample_corpus.txt", 5)
    # print(nwd.word_count)
    # print(nwd.left_neighbour)
    # print(nwd.right_neighbour)
    # print(nwd.pmi)
    # print(nwd.word_left_entropy)
    # print(nwd.word_right_entropy)
    values_sorted = sorted([(word, count) for word, count in nwd.word_values.items()], key=lambda x: x[1], reverse=True)
    print([x for x, c in values_sorted if len(x) == 2][:10])
    print([x for x, c in values_sorted if len(x) == 3][:10])
    print([x for x, c in values_sorted if len(x) == 4][:10])
