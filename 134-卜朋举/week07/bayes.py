import collections
import json

import jieba


class BayesByHand(object):

    def __init__(self, data_path):
        self.class_prob = collections.defaultdict(float)
        self.words_class_prob = collections.defaultdict(dict)
        self.all_words = set()
        self.stats_freq(data_path)

    def stats_freq(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                class_name = line["tag"]
                title = line["title"]
                words = jieba.lcut(title)
                self.all_words.union(set(words))
                self.class_prob[class_name] += 1
                words_freq = self.words_class_prob[class_name]
                for word in words:
                    words_freq.setdefault(word, 0)
                    words_freq[word] += 1
        self.freq_to_prop()

    def freq_to_prop(self):
        total_class_count = sum(self.class_prob.values())
        self.class_prob = {c: count / total_class_count for c, count in self.class_prob.items()}
        for class_name, words_freq in self.words_class_prob.items():
            total_word_count = sum(words_freq.values())
            for word in words_freq:
                prob = (words_freq[word] + 1) / (total_word_count + len(self.all_words))
                self.words_class_prob[class_name][word] = prob
            self.words_class_prob[class_name]["<UNK>"] = 1 / (total_word_count + len(self.all_words))

    def get_words_class_prob(self, words, class_name):
        result = 1
        for word in words:
            p_unk = self.words_class_prob[class_name]["<UNK>"]
            result *= self.words_class_prob[class_name].get(word, p_unk)

        return result

    def get_class_prob(self, words, class_name):
        px = self.class_prob[class_name]
        pwx = self.get_words_class_prob(words, class_name)
        return px * pwx

    def predict(self, sentence):
        words = jieba.lcut(sentence)
        result = []
        for class_name in self.class_prob:
            prob = self.get_class_prob(words, class_name)
            result.append([class_name, prob])
        result = sorted(result, key=lambda x: x[1], reverse=True)

        pw = sum(x[1] for x in result)
        result = [[c, p/pw]for c, p in result]

        for idx, (c, p) in enumerate(result):
            print("属于类别%s的概率为: %.4f" % (c, p))
            if idx > 5:
                break


if __name__ == '__main__':
    bayes = BayesByHand("./data/train_tag_news.json")
    bayes.predict("目瞪口呆 世界上还有这些奇葩建筑")
