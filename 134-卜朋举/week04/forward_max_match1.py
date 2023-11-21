import time


def load_word_dict(path):
    word_set = set()
    max_lens = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().split(" ")[0]
            word_set.add(word)
            max_lens = max(max_lens, len(word))
    return word_set, max_lens


def cut_word(sentence, word_dict, max_length):
    words = []
    while sentence != "":
        lens = min(max_length, len(sentence))
        word = sentence[:lens]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = word[:len(word)-1]
        sentence = sentence[len(word):]
        words.append(word)
    return words


def main(words_path, corpus_path, output_path):
    start = time.time()
    word_dict, max_length = load_word_dict(words_path)
    writer = open(output_path, "w", encoding="utf-8")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            words = cut_word(line.strip(), word_dict, max_length)
            writer.write("/".join(words)+"\n")
    writer.close()
    end = time.time()
    print("耗时: %.2f 秒!" % (end - start))


if __name__ == '__main__':
    word_dict_path = r"./data/dict.txt"
    corpus_path = r"./data/corpus.txt"
    output_path = r"./out/cut_word_method1.txt"
    main(word_dict_path, corpus_path, output_path)
