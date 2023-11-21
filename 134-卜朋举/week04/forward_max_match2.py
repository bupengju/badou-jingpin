import time


def load_prefix_word_dict(words_path):
    prefix_dict = {}
    with open(words_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().split(" ")[0]
            for i in range(1, len(word)):
                if word[i] not in prefix_dict:
                    prefix_dict[word[:i]] = 0
            prefix_dict[word] = 1
    return prefix_dict


def cut_word(sentence, prefix_dict):
    words = []
    if sentence == "":
        return words
    start_idx, end_idx = 0, 1
    window = sentence[start_idx: end_idx]
    find_word = window
    while start_idx < len(sentence):
        if window not in prefix_dict or end_idx > len(sentence):
            words.append(find_word)
            start_idx += len(find_word)
            end_idx = start_idx + 1
            window = sentence[start_idx: end_idx]
            find_word = window
            continue

        if prefix_dict[window] == 1:
            find_word = window
            end_idx += 1
            window = sentence[start_idx: end_idx]
            continue

        if prefix_dict[window] == 0:
            end_idx += 1
            window = sentence[start_idx: end_idx]
            continue

    if prefix_dict.get(window) != 1:
        words += list(window)
    else:
        words.append(window)
    return words


def main(words_path, corpus_path, output_path):
    start = time.time()
    prefix_dict = load_prefix_word_dict(words_path)
    writer = open(output_path, "w", encoding="utf-8")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            words = cut_word(line.strip(), prefix_dict)
            writer.write("/".join(words) + "\n")

    writer.close()
    end = time.time()
    print("耗时: %.2f 秒!" % (end - start))


if __name__ == '__main__':
    main("./data/dict.txt", "./data/corpus.txt", "./out/cut_word_method2.txt")
