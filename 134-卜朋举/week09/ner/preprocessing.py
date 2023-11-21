import re

import numpy as np


def count_line_len(in_file, out_file):
    with open(in_file, "r", encoding="utf-8") as f:
        segments = f.read().split("\n\n")
        sentences, labels = [], []
        for segment in segments:
            sentence, label = [], []
            for line in segment.split("\n"):
                if line.strip() == "":
                    continue

                char, tag = line.split()
                sentence.append(char)
                label.append(tag)
            sentences.append(sentence)
            labels.append(label)
    count = [len(sentence) for sentence in sentences]

    q1 = np.percentile(count, 25)  # 第一四分位数
    q2 = np.percentile(count, 50)  # 第二四分位数（中位数）
    q3 = np.percentile(count, 75)  # 第三四分位数

    print("第一四分位数 (Q1):", q1)
    print("第二四分位数 (Q2):", q2)
    print("第三四分位数 (Q3):", q3)
    print("95%: ", np.percentile(count, 95))
    print("第四四分位数 (Q4):", np.max(count))  # 第四四分位数等于最大值

    new_sentences, new_labels = [], []
    for idx, sentence in enumerate(sentences):
        label = labels[idx]

        matches = list(re.finditer("[,，；;\?!！:：]", "".join(sentence)))
        start_index = 0
        for match in matches:
            end_index = match.end()
            print(end_index, " ", "".join(sentence))
            cond1 = all([tag == "O" for tag in label[start_index:end_index]])
            if not cond1:
                new_sentences.append(sentence[start_index:end_index])
                new_labels.append(label[start_index:end_index])
            start_index = end_index

    count = [len(sentence) for sentence in new_sentences]

    q1 = np.percentile(count, 25)  # 第一四分位数
    q2 = np.percentile(count, 50)  # 第二四分位数（中位数）
    q3 = np.percentile(count, 75)  # 第三四分位数

    print()
    print("第一四分位数 (Q1):", q1)
    print("第二四分位数 (Q2):", q2)
    print("第三四分位数 (Q3):", q3)
    print("95%: ", np.percentile(count, 95))
    print("第四四分位数 (Q4):", np.max(count))  # 第四四分位数等于最大值

    with open(out_file, "w", encoding="utf-8") as f:
        for idx, sentence in enumerate(new_sentences):
            label = new_labels[idx]
            for i in range(len(sentence)):
                f.write("%s %s\n" % (sentence[i], label[i]))
            # f.write("\n")


if __name__ == '__main__':
    count_line_len("../data/ner/test.txt", "../data/ner/new_test.txt")
