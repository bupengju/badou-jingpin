import collections
import math
from pathlib import Path

import jieba


def tokenization(corpus_dict):
    token_dict = collections.defaultdict(list)
    for cate, sentences in corpus_dict.items():
        for sentence in sentences:
            token_dict[cate].extend(jieba.lcut(sentence.strip()))
    return token_dict


def calc_word_doc_count(token_dict):
    word_count_dict = collections.defaultdict(dict)
    doc_count_dict = collections.defaultdict(set)
    for key, words in token_dict.items():
        for word in words:
            word_count_dict[key][word] = word_count_dict[key].get(word, 0) + 1
            doc_count_dict[word].add(key)
    doc_count_dict = {key: len(value) for key, value in doc_count_dict.items()}
    return word_count_dict, doc_count_dict


def calc_tf_idf(word_count_dict, doc_count_dict):
    tf_idf_dict = collections.defaultdict(dict)
    for cate, count_dict in word_count_dict.items():
        for word, count in count_dict.items():
            tf = count / sum(count_dict.values())
            tf_idf_dict[cate][word] = tf * math.log(len(word_count_dict) / (doc_count_dict[word] + 1))
    return tf_idf_dict


def get_top_k_word_by_cate(tf_idf_dict, k=10):
    top_k_word_dict = collections.defaultdict(list)
    for cate, tfidf_dict in tf_idf_dict.items():
        for idx, (w, _) in enumerate(sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)):
            top_k_word_dict[cate].append(w)
            if idx >= k:
                break
    return top_k_word_dict


def main(corpus_root):
    corpus_root = Path(corpus_root)
    corpus_dict = collections.defaultdict(list)
    for txt in sorted(corpus_root.glob("*.txt")):
        with open(txt.as_posix(), "r", encoding="utf-8") as f:
            corpus_dict[txt.stem].extend(f.readlines())
    token_dict = tokenization(corpus_dict)
    word_count_dict, doc_count_dict = calc_word_doc_count(token_dict)
    tf_idf_dict = calc_tf_idf(word_count_dict, doc_count_dict)
    top_k_word_dict = get_top_k_word_by_cate(tf_idf_dict)
    print(top_k_word_dict)


if __name__ == '__main__':
    corpus_root = r"./data/category_corpus"
    main(corpus_root)
