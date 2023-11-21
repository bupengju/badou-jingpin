import collections
import json

import jieba

from calc_tfidf import tokenization
from calc_tfidf import calc_word_doc_count
from calc_tfidf import calc_tf_idf


def load_data(json_path):
    news_list = json.load(open(json_path, "r", encoding="utf-8"))
    corpus = collections.defaultdict(list)
    for news in news_list:
        corpus[news["title"]].append(news["content"])
    token_dict = tokenization(corpus)
    word_count_dict, doc_count_dict = calc_word_doc_count(token_dict)
    tf_idf_dict = calc_tf_idf(word_count_dict, doc_count_dict)
    return tf_idf_dict


def search_engine(query, tfidf_dict, top=5):
    query_words = jieba.lcut(query)
    result = {}
    for cate, tf_idf in tfidf_dict.items():
        score = 0
        for word in query_words:
            score += tf_idf.get(word, 0)
        result[cate] = score

    top_k = []
    for cate, score in sorted(result.items(), key=lambda x: x[1], reverse=True)[:top]:
        top_k.append(cate)
    return top_k


if __name__ == '__main__':
    tf_idf_dict = load_data("./data/news.json")
    test_str = "新能源汽车"
    top_list = search_engine(test_str, tf_idf_dict)
    print(top_list)
