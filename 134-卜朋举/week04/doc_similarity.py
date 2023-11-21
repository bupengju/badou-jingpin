import collections
import json

import numpy as np

from calc_tfidf import calc_tf_idf
from calc_tfidf import calc_word_doc_count
from calc_tfidf import tokenization


def load_data(json_path, query):
    news_list = json.load(open(json_path, "r", encoding="utf-8"))
    corpus = collections.defaultdict(list)
    corpus["query"].append(query)
    for news in news_list:
        corpus[news["title"]].append(news["content"])

    token_dict = tokenization(corpus)
    word_count_dict, doc_count_dict = calc_word_doc_count(token_dict)
    tf_idf_dict = calc_tf_idf(word_count_dict, doc_count_dict)
    return tf_idf_dict, news_list


def gen_doc_vec(query, tf_idf_dict, news_list, top=10):
    doc_vec = collections.defaultdict(list)
    for news in news_list:
        for _, score in sorted(tf_idf_dict[news["title"]].items(), key=lambda x: x[1], reverse=True)[:top]:
            doc_vec[news["content"]].append(score)

    for _, score in sorted(tf_idf_dict["query"].items(), key=lambda x: x[1], reverse=True)[:top]:
        doc_vec[query].append(score)

    if len(doc_vec[query]) < top:
        for _ in range(top - len(doc_vec[query])):
            doc_vec[query].append(0)

    return doc_vec


def cosine_similarity(vec1, vec2):
    sim = np.dot(vec1, vec2) / (np.sqrt(np.sum(vec1 ** 2)) * np.sqrt(np.sum(vec2 ** 2)))
    return sim


def get_most_similarity_doc(query, doc_vec_dict, top=3):
    result = {}
    for doc, vec2 in doc_vec_dict.items():
        if doc == query:
            continue
        result[doc] = cosine_similarity(np.array(doc_vec_dict[query]), np.array(vec2))

    most_sim_doc = []
    for doc, _ in sorted(result.items(), key=lambda x: x[1], reverse=True)[:top]:
        most_sim_doc.append(doc)

    return most_sim_doc


if __name__ == '__main__':
    query = "一系列在斑马线上发生的交通事故引起来全社会的关注"
    tf_idf_dict, news_list = load_data("./data/news.json", query)
    doc_vec_dict = gen_doc_vec(query, tf_idf_dict, news_list)
    result = get_most_similarity_doc(query, doc_vec_dict)
    for doc in result:
        print(doc)
