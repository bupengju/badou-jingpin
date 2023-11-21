import collections
import json
import re

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
    return tf_idf_dict, news_list


def gen_abstract(output, tf_idf_dict, news_list, top=5):
    result = []
    for news in news_list:
        sentences = re.split("。|？|!", news["content"])
        sentences_tfidf = {}
        for sentence in sentences:
            words = jieba.lcut(sentence)
            for word in words:
                sentences_tfidf[sentence] = sentences_tfidf.get(sentence, 0) + tf_idf_dict[news["title"]].get(word, 0)

        abstract = []
        for sentence, _ in sorted(sentences_tfidf.items(), key=lambda x: x[1], reverse=True)[:top]:
            abstract.append(sentence)

        result.append({"title": news["title"], "content": news["content"], "abstract": ",".join(abstract)})

    result = json.dumps(result, ensure_ascii=False, indent=2)
    with open(output, "w", encoding="utf-8") as f:
        f.write(result)


if __name__ == '__main__':
    tf_idf_dict, news_list = load_data("./data/news.json")
    gen_abstract("./out/abstract.json", tf_idf_dict, news_list)
