from pathlib import Path

from gensim.models import Word2Vec
import jieba


def train_w2v(corpus_path, output):
    sentences = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            sentences.append(jieba.lcut(line.strip()))
    model = Word2Vec(sentences, vector_size=32)
    model.save(Path(output).joinpath("w2v.model").as_posix())


def get_similarity_word(word, model_path):
    model = Word2Vec.load(model_path)
    result = None
    try:
        result = model.wv.most_similar(word)
    except:
        print("输入词不存在")
    return result


if __name__ == '__main__':
    train_w2v("./data/corpus.txt", "./out")
    res = get_similarity_word("北京", "./out/w2v.model")
    if res:
        print(res)
