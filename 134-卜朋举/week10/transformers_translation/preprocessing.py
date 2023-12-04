import re
from pathlib import Path


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"(['.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z'.!?]+", r" ", s)
    return s


def parse_lines(lines, col_num):
    vocab = set()
    for line in lines:
        words = line.strip().split("\t")
        words = normalize_string(words[col_num])
        for word in words.strip().split(" "):
            vocab.add(word)
    vocab = list(vocab)
    vocab.sort()
    vocab.insert(0, "<pad>")
    vocab.insert(1, "<sos>")
    vocab.insert(2, "<eos>")
    vocab.insert(3, "<unk>")
    return vocab


def build_vocab(corpus_path, out_path):
    eng_path = Path(out_path).joinpath("eng_vocab.txt").as_posix()
    fra_path = Path(out_path).joinpath("fra_vocab.txt").as_posix()
    with open(corpus_path, "r", encoding="utf-8") as f, \
        open(eng_path, "w", encoding="utf-8") as f1_out, \
        open(fra_path, "w", encoding="utf-8") as f2_out:
        lines = f.readlines()
        for ch1 in parse_lines(lines, 0):
            f1_out.write(ch1 + "\n")
            
        for ch2 in parse_lines(lines, 1):
            f2_out.write(ch2 + "\n")


if __name__ == "__main__":
    build_vocab("./data/translation/eng-fra.txt", "./data/translation/")
