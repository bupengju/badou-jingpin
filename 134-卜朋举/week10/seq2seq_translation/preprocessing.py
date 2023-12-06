import unicodedata
import re
import json
from pathlib import Path

from tqdm import tqdm

from config import config


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def build_vocab(data_path, eng_vocab_path, fra_vocab_path):
    eng_vocab, fra_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}, {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building vocab", unit=" lines"):
            eng_sentence, fra_sentence = line.strip().split("\t")[:2]
            eng_sentence, fra_sentence = normalize_string(eng_sentence), normalize_string(fra_sentence)

            for eng_word in eng_sentence.split(" "):
                if eng_word not in eng_vocab:
                    eng_vocab[eng_word] = len(eng_vocab)
            
            for fra_word in fra_sentence.split(" "):
                if fra_word not in fra_vocab:
                    fra_vocab[fra_word] = len(fra_vocab)

    if not Path(eng_vocab_path).parent.exists():
        Path(eng_vocab_path).parent.mkdir(parents=True)
    with open(eng_vocab_path, "w", encoding="utf-8") as f:
        json.dump(eng_vocab, f, indent=4, ensure_ascii=False)
    
    if not Path(fra_vocab_path).parent.exists():
        Path(fra_vocab_path).parent.mkdir(parents=True)
    with open(fra_vocab_path, "w", encoding="utf-8") as f:
        json.dump(fra_vocab, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    build_vocab(config["data_path"], config["eng_vocab_path"], config["fra_vocab_path"])
