import json

import torch
from transformers import BertTokenizer

from config import config


def predict(sentences):
    with open(config["schema_path"], "r", encoding="utf-8") as f:
        schema = json.load(f)
    idx2sign = {v: k for k, v in schema.items()}

    model = torch.load(config["model_path"])
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(config["vocab_path"])
    for sentence in sentences:
        out = tokenizer.encode(
            sentence, max_length=config["max_length"], pad_to_max_length=True)
        out = torch.tensor(out).unsqueeze(0)
        logits = model(out)
        selected = out["attention_mask"] == 1
        logits = logits[selected]
        pred = logits.argmax(dim=-1)
        for char in sentence:
            if char in idx2sign:
                print(idx2sign[char], end="")
            else:
                print(char, end="")
