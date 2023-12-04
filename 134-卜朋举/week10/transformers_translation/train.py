import torch
from torch.optim import AdamW
from transformers import get_scheduler
import torch.nn.functional as F

from loader import load_vocab
from loader import load_data
from model import Seq2SeqTransformer
from model import create_masks

# from model import generate_square_subsequent_mask
from config import config


def train():
    src_vocab = load_vocab(config["src_vocab_path"])
    trg_vocab = load_vocab(config["trg_vocab_path"])

    train_data = load_data(
        config["data_path"],
        src_vocab,
        trg_vocab,
        config["max_length"],
        config["batch_size"],
    )

    model = Seq2SeqTransformer(
        config["num_layers"],
        config["num_layers"],
        config["hidden_size"],
        config["n_heads"],
        len(src_vocab),
        len(trg_vocab),
        config["hidden_size"],
        config["dropout"],
    )

    optimizer = AdamW(model.parameters(), lr=config["lr"])

    scheduler = get_scheduler(
        "linear", optimizer, 0, config["num_epochs"] * len(train_data)
    )

    for epoch in range(config["num_epochs"]):
        model.train()
        for src, trg in train_data:
            src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_masks(
                src, trg
            )
            preds = model(
                src,
                trg[:, :-1],
                src_mask,
                trg_mask[:-1, :-1],
                src_padding_mask,
                trg_padding_mask[:, :-1],
                src_padding_mask,
            )
            ys = trg[:, 1:].contiguous().view(-1)
            optimizer.zero_grad()
            loss = F.cross_entropy(
                preds.view(-1, preds.size(-1)), ys, ignore_index=trg_vocab["<pad>"]
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), config["model_path"])


if __name__ == "__main__":
    train()
