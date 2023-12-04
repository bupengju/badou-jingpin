import torch

from config import config
from loader import load_vocab
from loader import padding
from model import create_masks
from model import Seq2SeqTransformer


def predict(text):
    src_vocab = load_vocab(config["src_vocab_path"])
    trg_vocab = load_vocab(config["trg_vocab_path"])
    idx_2_char = {idx: char for char, idx in trg_vocab.items()}
    src_inputs_id = [
        src_vocab.get(char, src_vocab["<unk>"]) for char in text.split(" ")
    ]
    src_inputs_id = padding(
        src_inputs_id,
        config["max_length"],
        src_vocab["<pad>"],
        src_vocab["<sos>"],
        src_vocab["<eos>"],
    )

    trg_inputs_id = [trg_vocab["<sos>"]] + [trg_vocab["<pad>"]] * (
        config["max_length"] - 1
    )

    # Load the model
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
    model.load_state_dict(torch.load(config.get("model_path")))
    model.eval()

    for i in range(config["max_length"]):
        src_inputs_id_tensor = torch.LongTensor(src_inputs_id).unsqueeze(0)
        trg_inputs_id_tensor = torch.LongTensor(trg_inputs_id).unsqueeze(0)
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_masks(
            src_inputs_id_tensor, trg_inputs_id_tensor
        )
        with torch.no_grad():
            outputs = model(
                src_inputs_id_tensor,
                trg_inputs_id_tensor[:, :-1],
                src_mask,
                trg_mask[:-1, :-1],
                src_padding_mask,
                trg_padding_mask[:, :-1],
                src_padding_mask,
            )
        pred = outputs.argmax(dim=-1)[:, -1].item()
        trg_inputs_id[i + 1] = pred
        if pred == trg_vocab["<eos>"]:
            break
    
    print(trg_inputs_id)

    print(text, " -> ", end="")
    print(
        "".join(
            [
                idx_2_char[idx]
                for idx in trg_inputs_id
                if idx
                not in [trg_vocab["<pad>"], trg_vocab["<sos>"], trg_vocab["<eos>"]]
            ]
        )
    )


if __name__ == "__main__":
    predict("I'm OK")
    predict("Back off!")
