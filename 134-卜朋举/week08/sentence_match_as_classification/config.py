config = {
    "model_path": "../out/sentence_match_as_classification",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "schema_path": "../data/schema.json",
    "vocab_path": "../data/chars.txt",
    "epoch": 20,
    "batch_size": 32,
    "lr": 1e-3,
    "hidden_size": 64,
    "maxlength": 20,
    "optim": "adam"
}
