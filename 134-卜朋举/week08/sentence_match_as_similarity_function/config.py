config = {
    "model_path": "../out/sentence_match_as_classification",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "schema_path": "../data/schema.json",
    "vocab_path": "../data/bert-base-chinese/vocab.txt",
    "vocab_size": 2000,
    "epoch": 2,
    "batch_size": 32,
    "lr": 1e-3,
    "hidden_size": 64,
    "maxlength": 20,
    "optim": "adam",
    "epoch_data_size": 100,
    "positive_sample_rate": 0.5
}
