config = {
    "data_path": "data/translation/eng-fra.txt",
    "src_vocab_path": "data/translation/eng_vocab.txt",
    "trg_vocab_path": "data/translation/fra_vocab.txt",
    "model_path": "out/transformer4translation.pth",
    "max_length": 50,
    "batch_size": 32,
    "num_epochs": 10,
    "lr": 1e-3,
    "num_layers": 1,
    "dropout": 0.1,
    "hidden_size": 512,
    "n_heads": 8,
}