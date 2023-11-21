config = {
    "train_data_path": "data/add_punctuation/train_corpus.txt",
    "test_data_path": "data/add_punctuation/valid_corpus.txt",
    "vocab_path": "data/bert-base-chinese",
    "schema_path": "data/add_punctuation/schema.json",
    "model_path": "out/add_punctuation/bert.pth",
    "epoch": 2,
    "lr": 1e-3,
    "max_length": 50,
    "batch_size": 32,
    "class_num": 4,
    "use_crf": False,
}
