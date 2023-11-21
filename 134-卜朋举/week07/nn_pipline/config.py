config = {
    "model_path": "../out",
    "train_data_path": "../data/train_tag_news.json",
    "test_data_path": "../data/valid_tag_news.json",
    "pretrain_model_path": "../data/bert-base-chinese",
    "vocab_path": "../data/chars.txt",
    # "fasttext", "textrnn", "rnn", "lstm", "textcnn", "gatedcnn", "bert", "bertlstm", "bertcnn", "bertmidlayer"
    "model_type": "bertmidlayer",
    "max_length": 20,
    "hidden_size": 32,
    "num_layers": 2,
    "kernel_size": 3,
    "pooling_style": "max",
    "epoch": 10,
    "batch_size": 32,
    "lr": 0.001,
    "optim": "adam",
    "seed": 2023
}
