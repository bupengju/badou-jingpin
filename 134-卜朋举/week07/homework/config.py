config = {
    "model_path": "../out/homework",
    "data_path": "../data/文本分类练习.csv",
    "train_data_path": "../data/homework/train_comment.json",
    "test_data_path": "../data/homework/valid_comment.json",
    "pretrain_model_path": "../data/bert-base-chinese",
    "vocab_path": "../data/chars.txt",
    "train_ratio": 0.8,
    # "fasttext", "textrnn", "rnn", "lstm", "textcnn", "gatedcnn", "bert", "bertlstm", "bertcnn", "bertmidlayer"
    "model_type": "bertlstm",
    "max_length": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "kernel_size": 3,
    "pooling_style": "max",
    "epoch": 20,
    "batch_size": 32,
    "lr": 0.001,
    "optim": "adam",
    "seed": 2023
}
