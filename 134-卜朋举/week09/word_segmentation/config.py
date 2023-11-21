config = {
    "vocab_path": "../data/chars.txt",
    "corpus_path": "../data/corpus.txt",
    "model_path": "../out/bert.pth",
    "model_type": "bert",  # rnn, lstm, gru, bert
    "max_length": 50,
    "embed_size": 32,
    "hidden_size": 64,
    "num_layer": 2,
    "epoch": 10,
    "batch_size": 32,
    "lr": 1e-3,
    "class_num": 2
}