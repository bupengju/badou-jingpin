config = {
    "train_data_path": "data/sentence_bert/train.txt",
    "test_data_path": "data/sentence_bert/test.txt",
    "pretrained_path": "data/bert-base-chinese",
    "model_path": "out/sentence_bert/bert.pth",
    "epoch": 2,
    "lr": 1e-3,
    "device": "cpu",
    "hidden_size": 768,
    "dropout_rate": 0.1,
    "max_length": 50,
    "max_sentence_length": 10, # "max_sentence_length"
    "batch_size": 10,
    "class_num": 3,
}