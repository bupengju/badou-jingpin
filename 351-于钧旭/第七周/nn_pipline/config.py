# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/文本分类练习.csv",
    "valid_data_path": "data/文本分类练习.csv",
    "vocab_path":"chars.txt",
    "model_type":"lstm",
    "max_length": 100,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 64,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"F:\Desktop\work_space\pretrain_models\bert-base-chinese",
    "seed": 987
}