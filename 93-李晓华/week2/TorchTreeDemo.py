# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个3维向量，如果第一个值大于其他，认为是分类1，如果第二个值大于其他，认为是分类2.其他为分类 0.

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层  3*3
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # x  20*3
        y_pred = self.activation(x)  # 20*3
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个3维向量，如果第一个值大于其他，认为是分类1，如果第二个值大于其他，认为是分类2.其他为分类 0.
def build_sample():
    x = np.random.random(3)
    if x[0] > x[1] and x[0] > x[2]:
        return x, 1
    elif x[1] > x[0] and x[1] > x[2]:
        return x, 2
    else:
        return x, 0


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 30000  # 每轮训练总共训练的样本总数
    input_size = 3  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 3
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        result = torch.argmax(result, dim=1)
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec, res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.89349776, 0.59416669, 0.92579291],
                [0.47889086, 0.15229675, 0.31082123],
                [0.84890681, 0.94963533, 0.65758807]
                ]
    predict("model.pth", test_vec)




