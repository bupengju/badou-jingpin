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
规律：x是一个5维向量：
如果第1个数>80，则为第一类
如果60<第1个数<=80，则为第二类
如果40<第1个数<=60，则为第三类
如果20<第1个数<=40，则为第四类
如果0<第1个数<=20，则为第五类

"""

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    # 生成一个随机5维向量
    x = np.random.random(6)
    if x[0] > 0.8:
        return x, [1, 0, 0, 0, 0]
    elif 0.6 < x[0] <= 0.8:
        return x, [0, 1, 0, 0, 0]
    elif 0.4 < x[0] <= 0.6:
        return x, [0, 0, 1, 0, 0]
    elif 0.2 < x[0] <= 0.4:
        return x, [0, 0, 0, 1, 0]
    elif 0 < x[0] <= 0.2:
        return x, [0, 0, 0, 0, 1]


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)



# 自建一个 TorchModel 类，继承 nn.Module 模型
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(TorchModel, self).__init__()
        # 将6维的输入向量，映射为5个概率
        self.linear1 = nn.Linear(input_size, hidden_size1)  # 线性层
        self.linear2 = nn.Linear(hidden_size1, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵误差损失

    # 如果传入真实标签y，则返回loss值；如果没有传入真实标签y，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 5)
        y_pred = self.linear2(x)
        #y_pred = torch.softmax(x)  # (batch_size, 5) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果



# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    for i in range(5):
        num = 0
        for sub_y in y:
            if sub_y[i] == 1:
                num += 1
        print("本次预测集中共有%d个第%d类" % (num, i+1))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            max_val = max(y_p)
            for i in range(len(y_p)):
                if y_p[i] == max_val:
                    if i == 0 and y_t[0] == 1:
                        correct += 1  # 第一类判断正确
                    elif i == 1 and y_t[1] == 1:
                        correct += 1  # 第二类判断正确
                    elif i == 2 and y_t[2] == 1:
                        correct += 1  # 第三类判断正确
                    elif i == 3 and y_t[3] == 1:
                        correct += 1  # 第四类判断正确
                    elif i == 4 and y_t[4] == 1:
                        correct += 1  # 第五类判断正确
                    else:
                        wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 6  # 输入向量维度
    hidden_size1 = 12
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, hidden_size1)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
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
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "work-model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 6
    hidden_size1 = 12
    model = TorchModel(input_size, hidden_size1)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    #print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        result = softmax(result.numpy())
    for vec, res in zip(input_vec, result):
        a = 0
        max_val = max(res)
        for i in range(len(res)):
            if res[i] == max_val:
                a = i + 1
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, a, res[a-1]))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843, 0.1],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681, 0.1],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392, 0.1],
                [0.89349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894, 0.1]]
    predict("work-model.pth", test_vec)
