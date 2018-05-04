#coding=utf-8
#-*-coding:utf-8-*

#
import mnist_loader
import network
import matplotlib.pyplot as plt

# 将数据集拆分成三个集合：训练集、交叉验证集、测试集
training_data, training_data_for_test,  validation_data, test_data = mnist_loader.load_data_wrapper()

# 生成神经网络对象，神经网络结构为三层，每层节点数依次为（784, 30, 10）
net = network.Network([784, 80, 10])

# 用（mini-batch）梯度下降法训练神经网络（权重与偏移），并生成测试结果。
# 训练回合数=200, 用于随机梯度下降法的最小样本数=10，学习率=3.0
net.SGD(training_data, training_data_for_test, 5, 10, 2.0, test_data=test_data)
train_error_rate = net.train_errorRate
val_error_rate = net.val_errorRate
plt.figure(figsize=(10,4))
l1,  = plt.plot(range(len(train_error_rate)), train_error_rate, linewidth=2, marker='o', color='r', markersize=6, label='train-acc')
l2,  = plt.plot(range(len(val_error_rate)), val_error_rate, linewidth=2, marker='s', color='b', markersize=6, label='val-acc')
plt.legend(loc='upper left')
plt.show()
end = 1
