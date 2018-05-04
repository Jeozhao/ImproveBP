#-*-coding:utf-8-*


import mnist_loader
import d_network
import matplotlib.pyplot as plt




# 将数据集拆分成三个集合：训练集、交叉验证集、测试集
training_data, training_data_for_test, validation_data, test_data = mnist_loader.load_data_wrapper()


# 生成神经网络对象，神经网络结构为三层，每层节点数依次为（784, 30, 10）
net = d_network.Network([784, 80, 10])


# 用（mini-batch）梯度下降法训练神经网络（权重与偏移），并生成测试结果。
# 训练回合数=200, 用于随机梯度下降法的最小样本数=10，学习率=3.0
net.SGD(training_data, 200, 20, 2.0, test_data=test_data)

