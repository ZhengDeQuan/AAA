# https://blog.csdn.net/marsjhao/article/details/72831021
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:56:30 2017

@author: marsjhao
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  # 输入节点数
OUTPUT_NODE = 10  # 输出节点数
LAYER1_NODE = 500  # 隐含层节点数
BATCH_SIZE = 100
LEARNING_RETE_BASE = 0.8  # 基学习率
LEARNING_RETE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化项的权重系数
TRAINING_STEPS = 10000  # 迭代训练次数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减系数


# 传入神经网络的权重和偏置，计算神经网络前向传播的结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 判断是否传入ExponentialMovingAverage类对象
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))
                            + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) \
               + avg_class.average(biases2)


# 神经网络模型的训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 定义神经网络结构的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE],
                                               stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE],
                                               stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算非滑动平均模型下的参数的前向传播的结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0, trainable=False)  # 定义存储当前迭代训练轮数的变量

    # 定义ExponentialMovingAverage类对象
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)  # 传入当前迭代轮数参数
    # 定义对所有可训练变量trainable_variables进行更新滑动平均值的操作op
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算滑动模型下的参数的前向传播的结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 定义交叉熵损失值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 定义L2正则化器并对weights1和weights2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization  # 总损失值

    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RETE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE, LEARNING_RETE_DECAY)
    # 定义梯度下降操作op，global_step参数可实现自加1运算
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss, global_step=global_step)
    # 组合两个操作op
    train_op = tf.group(train_step, variables_averages_op)
    '''
    # 与tf.group()等价的语句
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    '''
    # 定义准确率
    # 在最终预测的时候，神经网络的输出采用的是经过滑动平均的前向传播计算结果
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化回话sess并开始迭代训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 验证集待喂入数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 测试集待喂入数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training steps, validation accuracy'
                      ' using average model is %f' % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training steps, test accuracy'
              ' using average model is %f' % (TRAINING_STEPS, test_acc))


# 主函数
def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)


# 当前的python文件是shell文件执行的入口文件，而非当做import的python module。
if __name__ == '__main__':  # 在模块内部执行
    tf.app.run()  # 调用main函数并传入所需的参数list

