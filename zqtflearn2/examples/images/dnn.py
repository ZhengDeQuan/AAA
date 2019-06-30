# -*- coding: utf-8 -*-

""" Deep Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""
from __future__ import division, print_function, absolute_import

import zqtflearn

# Data loading and preprocessing
import zqtflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

# Building deep neural network
input_layer = zqtflearn.input_data(shape=[None, 784])
dense1 = zqtflearn.fully_connected(input_layer, 64, activation='tanh',
                                   regularizer='L2', weight_decay=0.001)
dropout1 = zqtflearn.dropout(dense1, 0.8)
dense2 = zqtflearn.fully_connected(dropout1, 64, activation='tanh',
                                   regularizer='L2', weight_decay=0.001)
dropout2 = zqtflearn.dropout(dense2, 0.8)
softmax = zqtflearn.fully_connected(dropout2, 10, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = zqtflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = zqtflearn.metrics.Top_k(3)
net = zqtflearn.regression(softmax, optimizer=sgd, metric=top_k,
                           loss='categorical_crossentropy')

# Training
model = zqtflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")
