# -*- coding: utf-8 -*-

""" Deep Neural Network for MNIST dataset classification task using 
a highway network

References:

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    [https://arxiv.org/abs/1505.00387](https://arxiv.org/abs/1505.00387)

"""
from __future__ import division, print_function, absolute_import

import zqtflearn

# Data loading and preprocessing
import zqtflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

# Building deep neural network
input_layer = zqtflearn.input_data(shape=[None, 784])
dense1 = zqtflearn.fully_connected(input_layer, 64, activation='elu',
                                   regularizer='L2', weight_decay=0.001)
                 
                 
#install a deep network of highway layers
highway = dense1                              
for i in range(10):
    highway = zqtflearn.highway(highway, 64, activation='elu',
                                regularizer='L2', weight_decay=0.001, transform_dropout=0.8)
                              
                              
softmax = zqtflearn.fully_connected(highway, 10, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = zqtflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = zqtflearn.metrics.Top_k(3)
net = zqtflearn.regression(softmax, optimizer=sgd, metric=top_k,
                           loss='categorical_crossentropy')

# Training
model = zqtflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
          show_metric=True, run_id="highway_dense_model")
