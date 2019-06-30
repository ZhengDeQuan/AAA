# -*- coding: utf-8 -*-
"""
MNIST Classification using RNN over images pixels. A picture is
representated as a sequence of pixels, coresponding to an image's
width (timestep) and height (number of sequences).
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import zqtflearn

import zqtflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = np.reshape(X, (-1, 28, 28))
testX = np.reshape(testX, (-1, 28, 28))

net = zqtflearn.input_data(shape=[None, 28, 28])
net = zqtflearn.lstm(net, 128, return_seq=True)
net = zqtflearn.lstm(net, 128)
net = zqtflearn.fully_connected(net, 10, activation='softmax')
net = zqtflearn.regression(net, optimizer='adam',
                           loss='categorical_crossentropy', name="output1")
model = zqtflearn.DNN(net, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1, validation_set=0.1, show_metric=True,
          snapshot_step=100)
