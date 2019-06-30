# -*- coding: utf-8 -*-

""" Deep Residual Network.

Applying a Deep Residual Network to CIFAR-10 Dataset classification task.

References:
    - K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image
      Recognition, 2015.
    - Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.

Links:
    - [Deep Residual Network](http://arxiv.org/pdf/1512.03385.pdf)
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""

from __future__ import division, print_function, absolute_import

import zqtflearn

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

# Data loading
from zqtflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()
Y = zqtflearn.data_utils.to_categorical(Y)
testY = zqtflearn.data_utils.to_categorical(testY)

# Real-time data preprocessing
img_prep = zqtflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = zqtflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

# Building Residual Network
net = zqtflearn.input_data(shape=[None, 32, 32, 3],
                           data_preprocessing=img_prep,
                           data_augmentation=img_aug)
net = zqtflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = zqtflearn.residual_block(net, n, 16)
net = zqtflearn.residual_block(net, 1, 32, downsample=True)
net = zqtflearn.residual_block(net, n - 1, 32)
net = zqtflearn.residual_block(net, 1, 64, downsample=True)
net = zqtflearn.residual_block(net, n - 1, 64)
net = zqtflearn.batch_normalization(net)
net = zqtflearn.activation(net, 'relu')
net = zqtflearn.global_avg_pool(net)
# Regression
net = zqtflearn.fully_connected(net, 10, activation='softmax')
mom = zqtflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = zqtflearn.regression(net, optimizer=mom,
                           loss='categorical_crossentropy')
# Training
model = zqtflearn.DNN(net, checkpoint_path='model_resnet_cifar10',
                      max_checkpoints=10, tensorboard_verbose=0,
                      clip_gradients=0.)

model.fit(X, Y, n_epoch=200, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnet_cifar10')
