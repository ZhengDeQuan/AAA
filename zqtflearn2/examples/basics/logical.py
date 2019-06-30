# -*- coding: utf-8 -*-
"""
Simple Example to train logical operators
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import zqtflearn

# Logical NOT operator
X = [[0.], [1.]]
Y = [[1.], [0.]]

# Graph definition
with tf.Graph().as_default():
    g = zqtflearn.input_data(shape=[None, 1])
    g = zqtflearn.fully_connected(g, 128, activation='linear')
    g = zqtflearn.fully_connected(g, 128, activation='linear')
    g = zqtflearn.fully_connected(g, 1, activation='sigmoid')
    g = zqtflearn.regression(g, optimizer='sgd', learning_rate=2.,
                             loss='mean_square')

    # Model training
    m = zqtflearn.DNN(g)
    m.fit(X, Y, n_epoch=100, snapshot_epoch=False)

    # Test model
    print("Testing NOT operator")
    print("NOT 0:", m.predict([[0.]]))
    print("NOT 1:", m.predict([[1.]]))

# Logical OR operator
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y = [[0.], [1.], [1.], [1.]]

# Graph definition
with tf.Graph().as_default():
    g = zqtflearn.input_data(shape=[None, 2])
    g = zqtflearn.fully_connected(g, 128, activation='linear')
    g = zqtflearn.fully_connected(g, 128, activation='linear')
    g = zqtflearn.fully_connected(g, 1, activation='sigmoid')
    g = zqtflearn.regression(g, optimizer='sgd', learning_rate=2.,
                             loss='mean_square')

    # Model training
    m = zqtflearn.DNN(g)
    m.fit(X, Y, n_epoch=100, snapshot_epoch=False)

    # Test model
    print("Testing OR operator")
    print("0 or 0:", m.predict([[0., 0.]]))
    print("0 or 1:", m.predict([[0., 1.]]))
    print("1 or 0:", m.predict([[1., 0.]]))
    print("1 or 1:", m.predict([[1., 1.]]))

# Logical AND operator
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y = [[0.], [0.], [0.], [1.]]

# Graph definition
with tf.Graph().as_default():
    g = zqtflearn.input_data(shape=[None, 2])
    g = zqtflearn.fully_connected(g, 128, activation='linear')
    g = zqtflearn.fully_connected(g, 128, activation='linear')
    g = zqtflearn.fully_connected(g, 1, activation='sigmoid')
    g = zqtflearn.regression(g, optimizer='sgd', learning_rate=2.,
                             loss='mean_square')

    # Model training
    m = zqtflearn.DNN(g)
    m.fit(X, Y, n_epoch=100, snapshot_epoch=False)

    # Test model
    print("Testing AND operator")
    print("0 and 0:", m.predict([[0., 0.]]))
    print("0 and 1:", m.predict([[0., 1.]]))
    print("1 and 0:", m.predict([[1., 0.]]))
    print("1 and 1:", m.predict([[1., 1.]]))

'''
Going further: Graph combination with multiple optimizers
Create a XOR operator using product of NAND and OR operators
'''
# Data
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y_nand = [[1.], [1.], [1.], [0.]]
Y_or = [[0.], [1.], [1.], [1.]]

# Graph definition
with tf.Graph().as_default():
    # Building a network with 2 optimizers
    g = zqtflearn.input_data(shape=[None, 2])
    # Nand operator definition
    g_nand = zqtflearn.fully_connected(g, 32, activation='linear')
    g_nand = zqtflearn.fully_connected(g_nand, 32, activation='linear')
    g_nand = zqtflearn.fully_connected(g_nand, 1, activation='sigmoid')
    g_nand = zqtflearn.regression(g_nand, optimizer='sgd',
                                  learning_rate=2.,
                                  loss='binary_crossentropy')
    # Or operator definition
    g_or = zqtflearn.fully_connected(g, 32, activation='linear')
    g_or = zqtflearn.fully_connected(g_or, 32, activation='linear')
    g_or = zqtflearn.fully_connected(g_or, 1, activation='sigmoid')
    g_or = zqtflearn.regression(g_or, optimizer='sgd',
                                learning_rate=2.,
                                loss='binary_crossentropy')
    # XOR merging Nand and Or operators
    g_xor = zqtflearn.merge([g_nand, g_or], mode='elemwise_mul')

    # Training
    m = zqtflearn.DNN(g_xor)
    m.fit(X, [Y_nand, Y_or], n_epoch=400, snapshot_epoch=False)

    # Testing
    print("Testing XOR operator")
    print("0 xor 0:", m.predict([[0., 0.]]))
    print("0 xor 1:", m.predict([[0., 1.]]))
    print("1 xor 0:", m.predict([[1., 0.]]))
    print("1 xor 1:", m.predict([[1., 1.]]))
