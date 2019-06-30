# -*- coding: utf-8 -*-
""" DCGAN Example

Use a deep convolutional generative adversarial network (DCGAN) to generate
digit images from a noise distribution.

References:
    - Unsupervised representation learning with deep convolutional generative
    adversarial networks. A Radford, L Metz, S Chintala. arXiv:1511.06434.

Links:
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434).

"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zqtflearn

# Data loading and preprocessing
import zqtflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data()
X = np.reshape(X, newshape=[-1, 28, 28, 1])

z_dim = 200 # Noise data points
total_samples = len(X)


# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = zqtflearn.fully_connected(x, n_units=7 * 7 * 128)
        x = zqtflearn.batch_normalization(x)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        x = zqtflearn.upsample_2d(x, 2)
        x = zqtflearn.conv_2d(x, 64, 5, activation='tanh')
        x = zqtflearn.upsample_2d(x, 2)
        x = zqtflearn.conv_2d(x, 1, 5, activation='sigmoid')
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = zqtflearn.conv_2d(x, 64, 5, activation='tanh')
        x = zqtflearn.avg_pool_2d(x, 2)
        x = zqtflearn.conv_2d(x, 128, 5, activation='tanh')
        x = zqtflearn.avg_pool_2d(x, 2)
        x = zqtflearn.fully_connected(x, 1024, activation='tanh')
        x = zqtflearn.fully_connected(x, 2)
        x = tf.nn.softmax(x)
        return x


# Input Data
gen_input = zqtflearn.input_data(shape=[None, z_dim], name='input_gen_noise')
input_disc_noise = zqtflearn.input_data(shape=[None, z_dim], name='input_disc_noise')
input_disc_real = zqtflearn.input_data(shape=[None, 28, 28, 1], name='input_disc_real')

# Build Discriminator
disc_fake = discriminator(generator(input_disc_noise))
disc_real = discriminator(input_disc_real, reuse=True)
disc_net = tf.concat([disc_fake, disc_real], axis=0)
# Build Stacked Generator/Discriminator
gen_net = generator(gen_input, reuse=True)
stacked_gan_net = discriminator(gen_net, reuse=True)

# Build Training Ops for both Generator and Discriminator.
# Each network optimization should only update its own variable, thus we need
# to retrieve each network variables (with get_layer_variables_by_scope).
disc_vars = zqtflearn.get_layer_variables_by_scope('Discriminator')
# We need 2 target placeholders, for both the real and fake image target.
disc_target = zqtflearn.multi_target_data(['target_disc_fake', 'target_disc_real'],
                                          shape=[None, 2])
disc_model = zqtflearn.regression(disc_net, optimizer='adam',
                                  placeholder=disc_target,
                                  loss='categorical_crossentropy',
                                  trainable_vars=disc_vars,
                                  batch_size=64, name='target_disc',
                                  op_name='DISC')

gen_vars = zqtflearn.get_layer_variables_by_scope('Generator')
gan_model = zqtflearn.regression(stacked_gan_net, optimizer='adam',
                                 loss='categorical_crossentropy',
                                 trainable_vars=gen_vars,
                                 batch_size=64, name='target_gen',
                                 op_name='GEN')

# Define GAN model, that output the generated images.
gan = zqtflearn.DNN(gan_model)

# Training
# Prepare input data to feed to the discriminator
disc_noise = np.random.uniform(-1., 1., size=[total_samples, z_dim])
# Prepare target data to feed to the discriminator (0: fake image, 1: real image)
y_disc_fake = np.zeros(shape=[total_samples])
y_disc_real = np.ones(shape=[total_samples])
y_disc_fake = zqtflearn.data_utils.to_categorical(y_disc_fake, 2)
y_disc_real = zqtflearn.data_utils.to_categorical(y_disc_real, 2)

# Prepare input data to feed to the stacked generator/discriminator
gen_noise = np.random.uniform(-1., 1., size=[total_samples, z_dim])
# Prepare target data to feed to the discriminator
# Generator tries to fool the discriminator, thus target is 1 (e.g. real images)
y_gen = np.ones(shape=[total_samples])
y_gen = zqtflearn.data_utils.to_categorical(y_gen, 2)

# Start training, feed both noise and real images.
gan.fit(X_inputs={'input_gen_noise': gen_noise,
                  'input_disc_noise': disc_noise,
                  'input_disc_real': X},
        Y_targets={'target_gen': y_gen,
                   'target_disc_fake': y_disc_fake,
                   'target_disc_real': y_disc_real},
        n_epoch=10)

# Create another model from the generator graph to generate some samples
# for testing (re-using same session to re-use the weights learnt).
gen = zqtflearn.DNN(gen_net, session=gan.session)

f, a = plt.subplots(4, 10, figsize=(10, 4))
for i in range(10):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[4, z_dim])
    g = np.array(gen.predict({'input_gen_noise': z}))
    for j in range(4):
        # Generate image from noise. Extend to 3 channels for matplot figure.
        img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                         newshape=(28, 28, 3))
        a[j][i].imshow(img)

f.show()
plt.draw()
plt.waitforbuttonpress()
