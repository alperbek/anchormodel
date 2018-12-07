from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
from read_data import ToyDatasetShapes_AE
tfe = tf.contrib.eager
tf.enable_eager_execution()

# SIMPLE CONVULATIONAL VAE MODEL


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = 28
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.input_dim, self.input_dim, 3)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim)                

            ]
        )
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='SAME',
                    activation=tf.nn.relu),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='SAME',
                    activation=tf.nn.relu),
                tf.keras.layers.Conv2DTranspose(
                    filters=3,
                    kernel_size=3,
                    strides=(1, 1),
                    padding='SAME'
                )
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

# CALCULATE LOSS AND GRADIENTS



def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  return tape.gradient(loss, model.trainable_variables), loss

optimizer = tf.train.AdamOptimizer(1e-4)

def apply_gradients(optimizer, gradients, variables, global_step=None):
  optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
# TRAINING and IMAGE GENERATION

epochs = 50000
latent_dim = 50
num_examples_to_generate = 16


random_vector_for_generation = tf.random_normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)
model.load_weights("./bae2_toy_data_11Kepoch.ckpt")

def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

generate_and_save_images(model, 0, random_vector_for_generation)

# GET MNIST DATA

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

train_images /= 255.
test_images /= 255.

train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.


TRAIN_BUF = 5000
BATCH_SIZE = 100

TEST_BUF = 1000

#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

#test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

train_dataset = ToyDatasetShapes_AE(batch_size=BATCH_SIZE).dataset
test_dataset = ToyDatasetShapes_AE(batch_size=BATCH_SIZE).test_dataset

for epoch in range(1, epochs+1):
    start_time = time.time()
    for train_x,train_y in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
    end_time = time.time()

    if epoch % 20 == 0:
        loss = tfe.metrics.Mean()
        for test_x,test_y in test_dataset.make_one_shot_iterator():
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch {}'.format(epoch,
                                                                                      elbo, 
                                                                            end_time - start_time))
        #generate_and_save_images(model, epoch, random_vector_for_generation)
        model.save_weights("./bae2_toy_data_50Kepoch.ckpt")

generate_and_save_images(model,epoch,random_vector_for_generation)