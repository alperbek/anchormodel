# VARIATIONAL AUTOENCODER EXAMPLE
# Originally by Kingma&Welling [1312.6114]
# Link: https://github.com/oduerr/dl_tutorial/tree/master/tensorflow/vae
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from read_data import MatterportDataset3DBBOX
from tensorflow.examples.tutorials.mnist import input_data

ver = tf.__version__
print("Tensorflow Version {}".format(ver))
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
n_samples = mnist.train.num_examples
print("Number of samples {} Shape of y[{}] Shape of X[{}]"
      .format(n_samples, mnist.train.labels.shape, mnist.train.images.shape))

# plt.imshow(np.reshape(-mnist.train.images[4242],(28,28)),interpolation='none', cmap=plt.get_cmap('gray'))
# print(mnist.train.images.min())


def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

n_z = 2  # Dimension of the latent space
# Input
x = tf.placeholder("float", shape=[None, 28*28])  # Batchsize x Number of Pixels
y_ = tf.placeholder("float", shape=[None, 10])    # Batchsize x 10 (one hot encoded)

batch_size = 64

dataSet = MatterportDataset3DBBOX(shuffle=True, batch_size=batch_size)
images, bboxes, classes, num_instance = dataSet.nextBatch()
images = ((tf.reshape(images, [batch_size, 28*28]))/255.0)

# First hidden layer
# W_fc1 = weights([784,500])
W_fc1 = weights([28*28, 500])  # Matterport
b_fc1 = bias([500])
h_1 = tf.nn.softplus(tf.matmul(images, W_fc1) + b_fc1)

# Second hidden layer
W_fc2 = weights([500, 501])  # 501 and not 500 to spot errors
b_fc2 = bias([501])
h_2 = tf.nn.softplus(tf.matmul(h_1, W_fc2) + b_fc2)

# Parameters for the Gaussian
z_mean = tf.add(tf.matmul(h_2, weights([501, n_z])), bias([n_z]))
z_log_sigma_sq = tf.add(tf.matmul(h_2, weights([501, n_z])), bias([n_z]))

eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32)  #  Adding a random number
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))   # The sampled z

W_fc1_g = weights([n_z, 500])
b_fc1_g = bias([500])
h_1_g = tf.nn.softplus(tf.matmul(z, W_fc1_g) + b_fc1_g)

W_fc2_g = weights([500, 501])
b_fc2_g = bias([501])
h_2_g = tf.nn.softplus(tf.matmul(h_1_g, W_fc2_g) + b_fc2_g)

x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(h_2_g,  weights([501, 784])), bias([784])))

# Defining the loss function.

reconstr_loss = -tf.reduce_sum(images * tf.log(1e-10 + x_reconstr_mean) + (1-x) * tf.log(1e-10 + 1 - x_reconstr_mean), 1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch.

# USE ADAM OPTIMIZER

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


runs = 10
init = tf.initialize_all_variables()


check_point_file = "model/model.ckpt"


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, check_point_file)
    sess.run(dataSet.getDataIterator().initializer, feed_dict={dataSet.filenames:dataSet.training_filenames})

    print("Model restored.")
    # x_sample = sess.run(images)[0]
    x_sample, x_reconstruct, z_vals, z_mean_val, z_log_sigma_sq_val = sess.run((images, x_reconstr_mean, z, z_mean, z_log_sigma_sq))

    print(z_mean)

    plt.figure(figsize=(8, 12))

    for i in range(5):
        plt.subplot(5, 3, 3*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1,  interpolation='none', cmap=plt.get_cmap('gray'))
        plt.title("Test input")
        
        # plt.colorbar()
        plt.subplot(5, 3, 3*i + 2)
        plt.scatter(z_vals[:, 0], z_vals[:, 1], c='gray', alpha=0.5)
        plt.scatter(z_mean_val[i, 0], z_mean_val[i, 1], c='green', s=64, alpha=0.5)
        plt.scatter(z_vals[i, 0], z_vals[i, 1], c='blue', s=16, alpha=0.5)
       
        plt.xlim((-3, 3))
        plt.ylim((-3, 3))
        plt.title("Latent Space")
        
        plt.subplot(5, 3, 3*i + 3)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, interpolation='none',cmap=plt.get_cmap('gray'))
        plt.title("Reconstruction")
        #plt.colorbar()
    plt.tight_layout()
    plt.show()

nx = ny = 30
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)
canvas = np.empty((28*ny, 28*nx))
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, check_point_file)
    d = np.zeros([batch_size, 2],dtype='float32')

    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            d[0] = z_mu
            x_mean = sess.run(x_reconstr_mean, feed_dict={z: d})
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))        
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", vmin=0, vmax=1, interpolation='none', cmap=plt.get_cmap('gray'))
plt.tight_layout()
plt.show()

