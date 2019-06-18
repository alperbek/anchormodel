#VARIATIONAL AUTOENCODER EXAMPLE
#Originally by Kingma&Welling [1312.6114]
## Link: https://github.com/oduerr/dl_tutorial/tree/master/tensorflow/vae
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from read_data import MatterportDataset3DBBOX
ver= tf.__version__
print ("Tensorflow Version {}".format(ver))


from tensorflow.examples.tutorials.mnist import input_data


batch_size = 64
mnist= input_data.read_data_sets('MNIST_data',one_hot=True)
n_samples=mnist.train.num_examples
# print("Number of samples {} Shape of y[{}] Shape of X[{}]"
#         .format(n_samples,mnist.train.labels.shape,mnist.train.images.shape))

# plt.imshow(np.reshape(-mnist.train.images[4242],(28,28)),interpolation='none', cmap=plt.get_cmap('gray'))
# print(mnist.train.images.min())

def weights(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

n_z = 2 #Dimension of the latent space
# Input
x = tf.placeholder("float", shape=[None, 28*28]) #Batchsize x Number of Pixels
y_ = tf.placeholder("float", shape=[None, 10])   #Batchsize x 10 (one hot encoded)


dataSet = MatterportDataset3DBBOX(shuffle=True, batch_size=batch_size)

images = ((tf.reshape(images,[batch_size,28*28]))/255.0)
# First hidden layer
W_fc1 = weights([28*28, 500])
b_fc1 = bias([500])
h_1   = tf.nn.softplus(tf.matmul(images, W_fc1) + b_fc1)

# Second hidden layer 
W_fc2 = weights([500, 501]) #501 and not 500 to spot errors
b_fc2 = bias([501])
h_2   = tf.nn.softplus(tf.matmul(h_1, W_fc2) + b_fc2)

# Parameters for the Gaussian
z_mean = tf.add(tf.matmul(h_2, weights([501, n_z])), bias([n_z]))
z_log_sigma_sq = tf.add(tf.matmul(h_2, weights([501, n_z])), bias([n_z]))


eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32) # Adding a random number
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))  # The sampled z

W_fc1_g = weights([n_z, 500])
b_fc1_g = bias([500])
h_1_g   = tf.nn.softplus(tf.matmul(z, W_fc1_g) + b_fc1_g)

W_fc2_g = weights([500, 501])
b_fc2_g = bias([501])
h_2_g   = tf.nn.softplus(tf.matmul(h_1_g, W_fc2_g) + b_fc2_g)

x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(h_2_g,  weights([501, 28*28])), bias([28*28])))

#Defining the loss function.

reconstr_loss = -tf.reduce_sum(images * tf.log(1e-10 + x_reconstr_mean) + (1-images) * tf.log(1e-10 + 1 - x_reconstr_mean), 1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch.

#USE ADAM OPTIMIZER

optimizer =  tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


runs=200
init =tf.initialize_all_variables()
saver= tf.train.Saver()
with tf.Session() as sess:
        sess.run(init)
        sess.run(dataSet.getDataIterator().initializer, feed_dict = {dataSet.filenames:dataSet.training_filenames})
        # batch_xs,_=mnist.train.next_batch(batch_size)
        # plt.imshow(np.reshape(batch_xs[0],[28,28]))
        # plt.show()
        # print(batch_xs[0])
        #print(batch_xs.shape)
        dd=sess.run([cost])
        print('Test run after starting {}'.format(dd))
        
        
        for epoch in range(runs):
                avg_cost=0
                total_batch=int(n_samples/batch_size)
                #Loop over all batches     
                for i in range(300):
                        _,d = sess.run((optimizer,cost))
                        avg_cost+=d/batch_size
                        if(i%10==0):
                                print("Epch:", '%04d' %(epoch+1), "Step: ", '%04d' %(i+1),"cost=","{:.9f}".format(avg_cost))
                avg_cost = avg_cost/300
                save_path=saver.save(sess,"model/model.ckpt")
                print("Model_saved in file: {}".format(save_path))
                print("Epch:", '%04d' %(epoch+1), "cost=","{:.9f}".format(avg_cost))