'''
model_utils.py
contains custom layers, etc. for building models.

created by shadySource
THE UNLICENSE

potential edits by
Cagri Hakan Zaman
zaman@mit.edu

'''

import tensorflow as tf
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
                                            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
                                            Reshape, GlobalAveragePooling2D, Layer)
from tensorflow.python.keras import backend as K


class ConvBnLRelu(object):
    def __init__(self, filters, kernelSize, strides=1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    # retunr conv + bn + leaky_relu

    def __call__(self, net):
        net = Conv2D(self.filters, self.kernelSize, strides=self.strides, padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        return net


class SampleLayer(Layer):
    '''
    Keras Layer to grab a random sample from a distribution 
    (by multiplication) Computes "(normal)*stddev + mean" for the vae sampling operation
    (written for tf backend)

    Additionally, 
        Applies regularization to the latent space representation.
        Can perform standard regularization or B-VAE regularization

    call:
        pass in mean then stddev layers to sample from the distrubtion
        ex.
            sample = SampleLayer('bvae',16)([mean,stddev])
    '''

    def __init__(self, latent_regularizer='bvae', beta=100., capacity=0., randomSample=True, **kwargs):

        '''
            args:
            ------
            latent_regularizer: str
                Either 'bvae', 'vae', or 'no'
                Determines whether regularization is applied
                to the latent space representation
            beta: float
                beta>1, used for 'bvae' latent_regularizer,
                (Unused if 'bvae' is not selected)
            capacity: float
                used for 'bvae' to try to break input down to a set number of basis
                (e.g. at 25, the network will try to use 25 dimensions of the latent space)
                (unused if 'bvae' is not selected)
            randomSample:bool
                whether or not to use random sampling when selecting from distrubtion.
                if false, the latent vector equals the mean, essentially turning this into a 
                standard autoencoder
            ------
            ex.
                sample = SampleLayer('bvae',16)([mean,stddev])
        '''

        self.reg=latent_regularizer
        self.beta= beta
        self.capacity = capacity
        self.random = randomSample
        super(SampleLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        # save the shape for distribution sampling
        self.shape = input_shape[0]
        super(SampleLayer, self).build(input_shape)  # needed for layers
    
    def call(self, x):
        if len(x) != 2:
            raise Exception('input layers must be a list: mean and stddev')
        if len(x[0].shape) != 2 or len(x[1].shape) != 2:
            raise Exception('input shape is not a vector [batchSize, latentSize')
        
        mean = x[0]
        stddev = x[1]

        if self.reg == 'bvae':
            # kl divergence:
            latent_loss = -0.5*K.mean(1+stddev - K.square(mean) - K.exp(stddev), axis=-1)
            
            # use beta to force less usage of vector space:
            # also try to use <capacity> dimensions of the space
            latent_loss = self.beta * K.abs(latent_loss - self.capacity / self.shape.as_list()[1])
            self.add_loss(latent_loss, x)
        elif self.reg == 'vae':
            # kl divergence
            latent_loss = -0.5*K.mean(1+stddev - K.square(mean) - K.exp(stddev), axis=-1)
            self.add_loss(latent_loss, x)
        epsilon = K.random_normal(shape=self.shape, mean=0., stddev=1.)

        if self.random:
            # 'reparametrization trick':
            return mean + K.exp(stddev)*epsilon
        else:
            return mean + 0 * stddev  # Keras needs the *0 so the gradient is not none.
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]