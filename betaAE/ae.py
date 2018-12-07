'''
vae.py
contains the setup for autoencoders.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from read_data import MatterportDataset3DBBOX
from read_data import VRDataSet_AE
from read_data import ToyDatasetShapes_AE

class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))

def test():
    import os
    import numpy as np
    from PIL import Image
    from tensorflow.python.keras.preprocessing.image import load_img

    from models import Darknet19Encoder, Darknet19Decoder

    inputShape = (256, 256, 3)
    batchSize = 6
    latentSize = 50

    # dataSet = MatterportDataset3DBBOX(shuffle=False, repeat=False, batch_size=batchSize).dataset
    dataSet = ToyDatasetShapes_AE(batch_size=batchSize, output_size=256).dataset
    img = load_img("/d/PhD/AnchorModel/ToyDataShapes/sketch_181113a/{}.jpg".format(120), target_size=inputShape[:-1])
    img.show()

    img = np.array(img, dtype=np.float32) / 255 
    img = np.array([img]*batchSize) # make fake batches to improve GPU utilization
    print(img[0])
    # This is how you build the autoencoder
    encoder = Darknet19Encoder(inputShape, batchSize, latentSize, 'bvae', beta=69, capacity=5, randomSample=True)
    decoder = Darknet19Decoder(inputShape, batchSize, latentSize)
    bvae = AutoEncoder(encoder, decoder)

    bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
    
    counter = 0
    #bvae.ae.load_weights("./toydata_weights")
    while(True):
        bvae.ae.fit(dataSet, steps_per_epoch=100,  
                    epochs=5)

        input = bvae.ae.get_input_at(0)
        
        bvae.ae.save_weights('./toydata_weights')

        # example retrieving the latent vector
        latentVec = bvae.encoder.predict(img)[0]
        if(counter % 1 == 0):
                pred = bvae.ae.predict(img) # get the reconstructed image
                pred[pred > 1] = 1 # clean it up a bit
                pred[pred < -0] = 0
                pred = np.uint8((pred)* 255) # convert to regular image values
                print(pred[0])
                pred = Image.fromarray(pred[0])
                pred.show()  # display popup
        counter += 1
if __name__ == "__main__":
    test()