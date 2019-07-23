from autoencoder_classes.autoEncoder import AutoEncoder
from keras import layers
import keras
from keras import backend as K
from keras.models import Model
import numpy as np
import os


class CNN_ConvLatentSpace(AutoEncoder):

    def __init__(self, img_shape, latent_space_dims, batch_size):
        super().__init__(img_shape, latent_space_dims, batch_size)

        self.model_type_flag = 'cnn_lts'


    def define_model(self):
        input_img = layers.Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(self.latent_space_dims, (3, 3), activation='relu', padding='same')(x)
        latent_space = layers.MaxPooling2D((2, 2), padding='same')(x)

        encoder = Model(input_img, latent_space, name='Encoder')
        encoder.summary()

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        decoder_inputs = layers.Input(shape=K.int_shape(latent_space)[1:])
        x = layers.Conv2D(self.latent_space_dims, (3, 3), activation='relu', padding='same')(decoder_inputs)
        x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded_img = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        decoder = Model(decoder_inputs, decoded_img, name='decoder_model')
        decoder.summary()
        z_decoded = decoder(latent_space)

        AE = Model(input_img, z_decoded)
        AE.compile(optimizer='adadelta', loss='binary_crossentropy')
        AE.summary()

        encoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
        decoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

        self.model = AE
        self.encoder = encoder
        self.decoder = decoder
        self.define_flag = True


    def load_weights(self, full_path):
        self.define_model()
        self.model.load_weights(os.path.join(full_path, 'weights_model.h5'))
        self.encoder.load_weights(os.path.join(full_path, 'weights_encoder_model.h5'))
        self.decoder.load_weights(os.path.join(full_path, 'weights_decoder_model.h5'))

        print('LOAD WEIGHTS COMPLETE')

if __name__ == '__main__':

    CNN_AE = CNN_ConvLatentSpace(img_shape=(28, 28, 1), latent_space_dims=10, batch_size=16)
    CNN_AE.data_prep(keep_labels=[0,1,2,3,4,5,6,7,8,9])
    CNN_AE.define_model()
    CNN_AE.train(epochs=10)
    CNN_AE.inspect_model(dim_reduction_model='tsne', dimensions=2)
    CNN_AE.save(name='cnn_convlts_all_labels_10D', save_type='weights')