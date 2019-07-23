from autoencoder_classes.autoEncoder import AutoEncoder
from keras import layers
import keras
from keras import backend as K
from keras.models import Model
import numpy as np
import os


class CNN_DenseLatentSpace(AutoEncoder):


    def __init__(self, img_shape, latent_space_dims, batch_size):

        super().__init__(img_shape, latent_space_dims, batch_size)


    def define_model(self):
        # INPUT LAYER

        input_img = layers.Input(shape=self.img_shape)

        # ENCODER ==================================================================================================

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)


        shape_before_flattening = K.int_shape(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=50, activation='relu')(x)


        latent_vector = layers.Dense(units=self.latent_space_dims, name='Latent_space',activity_regularizer=layers.regularizers.l1(10e-5) )(x)
        encoder = Model(input_img, latent_vector, name='Encoder')
        encoder.summary()


        # DECODER =========================================================================================
        decoder_inputs = layers.Input(shape=K.int_shape(latent_vector)[1:])

        d = layers.Dense(units=50, activation='relu')(decoder_inputs)
        d = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(d)
        d = layers.Reshape(target_shape=shape_before_flattening[1:])(d)


        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(d)
        d = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(d)


        decoded_img = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                                    activation='sigmoid', name='decoded_img')(d)

        decoder = Model(decoder_inputs, decoded_img, name='decoder_model')
        decoder.summary()
        z_decoded = decoder(latent_vector)

        AE = Model(input_img, z_decoded)
        AE.compile(optimizer='rmsprop', loss='binary_crossentropy')
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

    CNN_AE = CNN_DenseLatentSpace(img_shape=(28, 28, 1), latent_space_dims=3, batch_size=64)
    CNN_AE.data_prep(keep_labels=[0])
    CNN_AE.define_model()
    CNN_AE.train(epochs=10)
    CNN_AE.inspect_model(dim_reduction_model='pca', dimensions=3)
    CNN_AE.save(name='cnn_0_labels_3D', save_type='weights')