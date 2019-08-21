from autoencoder_classes.autoEncoder import AutoEncoder
from keras import layers
import keras
from keras import backend as K
from keras.models import Model
import numpy as np
import os

class FullyConnectedAE(AutoEncoder):


    def __init__(self, img_shape, batch_size):

        super().__init__(img_shape, batch_size)

        self.encoder_input_layer = None
        self.encoder_output_layer = None
        self.decoder_input_layer = None
        self.decoder_output_layer = None

    def model1(self, latent_dim):


        self.encoder_input_layer = layers.Input(shape=self.img_shape)

        x = layers.Flatten()(self.encoder_input_layer)
        flat_img_dims = K.int_shape(x)[1]
        x = layers.Dense(units=500, activation='sigmoid')(x)
        x = layers.Dense(units=375, activation='sigmoid')(x)
        x = layers.Dense(units=125, activation='sigmoid')(x)

        self.encoder_output_layer = layers.Dense(units=latent_dim, activation='sigmoid')(x)


        self.decoder_input_layer = layers.Input(shape=K.int_shape(self.encoder_output_layer)[1:])

        x = layers.Dense(units=125, activation='sigmoid')(self.decoder_input_layer)
        x = layers.Dense(units=375, activation='sigmoid')(x)
        x = layers.Dense(units=500, activation='sigmoid')(x)
        x = layers.Dense(units=flat_img_dims, activation='sigmoid')(x)

        self.decoder_output_layer = layers.Reshape(target_shape=self.img_shape)(x)


    def model2(self, latent_dim):


        self.encoder_input_layer = layers.Input(shape=self.img_shape)

        x = layers.Flatten()(self.encoder_input_layer)
        flat_img_dims = K.int_shape(x)[1]
        x = layers.Dense(units=100, activation='sigmoid')(x)
        x = layers.Dense(units=125, activation='sigmoid')(x)
        x = layers.Dense(units=25, activation='sigmoid')(x)

        self.encoder_output_layer = layers.Dense(units=latent_dim, activation='sigmoid')(x)


        self.decoder_input_layer = layers.Input(shape=K.int_shape(self.encoder_output_layer)[1:])

        x = layers.Dense(units=125, activation='sigmoid')(self.decoder_input_layer)
        x = layers.Dense(units=375, activation='sigmoid')(x)
        x = layers.Dense(units=500, activation='sigmoid')(x)
        x = layers.Dense(units=flat_img_dims, activation='sigmoid')(x)

        self.decoder_output_layer = layers.Reshape(target_shape=self.img_shape)(x)


    def model3(self, latent_dim):


        self.encoder_input_layer = layers.Input(shape=self.img_shape)

        x = layers.Flatten()(self.encoder_input_layer)
        flat_img_dims = K.int_shape(x)[1]
        x = layers.Dense(units=532, activation='sigmoid')(x)

        self.encoder_output_layer = layers.Dense(units=latent_dim, activation='sigmoid')(x)

        self.decoder_input_layer = layers.Input(shape=K.int_shape(self.encoder_output_layer)[1:])

        x = layers.Dense(units=532, activation='sigmoid')(self.decoder_input_layer)
        x = layers.Dense(units=flat_img_dims, activation='sigmoid')(x)

        self.decoder_output_layer = layers.Reshape(target_shape=self.img_shape)(x)


    def model4(self, latent_dim):


        self.encoder_input_layer = layers.Input(shape=self.img_shape)

        x = layers.Flatten()(self.encoder_input_layer)

        self.encoder_output_layer = layers.Dense(latent_dim, activation='sigmoid')(x)

        self.decoder_input_layer = layers.Input(shape=K.int_shape(self.encoder_output_layer)[1:])
        x = layers.Dense(K.int_shape(x)[1], activation='sigmoid')(self.decoder_input_layer)

        self.decoder_output_layer = layers.Reshape(target_shape=self.img_shape)(x)



    def build(self, modelToRun, latent_dim, show_summary=False):

        modelToRun(latent_dim)
        encoder = Model(self.encoder_input_layer, self.encoder_output_layer, name='Encoder')


        name = 'decoder_model'

        decoder = Model(self.decoder_input_layer, self.decoder_output_layer, name=name)
        decoded_data = decoder(self.encoder_output_layer)

        AE = Model(self.encoder_input_layer, decoded_data)
        AE.compile(optimizer='rmsprop', loss='binary_crossentropy')

        encoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
        decoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

        if show_summary:
            encoder.summary()
            decoder.summary()
            AE.summary()

        self.model = AE
        self.encoder = encoder
        self.decoder = decoder
        self.define_flag = True


    def load_weights(self, full_path, modelToRun, latent_dim):

        self.build(modelToRun=modelToRun, latent_dim=latent_dim)

        self.model.load_weights(os.path.join(full_path, 'weights_model.h5'))
        self.encoder.load_weights(os.path.join(full_path, 'weights_encoder_model.h5'))
        self.decoder.load_weights(os.path.join(full_path, 'weights_decoder_model.h5'))

        print('LOAD WEIGHTS COMPLETE')

def run_main():
    FC = FullyConnectedAE(img_shape=(28, 28, 1), batch_size=128)
    FC.data_prep(keep_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    epochs = 50
    latent_dims = [30, 10, 2]
    show_results = False
    model_functions = [FC.model1, FC.model2, FC.model3, FC.model4]

    for i in range(len(model_functions)):
        for latent_dim in latent_dims:

            print('==========================================================================================')
            print('Iteration: %s model: %s latent dim: %s' % (i, model_functions[i].__name__, latent_dim))
            print('==========================================================================================')
            model_func = model_functions[i]
            FC.build(modelToRun=model_func, latent_dim=latent_dim, show_summary=False)
            FC.train(epochs=epochs)
            if show_results:
                FC.inspect_model(dim_reduction_model='pca', dimensions=3)

            n = list(model_func.__name__)[-1]
            save_name = 'FC'+str(n)+'_'+str(latent_dim)
            FC.save(name=save_name, save_type='weights')

if __name__ == '__main__':