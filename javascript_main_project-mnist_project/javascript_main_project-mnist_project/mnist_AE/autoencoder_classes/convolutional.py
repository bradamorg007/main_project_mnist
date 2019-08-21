from autoencoder_classes.autoEncoder import AutoEncoder
from keras import layers
import keras
from keras import backend as K
from keras.models import Model
import numpy as np
import os

class Convolutional(AutoEncoder):


    def __init__(self, img_shape, batch_size):

        super().__init__(img_shape, batch_size)

        self.encoder_input_layer = None
        self.encoder_output_layer = None
        self.decoder_input_layer = None
        self.decoder_output_layer = None

    def model5(self, latent_dim):


        self.encoder_input_layer = layers.Input(shape=self.img_shape, name='encoder_inputs')

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='sigmoid', padding='same')(self.encoder_input_layer)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='sigmoid', padding='same', strides=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

        shape_before_flattening = K.int_shape(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=50, activation='sigmoid')(x)

        self.encoder_output_layer = layers.Dense(units=latent_dim, name='Latent_space',)(x)



        # DECODER =========================================================================================
        self.decoder_input_layer = layers.Input(shape=K.int_shape(self.encoder_output_layer)[1:])

        d = layers.Dense(units=50, activation='sigmoid')(self.decoder_input_layer)
        d = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='sigmoid')(d)
        d = layers.Reshape(target_shape=shape_before_flattening[1:])(d)

        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='sigmoid')(d)
        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='sigmoid')(d)
        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='sigmoid', strides=2)(d)
        d = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='sigmoid')(d)

        self.decoder_output_layer = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                                    activation='sigmoid', name='decoded_img')(d)


    def model4(self, latent_dim):


        self.encoder_input_layer = layers.Input(shape=self.img_shape, name='encoder_inputs')

        x = layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', strides=2)(self.encoder_input_layer)

        shape_before_flattening = K.int_shape(x)
        x = layers.Flatten()(x)

        self.encoder_output_layer = layers.Dense(units=latent_dim, name='Latent_space')(x)

        # DECODER =========================================================================================
        self.decoder_input_layer = layers.Input(shape=K.int_shape(self.encoder_output_layer)[1:])

        d = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(self.decoder_input_layer)
        d = layers.Reshape(target_shape=shape_before_flattening[1:])(d)
        d = layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(d)


        self.decoder_output_layer = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                                    activation='sigmoid', name='decoded_img')(d)


    def model3(self, latent_dim):


        self.encoder_input_layer = layers.Input(shape=self.img_shape, name='encoder_inputs')

        x = layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', strides=2)(self.encoder_input_layer)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(x)

        shape_before_flattening = K.int_shape(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(x)

        self.encoder_output_layer = layers.Dense(units=latent_dim, name='Latent_space',
                                     )(x)



        # DECODER =========================================================================================
        self.decoder_input_layer = layers.Input(shape=K.int_shape(self.encoder_output_layer)[1:])

        d = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(self.decoder_input_layer)
        d = layers.Reshape(target_shape=shape_before_flattening[1:])(d)
        d = layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(d)
        d = layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(d)


        self.decoder_output_layer = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                                    activation='sigmoid', name='decoded_img')(d)


    def model2(self, latent_dim):


        self.encoder_input_layer = layers.Input(shape=self.img_shape, name='encoder_inputs')

        x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(self.encoder_input_layer)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)

        shape_before_flattening = K.int_shape(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=50, activation='relu')(x)

        self.encoder_output_layer = layers.Dense(units=latent_dim, name='Latent_space')(x)



        # DECODER =========================================================================================
        self.decoder_input_layer = layers.Input(shape=K.int_shape(self.encoder_output_layer)[1:])

        d = layers.Dense(units=50, activation='relu')(self.decoder_input_layer)
        d = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(d)
        d = layers.Reshape(target_shape=shape_before_flattening[1:])(d)

        d = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(d)
        d = layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(d)

        self.decoder_output_layer = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                                    activation='sigmoid', name='decoded_img')(d)


    def model1(self, latent_dim):


        self.encoder_input_layer = layers.Input(shape=self.img_shape, name='encoder_inputs')

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(self.encoder_input_layer)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

        shape_before_flattening = K.int_shape(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=50, activation='relu')(x)

        self.encoder_output_layer = layers.Dense(units=latent_dim, name='Latent_space',
                                     )(x)



        # DECODER =========================================================================================
        self.decoder_input_layer = layers.Input(shape=K.int_shape(self.encoder_output_layer)[1:])

        d = layers.Dense(units=50, activation='relu')(self.decoder_input_layer)
        d = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(d)
        d = layers.Reshape(target_shape=shape_before_flattening[1:])(d)

        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(d)
        d = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(d)

        self.decoder_output_layer = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                                    activation='sigmoid', name='decoded_img')(d)




    def build(self, modelToRun, latent_dim, show_summary=False):

        modelToRun(latent_dim)
        encoder = Model(self.encoder_input_layer, self.encoder_output_layer, name='Encoder')


        name = 'decoder_model'

        decoder = Model(self.decoder_input_layer, self.decoder_output_layer, name=name)

        decoded_data = decoder(self.encoder_output_layer)

        AE = Model(self.encoder_input_layer, decoded_data)
        AE.compile(optimizer='adadelta', loss='binary_crossentropy')


        encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        decoder.compile(optimizer='adadelta', loss='binary_crossentropy')

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
    CN = Convolutional(img_shape=(28, 28, 1), batch_size=128)
    CN.data_prep(keep_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    epochs = 50
    latent_dims = [30, 10, 2]
    show_results = False
    model_functions = [CN.model1, CN.model2, CN.model3, CN.model4]


    for i in range(len(model_functions)):
        for latent_dim in latent_dims:

            print('==========================================================================================')
            print('Iteration: %s model: %s latent dim: %s' % (i, model_functions[i].__name__, latent_dim))
            print('==========================================================================================')
            model_func = model_functions[i]
            CN.build(modelToRun=model_func, latent_dim=latent_dim, show_summary=False)
            CN.train(epochs=epochs)
            if show_results:
                CN.inspect_model(dim_reduction_model='pca', dimensions=3)

            n = list(model_func.__name__)[-1]
            save_name = 'CN'+str(n)+'_'+str(latent_dim)
            CN.save(name=save_name, save_type='weights')


if __name__ == '__main__':
    run_main()