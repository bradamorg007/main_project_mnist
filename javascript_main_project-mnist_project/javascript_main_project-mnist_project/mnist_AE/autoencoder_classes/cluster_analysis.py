import keras
import keras.layers as layers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from autoencoder_classes.cnn_with_cnn_lts import CNN_ConvLatentSpace
from autoencoder_classes.variation_AutoEncoder import VariationAutoEncoder
import pandas as pd



class ClusterAnalysis:

    def __init__(self, test_model, data_x, data_y):

        self.test_model = test_model
        self.data_x = data_x
        self.latent_data_x = None
        self.data_y = data_y
        self.data_y_onehot = None
        self.model = None
        self.latent_data_x_flag = False


    def define_data(self, batch_size):


        pred = self.test_model.predict(self.test_model.encoder,
                                       self.data_x, batch_size=batch_size,
                                       dim_reduction_model='NA', dimensions='NA', dim_reduce=False)

        self.data_y_onehot = pd.get_dummies(self.data_y).values

        self.latent_data_x = pred
        self.latent_data_x_flag = True

    def get_latent_data_shape(self):
        if self.latent_data_x_flag == True:
            return self.latent_data_x[0].shape
        else:
            raise ValueError('ERROR get_latent_data_shape: '
                             'Please define the latent data first as currently it is set to None')



    def define_multilayered_perceptron(self, hidden_units):

        input = layers.Input(shape=self.get_latent_data_shape())

        x = layers.Dense(units=hidden_units, activation='relu')(input)
        output = layers.Dense(units=10, activation='softmax')(x)

        model = keras.Model(input, output, name='latent_test_perception')
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model



    def train(self, epochs, batch_size, validation_split):

        history = self.model.fit(x=self.latent_data_x, y=self.data_y_onehot,
                                 shuffle=True, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, verbose=2)


        loss = history.history['loss']
        val_loss = history.history['val_loss']

        acc = history.history['acc']
        val_acc = history.history['val_acc']

        epochs = range(1, len(loss) + 1)

        plt.figure()
        plt.plot(epochs, loss, 'g', label='training loss')
        plt.plot(epochs, val_loss, 'r', label='validation loss')
        plt.title('Training and Validation loss Error')
        plt.legend()

        plt.show()

        epochs = range(1, len(loss) + 1)

        plt.figure()
        plt.plot(epochs, acc, 'g', label='training loss')
        plt.plot(epochs, val_acc, 'r', label='validation loss')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.show()




if __name__ == '__main__':



    MODEL = VariationAutoEncoder(img_shape=(28, 28, 1), latent_space_dims=10, batch_size=16)
    MODEL.data_prep(keep_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    MODEL.load_weights(full_path='models/vae_all_labels_10D')

    cluster = ClusterAnalysis(test_model=MODEL, data_x=MODEL.x_train, data_y=MODEL.y_train)
    cluster.define_data(batch_size=16)
    cluster.define_multilayered_perceptron(hidden_units=28)
    cluster.train(epochs=50, batch_size=128, validation_split=0.2)


