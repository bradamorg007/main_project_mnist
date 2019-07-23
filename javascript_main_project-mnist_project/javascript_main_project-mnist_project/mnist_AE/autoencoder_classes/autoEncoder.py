
import keras

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import sklearn.decomposition as decomposition
import time
import os
import seaborn as sns
import pickle as pkl


class AutoEncoder:

    def __init__(self, img_shape, latent_space_dims, batch_size):

       self.img_shape = img_shape
       self.latent_space_dims = latent_space_dims
       self.encoder = None
       self.latent_space = None
       self.decoder = None
       self.model = None
       self.batch_size = batch_size
       self.x_train = None
       self.y_train = None
       self.x_test = None
       self.y_test = None
       self.history = None
       self.reconstruction_error = None

       self.define_flag = False
       self.data_flag = False
       self.train_flag = False
       self.model_type_flag = None


    def data_prep(self, keep_labels):

        # DATA PREP=============================================================================================================
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        keep_index1 = []
        for i, label in enumerate(y_train):

            for item in keep_labels:

                if label == item:
                    keep_index1.append(i)
                    break

        keep_index2 = []
        for i, label in enumerate(y_test):

            for item in keep_labels:

                if label == item:
                    keep_index2.append(i)
                    break

        x_train = x_train[keep_index1]
        y_train = y_train[keep_index1]
        x_test = x_test[keep_index2]
        y_test = y_test[keep_index2]

        x_train = x_train.astype('float32') / 255
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.astype('float32') / 255
        x_test = x_test.reshape(x_test.shape + (1,))

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.keep_labels = keep_labels
        self.data_flag = True


    def train(self, epochs):

        if self.define_flag and self.data_flag:
            # =======================================================================================================================
            history = self.model.fit(x=self.x_train, y=self.x_train,
                                     shuffle=True, epochs=epochs, batch_size=self.batch_size,
                                     validation_data=(self.x_test, self.x_test), verbose=2)

            self.history = history
            self.train_flag = True

        else:
            raise ValueError('ERROR: THE MODEL AND THE DATA MUST BE DEFINED BEFORE TRAIN CAN BE CALLED')


    def predict(self, model, input, batch_size, dim_reduction_model, dimensions, dim_reduce=True):

        pred = model.predict(input, batch_size=batch_size)

        if isinstance(pred, list) and len(pred) > 1:
            # if variational AE used just use the mean value outputs not the stdevs
            pred = pred[0]

        if self.model_type_flag == 'cnn_lts':
            pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1] * pred.shape[2] * pred.shape[3]))

        if pred.shape[1] > 3 and dim_reduce == True:

            if dim_reduction_model == 'tsne':
                # perform tsne
                pred = self.tsne(pred, dimensions=dimensions)

            elif dim_reduction_model == 'pca':
                pred = self.pca(pred, dimensions=dimensions)


        return pred



    def save(self, name, save_type):

        folder = "../autoencoder_classes/models/"

        if os.path.exists(folder) == False:
            os.mkdir(folder)

        if os.path.exists(os.path.join(folder, name)) == False:
            os.mkdir(os.path.join(folder, name))


        if save_type == 'model':

            self.model.save(os.path.join(folder, name, 'full_model.h5'))
            self.encoder.save(os.path.join(folder, name, 'encoder_model.h5'))
            self.latent_space.save(os.path.join(folder, name, 'latent_space_model.h5'))
            self.decoder.save(os.path.join(folder, name, 'decoder_model.h5'))

            self.model.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            self.encoder.save_weights(os.path.join(folder, name, 'weights_encoder_model.h5'))
            self.decoder.save_weights(os.path.join(folder, name, 'weights_decoder_model.h5'))
            print('SAVE MODEL COMPLETE')

        elif save_type == 'weights':
            self.model.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            self.encoder.save_weights(os.path.join(folder, name, 'weights_encoder_model.h5'))
            self.decoder.save_weights(os.path.join(folder, name, 'weights_decoder_model.h5'))
            print('SAVE WEIGHTS COMPLETE')

        elif save_type == 'weights_and_reconstruction_error':

            if self.x_test is not None:
              RE = self.model.evaluate(self.x_test, self.x_test)

            elif self.x_train is not None:
              RE = self.model.evaluate(self.x_train, self.x_train)

            else:
                raise ValueError('ERROR AUTOENCODER: No Data is available to perform an evaluation')


            self.model.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            self.encoder.save_weights(os.path.join(folder, name, 'weights_encoder_model.h5'))
            self.decoder.save_weights(os.path.join(folder, name, 'weights_decoder_model.h5'))
            filename = open(os.path.join(folder, name,'reconstruction_error.pkl'), 'wb')
            pkl.dump(RE, filename)
            filename.close()
            print('SAVE WEIGHTS COMPLETE')





    def tsne(self, input, dimensions):
        time_start = time.time()
        tsne = TSNE(n_components=dimensions, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(input)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
        return tsne_results


    def pca(self, input, dimensions):

        pca = decomposition.PCA(n_components=dimensions)
        pca.fit(input)
        input = pca.transform(input)
        return input


    def inspect_model(self, dim_reduction_model='tsne', dimensions=2):
        # PLOTTING & METRICS===================================================================================================

        # plot the latent space of the VAE
        #encoder = Model(input_img, [z_mean, z_log_var, latent_space], name='encoder')

        pred = self.predict(model=self.encoder, input=self.x_test,
                            batch_size=self.batch_size, dim_reduction_model=dim_reduction_model,
                            dimensions=dimensions)

        if pred.shape[1] == 2:
            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x=pred[:, 0], y=pred[:, 1],
                hue=self.y_test,
                palette=sns.color_palette("hls", len(self.keep_labels)),
                legend="full",
                alpha=0.3
            )

        elif pred.shape[1] == 3:
            fig = plt.figure(figsize=(6, 6))
            ax = Axes3D(fig)
            p = ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=self.y_test)
            fig.colorbar(p)
            fig.show()

        # Plot comparisons between original and decoded images with test data
        decoded_imgs = self.model.predict(self.x_test)
        n = 10
        plt.figure(figsize=(20, 4))

        for i in range(n):
            # disp original

            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.x_test[i].reshape(28, 28))
            plt.gray()

            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()

        plt.show()

        # Plot Traning and validation reconstruction error
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(loss) + 1)

        plt.figure()

        plt.plot(epochs, loss, 'g', label='training loss')
        plt.plot(epochs, val_loss, 'r', label='validation loss')
        plt.title('Training and Validation Reconstruction Error')
        plt.legend()

        plt.show()



if __name__ == '__main__':
    pass


