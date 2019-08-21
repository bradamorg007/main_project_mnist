import keras
import keras.layers as layers
import matplotlib.pyplot as plt
from autoencoder_classes.fully_connected import FullyConnectedAE
from autoencoder_classes.convolutional import Convolutional
import pandas as pd
import numpy as np
import re
import os
import pickle as pkl


def file_exist(root_dir, target_dir):
   if os.path.isdir(os.path.join(root_dir, target_dir)) == False:
       raise ValueError('ERROR Find Model: File Does Not exist: %s' % (os.path.join(root_dir, root_dir)))


def find_model(model_names, model_func, latent_dim):


    load = None
    for m_name in model_names:
            if model_func.__name__ == m_name:
                n = list(model_func.__name__)[-1]
                load = n + '_' + str(latent_dim)
                break

    if load is None:
        raise ValueError('ERROR: Unrecognised model function from class FullyConnected')

    return load


def load_model(model_func, FC, CN, model_names, latent_dim):

    func_class = model_func.__qualname__

    if re.compile('FullyConnected').match(func_class):
        load = 'FC' + find_model(model_names, model_func,  latent_dim)
        file_exist(root_dir='models', target_dir=load)
        FC.load_weights(full_path='models/'+load, modelToRun=model_func, latent_dim=latent_dim)
        return FC, load[0:3]


    elif re.compile('Convolutional').match(func_class):
        load = 'CN' + find_model(model_names, model_func, latent_dim)
        file_exist(root_dir='models', target_dir=load)
        CN.load_weights(full_path='models/'+load, modelToRun=model_func, latent_dim=latent_dim)
        return  CN, load[0:3]

    else:
        raise ValueError('Function Class name not recognised')


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
        model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model



    def train(self, epochs, batch_size, validation_split, preview_results=False):

        history = self.model.fit(x=self.latent_data_x, y=self.data_y_onehot,
                                 shuffle=True, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, verbose=2)


        loss = history.history['loss']
        val_loss = history.history['val_loss']

        acc = history.history['acc']
        val_acc = history.history['val_acc']

        if preview_results:
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

        return np.array(loss), np.array(val_loss), np.array(acc), np.array(val_acc)


def run_main():

    # NEEEED TO ADD PCA VERSION TO IT AND NEEED TO BASELINE WITH JUST CLUSTER FUNCTION ********************************


    folder = 'clustering_analysis_data'
    filename = 'average_acc_loss_data'

    FC = FullyConnectedAE(img_shape=(28, 28, 1), batch_size=16)
    FC.data_prep(keep_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    CN = Convolutional(img_shape=(28, 28, 1), batch_size=16)
    CN.data_prep(keep_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    model_names = ['model1', 'model2', 'model3', 'model4']


    test_latent_dims = [30, 10, 2]
    """ model_functions = [FC.model1, FC.model2, FC.model3, FC.model4,
                       CN.model1, CN.model2, CN.model3, CN.model4] """
    model_functions = [FC.model1, FC.model2, FC.model3, FC.model4,
                       CN.model1, CN.model2, CN.model3, CN.model4]

    model_functions_load_paths = ['FC1', 'FC2', 'FC3', 'FC4',
                                  'CN1', 'CN2', 'CN3', 'CN4']

    test_per_model = 10
    epochs_per_test = 50
    perceptron_training_batch_size = 128
    perceptron_define_data_batch_size = 16
    perceptron_validation_split = 0.2
    perceptron_hidden_units = 20
    perceptron_result_outputs = 4

    #results = np.zeros(shape=(len(model_functions), len(test_latent_dims), epochs_per_test))

    results = {}

    for i, model_func in enumerate(model_functions):

        add_new_model_dic = True
        for j, latent_dim in enumerate(test_latent_dims):

            name = None
            add_new_ld_dic = True
            for k in range(test_per_model):

                model, name = load_model(model_func, FC=FC, CN=CN, model_names=model_names, latent_dim=latent_dim)

                if add_new_model_dic:
                    results[name] = {latent_dim: np.zeros(shape=(perceptron_result_outputs, epochs_per_test))}
                    add_new_model_dic = False
                elif add_new_ld_dic:
                    results[name].update({latent_dim: np.zeros(shape=(perceptron_result_outputs, epochs_per_test))})
                    add_new_ld_dic = False

                print()
                print('===============================================================================================')
                print('Model No: %s / %s Test: %s / %s Latent Dim: %s Model: %s' % (i+1, len(model_functions), k+1, test_per_model, latent_dim, name))
                print('===============================================================================================')
                print()

                cluster = ClusterAnalysis(test_model=model, data_x=model.x_train, data_y=model.y_train)
                cluster.define_data(batch_size=perceptron_define_data_batch_size)
                cluster.define_multilayered_perceptron(hidden_units=perceptron_hidden_units)
                loss, val_loss, acc, val_acc = cluster.train(epochs=epochs_per_test, batch_size=perceptron_training_batch_size,
                              validation_split=perceptron_validation_split,preview_results=False)

                results_matrix = np.array([loss, val_loss, acc, val_acc ])

                current_results = (results.get(name)).get(latent_dim)
                current_results = np.add(current_results, results_matrix)
                results[name][latent_dim] = current_results

            current_results = (results.get(name)).get(latent_dim)
            current_results = np.divide(current_results, test_per_model)
            results[name][latent_dim] = current_results

    if os.path.exists(folder) == False:
        os.mkdir(folder)

    if os.path.exists(os.path.join(folder, filename)) == False:
        os.mkdir(os.path.join(folder, filename))

    file = open(os.path.join(folder, filename , 'data.pkl'), 'wb')
    pkl.dump(results, file)
    file.close()
    print()
    print('===============================================================================================')
    print('CLUSTER ANALYSIS COMPLETE: Save Path %s' % (os.path.join(folder, filename)))
    print('===============================================================================================')
    print()

    # NEEEED TO ADD PCA VERSION TO IT AND NEEED TO BASELINE WITH JUST CLUSTER FUNCTION ********************************

if __name__ == '__main__':
    run_main()