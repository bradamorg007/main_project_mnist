import keras
import numpy as np
import pandas as pd

class Process:

    @staticmethod
    def data_prep(keep_labels):

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



        return x_train, y_train, x_test, y_test

    @staticmethod
    def flatten_image(input):

        output = []
        for i in range(len(input)):
            reshape = np.reshape(input[i], newshape=(np.prod(input[i].shape), 1))
            output.append(reshape)

        return np.array(output)


    @staticmethod
    def organise(order_sequence, x_data, y_data):

        if len(x_data) != len(y_data):
            raise ValueError('ERROR Process Organise: The sample data and label data must have the same number of elements')

        if order_sequence == 'random':
            indices = np.arange(len(y_data))
            np.random.shuffle(indices)

            x_data = x_data[indices]
            y_data = y_data[indices]

        elif isinstance(order_sequence, list):
            occurrence_table = Process.count_unquie(y_data)

            indices = []
            for item in order_sequence:

                get_data = occurrence_table.get(item)

                if get_data is not None:
                    indices = indices + get_data.get('indices')

            x_data = x_data[indices]
            y_data = y_data[indices]

        else:
            raise ValueError('ERROR Process Organise: invalid value for order_sequence argument'
                             'must be either random or a list of labels')

        return x_data, y_data



    @staticmethod
    def count_unquie(input):
        table = {}

        for i, rows in enumerate(input):

            # try to retrieve a label key, if it exists append the freq count and add i to indices

            lookup = table.get(rows)

            if lookup is not None:

                lookup['freq'] += 1
                lookup['indices'].append(i)

            else:
                # if it doesnt exist in the dict then make a new entry
                table[rows] = {'freq': 1, 'indices': [i]}


        return table

    @staticmethod
    def one_hot_encode(y_train=None, y_test=None, labels=None):


        lookup = {}
        if labels is not None:
            code = pd.get_dummies(labels).values

            for i in range(len(labels)):
                lookup[labels[i]] = code[i]

        y_train_output = []
        if y_train is not None:
            if isinstance(y_train, np.ndarray) == False:
                l = []
                l.append(y_train)
                l = np.array(l)
                y_train = l.copy()
            for item in y_train:
                one_hot = lookup.get(item)
                y_train_output.append(one_hot)

        y_test_output = []
        if y_test is not None:
            if isinstance(y_test, np.ndarray) == False:
                l = []
                l.append(y_test)
                l = np.array(l)
                y_test = l.copy()
            for item in y_test:
                one_hot = lookup.get(item)
                y_test_output.append(one_hot)


        return y_train_output, y_test_output






