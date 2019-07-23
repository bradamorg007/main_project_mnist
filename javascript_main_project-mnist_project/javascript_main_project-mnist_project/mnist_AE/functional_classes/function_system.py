from data_preprocessing_classes.process import Process
import keras
import keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import os
import json


class FunctionSystem:

    def __init__(self, img_shape, keep_labels, ORDER_SEQUENCE):

        self.img_shape = img_shape
        self.model = None
        self.history = None
        self.ORDER_SEQUENCE = ORDER_SEQUENCE
        self.keep_labels = keep_labels

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.train_label_table = None
        self.train_label_table = None

        self.y_train_onehot = None
        self.y_test_onehot = None


    def define_data(self):

        x_train, y_train, x_test, y_test = Process.data_prep(keep_labels=self.keep_labels)
        x_train, y_train = Process.organise(self.ORDER_SEQUENCE, x_train, y_train)
        x_test, y_test = Process.organise(self.ORDER_SEQUENCE, x_test, y_test)

        self.train_label_table = Process.count_unquie(y_train)
        self.test_label_table = Process.count_unquie(y_test)

        self.y_train_onehot, self.y_test_onehot = Process.one_hot_encode(y_train, y_test, labels=[0,1,2,3,4,5,6,7,8,9])

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def define_model(self):

        input = layers.Input(shape=self.img_shape)

        x = layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', strides=2)(input)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=2)(x)

        x = layers.Flatten()(input)
        x = layers.Dense(units=32, activation='relu')(x)
        output = layers.Dense(units=10, activation='softmax')(x)

        model = keras.Model(input, output, name='Functional_System')
        optimizer = keras.optimizers.SGD(lr=0.01, nesterov=True)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model


    def train(self, epochs, batch_size):

        history = self.model.fit(self.x_train, self.y_train_onehot, shuffle=False, epochs=epochs, batch_size=batch_size,
                                 validation_data=(self.x_test, self.y_test_onehot), verbose=2)

        self.history = history


    def inspect_model(self):

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']

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


    def save(self, name, save_type):

        folder = "models"

        if os.path.exists(folder) == False:
            os.mkdir(folder)

        if os.path.exists(os.path.join(folder, name)) == False:
            os.mkdir(os.path.join(folder, name))


        if save_type == 'model':

            self.model.save(os.path.join(folder, name, 'full_model.h5'))
            self.model.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            print('SAVE MODEL COMPLETE')

        elif save_type == 'weights':
            self.model.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            print('SAVE WEIGHTS COMPLETE')

        filename = os.path.join(folder, name, 'order_sequence_config.json')

        with open(filename, 'w') as f:
            json.dump(self.ORDER_SEQUENCE, f, indent=4, sort_keys=True)


    def load_weights(self, full_path):

        self.define_model()
        self.model.load_weights(os.path.join(full_path, 'weights_model.h5'))

        print('LOAD WEIGHTS COMPLETE')


if __name__ == '__main__':

    #test for catastrophic forgetting

    # train on digit 0
    ORDER_SEQUENCE = 'random' # or can be random
    model = FunctionSystem(img_shape=(28, 28, 1), keep_labels=[1], ORDER_SEQUENCE=ORDER_SEQUENCE)
    model.define_data()
    model.define_model()
    model.train(epochs=10, batch_size=128)
    model.inspect_model()
    model.save(name='functional_system_digit1', save_type='weights')


    # sample = model.x_test[30]
    # sample = sample.reshape((1,) + sample.shape)
    # pred = model.model.predict(sample)


    # score_digit1 = model.model.evaluate(model.x_test, model.y_test_onehot, verbose=0)
    # print('digit 1 data: loss %s acc %s' % (score_digit1[0], score_digit1[1]))
    #
    # model.keep_labels = [0,8]
    # model.define_data()
    # model.train(epochs=10, batch_size=128)
    # model.inspect_model()
    #
    # score_digit1 = model.model.evaluate(model.x_test, model.y_test_onehot, verbose=0)
    # print('digit 0: loss %s acc %s' % (score_digit1[0], score_digit1[1]))
    #
    # model.keep_labels = [1,4]
    # model.define_data()
    # score_digit1 = model.model.evaluate(model.x_test, model.y_test_onehot, verbose=0)
    # print('digit 1: loss %s acc %s' % (score_digit1[0], score_digit1[1]))
    #
    # a = 1

