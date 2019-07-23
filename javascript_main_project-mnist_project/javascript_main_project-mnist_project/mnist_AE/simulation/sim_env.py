import numpy as np
from functional_classes.function_system import FunctionSystem
from autoencoder_classes.autoEncoder import AutoEncoder
from visual_classes.visual_system import VisualSystem
from data_preprocessing_classes.process import Process
from memory_classes.memory_system import MemorySystem
import matplotlib.pyplot as plt

def crop_data(data_x, data_y, range, mode=' '):

    inds = None
    if mode == 'shuffle':
        inds = np.arange(start=range[0], stop=range[1])
        np.random.shuffle(inds)
    else:
        inds = np.arange(start=range[0], stop=range[1])

    data_x = data_x[inds]
    data_y = data_y[inds]

    return data_x, data_y

def arg_max(data):
    max = data[0]
    index = 0
    for i, class_label in enumerate(data):
        if class_label > max:
            max = class_label
            index = i

    return max, index

def prediction_check(prediction, label):
    if index != label:
        raise ValueError('MODEL ERROR: The Functional System predicted an incorrect class, this could mean'
                         'that the VS system wrongly wrongly stated a sample being familiar when it in fact it'
                         'was not')


def image_comparison(input1, input2):
    plt.figure()
    plt.title('decoded memory image')
    plt.imshow(input1.reshape(28, 28))
    plt.gray()

    plt.figure()
    plt.title('latent representation image')
    plt.imshow(input2.reshape(28, 28))
    plt.gray()
    plt.show()



# create data
data = AutoEncoder((28, 28, 1), None, None)

data.data_prep([1])
x_train_seen = data.x_train
y_train_seen = data.y_train

data.data_prep([4])
x_train_unseen = data.x_train
y_train_unseen = data.y_train

sample_range = (5, 10)

x_train_seen, y_train_seen = crop_data(x_train_seen, y_train_seen, range=sample_range, mode='shuffle')
x_train_unseen, y_train_unseen = crop_data(x_train_unseen, y_train_unseen, range=sample_range, mode='shuffle')

sim_space_x = np.concatenate((x_train_seen, x_train_unseen))
sim_space_y = np.concatenate((y_train_seen, y_train_unseen))

one_hot_labels, _ = Process.one_hot_encode(y_train=sim_space_y, labels=[0,1,2,3,4,5,6,7,8,9])


image_shape = (28,28, 1)

visual_system = VisualSystem(img_shape=image_shape, latent_dimensions=15, batch_size=1, RE_delta=0.1)

functional_system = FunctionSystem(img_shape=image_shape, keep_labels=None, ORDER_SEQUENCE=None)
functional_system.load_weights(full_path='../functional_classes/models/functional_system_digit1')

weights = functional_system.model.get_weights()

memory_system = MemorySystem(low_simularity_threshold=6, high_simularity_threshold=2,
                             forget_usage_threshold=1, forget_age_threshold=30, max_memory_size=50)

# create some dummy memory data

data.data_prep([1])
x_train_seen_d = data.x_train
y_train_seen_d = data.y_train

data.data_prep([4])
x_train_unseen_d = data.x_train
y_train_unseen_d = data.y_train

x_train_seen_dummy, y_train_seen_dummy = crop_data(x_train_seen_d, y_train_seen_d, range=(60,65), mode='shuffle')
x_train_unseen_dummy, y_train_unseen_dummy = crop_data(x_train_unseen_d, y_train_unseen_d, range=(60,65), mode='shuffle')

latent_x_seen_dummy, _ = visual_system.model.encoder.predict(x_train_seen_dummy)
latent_x_unseen_dummy, _ = visual_system.model.encoder.predict(x_train_unseen_dummy)

memory_system.create_memory(latent_representation=latent_x_seen_dummy[0], solution='dummy_seen0')
memory_system.create_memory(latent_representation=latent_x_seen_dummy[1], solution='dummy_seen1')
memory_system.create_memory(latent_representation=latent_x_seen_dummy[2], solution='dummy_seen2')
memory_system.create_memory(latent_representation=latent_x_seen_dummy[3], solution='dummy_seen3')

memory_system.create_memory(latent_representation=latent_x_unseen_dummy[0], solution='dummy_unseen0')
memory_system.create_memory(latent_representation=latent_x_unseen_dummy[1], solution='dummy_unseen1')
memory_system.create_memory(latent_representation=latent_x_unseen_dummy[2], solution='dummy_unseen2')
memory_system.create_memory(latent_representation=latent_x_unseen_dummy[3], solution='dummy_unseen3')





for i, (sample, label) in enumerate(zip(sim_space_x, sim_space_y)):

    if len(sample.shape) == 3:
        sample = sample.reshape((1,) + sample.shape)

    # AE checks to determine if the current sample is familiar or not
    is_familiar = visual_system.is_familular(sample)

    if is_familiar:
        # if sample has been seen before that it should be okay to use the current state of the functional model
        prediction = functional_system.model.predict(sample)
        max, index = arg_max(prediction[0])
        prediction_check(max, label)

    else:
        # get latent representation from vs
        latent_representation, _ = visual_system.model.encoder.predict(sample)
        memory, action = memory_system.query(latent_representation[0])

        if memory is not None and action == 'memory_to_fs_system_switch':

            a = 0
        elif memory is None and action == 'adaption_using_medium_memory_as_init_foundation':
            pass

        elif memory is None and action == 'adaption_using_low_memory_and_random_init_foundation':
            pass


        # Query memories to find

    print('index = %s Is Familiar: %s' % (i, is_familiar))






# NOTES AND USEFULL CODE
# memory_lts = memory.get('latent_representation')
# memory_lts = memory_lts.reshape((1,) + memory_lts.shape)
# decoded_memory_img = visual_system.model.decoder.predict(memory_lts)
# decoded_sample_lts = visual_system.model.decoder.predict(latent_representation)





