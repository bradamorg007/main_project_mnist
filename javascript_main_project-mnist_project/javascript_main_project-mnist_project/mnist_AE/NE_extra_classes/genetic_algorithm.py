# Imports
import numpy as np
from data_preprocessing_classes.process import Process
from NE_extra_classes.feedfoward_neural_network import NeuralNetwork
from NE_extra_classes.agent import Agent

# Parameters
POPULATION_SIZE = 200
GLOBAL_TARGET = 97 # percent accuracy
MUTATION_RATE = 0.1
IMG_SIZE = (28, 28)
BATCH_SIZE = 16
SURVIVAL_THRESHOLD = 0.1 #only breed top 10% of agents
ORDER_SEQUENCE = [9,8,7,6,5,4,3,2,1,0] # or can be random
ACTIVE_AGENTS = []
DEAD_AGENTS = []


TOPOLOGY = [784, 392, 196, 98, 49, 10]
NON_LINEARITIES = ['relu', 'relu', 'relu', 'relu', 'softmax']
WEIGHT_INIT_TYPE = 'he_normal'
FS_TEMPLATE = NeuralNetwork(layer_units=TOPOLOGY, activation_func_list=NON_LINEARITIES)
FS_TEMPLATE.init_layers(init_type=WEIGHT_INIT_TYPE)

# Data pre-processing
x_train, y_train, x_test, y_test = Process.data_prep(keep_labels=[0,1,2,3,4,5,6,7,8,9])
x_train, y_train = Process.organise(ORDER_SEQUENCE, x_train, y_train)

x_train = Process.flatten_image(x_train)
x_test = Process.flatten_image(x_test)

# Create agents
for i in range(POPULATION_SIZE):
    ACTIVE_AGENTS.append(Agent(fs_template=FS_TEMPLATE))


#Main simulation loop

