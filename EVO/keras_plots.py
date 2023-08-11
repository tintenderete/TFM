

import matplotlib.pyplot as plt

def plot_histogram(data, title, x_label):
    plt.hist(data, bins='auto', alpha=0.7)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.show()


def get_activation_functions(models, layer_num):
    activation_functions = []
    for model in models:
        layers = model.get_config()["layers"]
        if len(layers) > layer_num:
            activation_functions.append(layers[layer_num]["config"]["activation"])
    return activation_functions

def get_neurons(models, layer_num):
    neurons = []
    for model in models:
        layers = model.get_config()["layers"]
        if len(layers) > layer_num:
            neurons.append(layers[layer_num]["config"]["units"])
    return neurons

def get_number_of_layers(models):
    num_layers = []
    for model in models:
        layers = model.get_config()["layers"]
        num_layers.append(len(layers))
    return num_layers
