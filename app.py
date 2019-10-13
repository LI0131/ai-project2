import os
import keras
import logging
from keras.datasets import mnist
from config import TRAINING_PERCENTAGE, TESTING_PERCENTAGE, VALIDATION_PERCENTAGE
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from weight_matrix import WeightMatrix

logging.basicConfig(level=logging.INFO)

unified_mnist = []


def create_model(image):
    ''' Create a model with two hidden layers and softmax output '''
    input_layer = InputLayer(image)
    first_hidden = HiddenLayer()
    second_hidden = HiddenLayer()
    softmax = OutputLayer()

    first_weight_matrix = WeightMatrix(input_layer, first_hidden)
    second_weight_matrix = WeightMatrix(first_hidden, second_hidden)
    third_weight_matrix = WeightMatrix(second_hidden, softmax)

    return input_layer, first_hidden, second_hidden, softmax, \
            first_weight_matrix, second_weight_matrix, third_weight_matrix


def run(model):
    pass


def create_unified_dataset(set1, label1, set2, label2):
    output_set = []
    for (data, labels) in [(set1, label1), (set2, label2)]:
        for index in range(len(data)):
            matrix = data[index]
            label = labels[index]
            output_set.append({
                'image': [list(row) for row in matrix],
                'label': label
            })
    return output_set


if __name__ == '__main__':
    logging.info(f'Using Training Percentage: {TRAINING_PERCENTAGE}')
    logging.info(f'Using Testing Percentage: {TESTING_PERCENTAGE}')
    logging.info(f'Using Validation Percentage: {VALIDATION_PERCENTAGE}')

    logging.info('Pulling MNIST from Keras API...')
    (mnist_data_1, mnist_data_2), (mnist_data_3, mnist_data_4) = mnist.load_data()

    logging.info('Creating Unified MNIST set from Training and Testing...')
    unified_mnist = create_unified_dataset(
        mnist_data_1, mnist_data_2,
        mnist_data_3, mnist_data_4
    )

    logging.info(f'Number of Images in Set: {len(unified_mnist)}')

    logging.info(f'Create Model')
    input_layer, first_hidden, second_hidden, softmax, \
            first_weight_matrix, second_weight_matrix, third_weight_matrix = create_model(unified_mnist[0]['image'])

    logging.info(f'Hidden Layer: {first_hidden}')

    first_weight_matrix.propagate_forward()

    logging.info(f'Hidden Layer: {first_hidden}')