import os
import keras
import logging
from keras.datasets import mnist
from model import Model

logging.basicConfig(level=logging.INFO)

unified_mnist = []


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
    logging.info('Pulling MNIST from Keras API...')
    (mnist_data_1, mnist_data_2), (mnist_data_3, mnist_data_4) = mnist.load_data()

    logging.info('Creating Unified MNIST set from Training and Testing...')
    unified_mnist = create_unified_dataset(
        mnist_data_1, mnist_data_2,
        mnist_data_3, mnist_data_4
    )

    logging.info(f'Number of Images in Set: {len(unified_mnist)}')

    logging.info(f'Creating Model...')
    model = Model()

    model.train(unified_mnist[:60000])

    model.test(unified_mnist[60001:])