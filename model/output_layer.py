import sys
import random
import operator
import logging
import numpy as np
from utils import error
from model.layer import Layer
from config import NUM_OUTPUT_NODES

logging.basicConfig(filename='test_data.log', level=logging.DEBUG)


class OutputLayer(Layer):

    def __init__(self):
        self.nodes = []
        self.expected_output = None
        self.error_matrix = []

    def set_expected_output(self, number):
        self.expected_output = int(number)

    def set_error_matrix(self):
        output_values = self.get_node_array()[0]
        error_matrix = []
        for i in range(len(output_values)):
            if i == self.expected_output:
                error_matrix.append(error(1, output_values[i]))
            else:
                error_matrix.append(error(0, output_values[i]))
        self.error_matrix = np.array(error_matrix)

    def get_classification(self):
        node_array = list(self.get_node_array()[0])
        return node_array.index(max(node_array))

    def __len__(self):
        return NUM_OUTPUT_NODES


def _test():
    output = OutputLayer()
    print(output)


if __name__ == '__main__':
    _test()