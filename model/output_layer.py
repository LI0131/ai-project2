import sys
import random
import operator
import logging
import numpy as np
from utils import squared_error
from model.layer import Layer
from config import NUM_OUTPUT_NODES

logging.basicConfig(level=logging.INFO)


class OutputLayer(Layer):

    def __init__(self):
        self.nodes = []
        self.expected_output = None
        self.error_matrix = []

    # def _build_output_nodes(self):
    #     return [0] * NUM_OUTPUT_NODES

    def set_expected_output(self, number):
        self.expected_output = int(number)

    def set_error_matrix(self):
        output_values = self.get_node_array()
        error_matrix = []
        for i in range(len(output_values)):
            if i == self.expected_output:
                error_matrix.append(squared_error(1, output_values[i]))
            else:
                error_matrix.append(squared_error(0, output_values[i]))
        # logging.info(f'Appending Error Matrix: {[(error/ sum(error_matrix)) for error in error_matrix]}')
        self.error_matrix.append(
            np.array([(error/ sum(error_matrix)) for error in error_matrix])
        )
                
    def get_classification(self):
        node_array = self.get_node_array()
        return node_array.index(max(node_array))

    # Kind of Hacky but it solves the issue of length within the WeightMatrix w/o Abstract Classes
    def __len__(self):
        return NUM_OUTPUT_NODES


def _test():
    output = OutputLayer()
    print(output)


if __name__ == '__main__':
    _test()