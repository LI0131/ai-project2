import sys
import random
import operator
import numpy as np
from utils import squared_error
from model.layer import Layer
from model.node import Node
from config import NUM_OUTPUT_NODES


class OutputLayer(Layer):

    def __init__(self, seed=None):
        self.nodes = self._build_output_nodes()
        self.expected_output = seed
        self.error_matrix = []

    def _build_output_nodes(self):
        return [Node()] * NUM_OUTPUT_NODES

    def reset_node_values(self):
        self._build_output_nodes()

    def set_expected_output(self, number):
        self.expected_output = int(number)

    def set_error_matrix(self):
        output_values = self.get_node_array()
        error_matrix = []
        for i in range(len(self.nodes)):
            if i == self.expected_output:
                error_matrix.append(squared_error(1, output_values[i]))
            else:
                error_matrix.append(squared_error(0, output_values[i]))
        self.error_matrix = np.array([(error/ sum(error_matrix)) for error in error_matrix])
                
    def get_classification(self):
        node_array = self.get_node_array()
        return node_array.index(max(node_array))


def _test():
    output = OutputLayer()
    print(output)


if __name__ == '__main__':
    _test()