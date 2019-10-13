import sys
import math
import random
import numpy as np
from node import Node

class WeightMatrix:

    def __init__(self, inlayer=None, outlayer=None):
        self.inlayer = inlayer
        self.outlayer = outlayer
        self.weights = np.random.random((len(inlayer), len(outlayer)))

    def propagate_forward(self):
        ''' Matrix Multiplication to forward propagate '''
        if not self.inlayer or not self.outlayer:
            sys.exit('None layer detected')

        node_values = self.inlayer.get_node_array()
        output_values = node_values.dot(self.weights)
        print(output_values)
        for node, value in zip(self.outlayer, output_values):
            node.set_value(
                self.sigmoid(value)
            )

    def propagate_backward(self):
        ''' Use Sigmoid and MSE to back propagate '''
        if not self.inlayer or not self.outlayer:
            sys.exit('None layer detected')

    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))