import sys
import math
import random
import logging
import numpy as np
from utils import *
from model.node import Node

logging.basicConfig(level=logging.INFO)

class WeightMatrix:

    def __init__(self, inlayer=None, outlayer=None):
        self.inlayer = inlayer
        self.outlayer = outlayer
        self.weights = np.random.random((len(inlayer), len(outlayer)))

    def propagate_forward(self):
        ''' Matrix Multiplication to forward propagate '''
        if not self.inlayer or not self.outlayer:
            sys.exit('NoneType layer detected')

        node_values = self.inlayer.get_node_array()
        output_values = node_values.dot(self.weights)
        for node, value in zip(self.outlayer, output_values):
            node.set_value(
                sigmoid(value)
            )

    def propagate_backward(self):
        ''' Use Sigmoid and MSE to back propagate '''
        if not self.inlayer or not self.outlayer:
            sys.exit('NoneType layer detected')

        error_matrix = self.outlayer.error_matrix

        # compute hidden error
        self.inlayer.set_error(
            self.weights.dot(error_matrix)
        )

        # compute new weight matrix
        self.weights = stochastic_gradient_descent(
            error_matrix, self.inlayer.get_node_array(), self.weights
        )
