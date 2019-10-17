import sys
import math
import random
import logging
import numpy as np
from utils import *

logging.basicConfig(level=logging.INFO)


class WeightMatrix:

    def __init__(self, inlayer=None, outlayer=None):
        self.inlayer = inlayer
        self.outlayer = outlayer
        self.weights = np.random.normal(
            loc=0, scale=((len(inlayer)+len(outlayer))**(-0.5)), size=(len(inlayer), len(outlayer))
        )

    def propagate_forward(self):
        ''' Matrix Multiplication to forward propagate '''
        if not self.inlayer or not self.outlayer:
            sys.exit('NoneType layer detected')
        node_values = self.inlayer.get_node_array()
        output_values = node_values.dot(self.weights)
        self.outlayer.set_node_values(
            sigmoid(output_values)
        )

    def propagate_backward(self):
        ''' Use stochastic gradient descent to backpropagate '''
        if not self.inlayer or not self.outlayer:
            sys.exit('NoneType layer detected')

        error_matrix = self.outlayer.error_matrix

        # compute hidden error
        self.inlayer.set_error(
            np.dot(error_matrix, self.weights.T)
        )

        # compute new weight matrix
        w_prime = stochastic_gradient_descent(
            error_matrix, self.inlayer.get_node_array(), 
            self.outlayer.get_node_array(), self.weights
        )

        self.weights = w_prime
