import math
import logging
import numpy as np
from config import LEARNING_RATE

logging.basicConfig(filename='test_data.log', level=logging.DEBUG)


def normalize_pixel_value(pixel):
    val = pixel/255
    if val == 1:
        return 0.99
    elif val == 0:
        return 0.01
    else:
        return val


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def stochastic_gradient_descent(error, prev_layer, layer, weights):

    # compute gradient
    gradient = prev_layer.T.dot(
        error * (layer * (1 - layer))
    )
    
    # scale by learning rate
    scaled_gradient = LEARNING_RATE * gradient

    # return new weights
    return weights - scaled_gradient


def error(target, output):
    return output - target