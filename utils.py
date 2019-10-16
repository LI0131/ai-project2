import math
import logging
import numpy as np
from config import LEARNING_RATE

logging.basicConfig(level=logging.INFO)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# def sigmoid_for_array(arr):
#     return 1/(1 * np.exp(arr))


# def derivative_sigmoid(x):
#     return sigmoid_for_array(x) * (1 - sigmoid_for_array(x))


# def stochastic_gradient_descent(error, prev_layer, weights):
#     gradient = prev_layer.dot(
#         error * derivative_sigmoid(prev_layer.dot(weights))
#     )
#     scaled_gradient = np.multiply(LEARNING_RATE, gradient)
#     return np.add(weights, scaled_gradient)

def sigmoid_for_array(arr):
    return 1/(1 * np.exp(arr))


def derivative_sigmoid(x):
    return sigmoid_for_array(x) * (1 - sigmoid_for_array(x))


def stochastic_gradient_descent(error, prev_layer, weights):
    gradient = np.dot(np.array(prev_layer).T, error * np.array(derivative_sigmoid(prev_layer.dot(weights))))
    scaled_gradient = np.multiply(LEARNING_RATE, gradient)
    return np.add(weights, scaled_gradient)


def squared_error(target, output):
    return (target - output)**2