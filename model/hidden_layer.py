import random
from model.layer import Layer
from config import NUM_HIDDEN_NODES


class HiddenLayer(Layer):

    def __init__(self):
        self.nodes = []
        self.error_matrix = []

    def set_error(self, matrix):
        self.error_matrix = matrix

    # Kind of Hacky but it solves the issue of length within the WeightMatrix w/o Abstract Classes
    def __len__(self):
        return NUM_HIDDEN_NODES