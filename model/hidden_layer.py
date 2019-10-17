import random
from model.layer import Layer
from config import NUM_HIDDEN_NODES


class HiddenLayer(Layer):

    def __init__(self):
        self.nodes = []
        self.error_matrix = []

    def set_error(self, matrix):
        self.error_matrix = matrix

    def __len__(self):
        return NUM_HIDDEN_NODES