import random
from model.layer import Layer
from model.node import Node
from config import NUM_HIDDEN_NODES


class HiddenLayer(Layer):

    def __init__(self):
        self.nodes = [Node()] * NUM_HIDDEN_NODES
        self.error_matrix = []

    def set_error(self, matrix):
        self.error_matrix = matrix