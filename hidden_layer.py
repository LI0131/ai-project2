import random
from layer import Layer
from node import Node
from config import NUM_HIDDEN_NODES


class HiddenLayer(Layer):

    def __init__(self):
        self.nodes = [Node()] * NUM_HIDDEN_NODES