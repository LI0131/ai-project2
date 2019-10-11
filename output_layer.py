import sys
from layer import Layer
from node import Node
from config import NUM_OUTPUT_NODES


class OutputLayer(Layer):

    def __init__(self):
        output_nodes = {}

    def calculate_MSE(self):
        pass

    def propagate_forward(self):
        sys.exit('Cannot propagate forward from Output Layer')