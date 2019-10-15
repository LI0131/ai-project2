from abc import ABCMeta, abstractmethod
import numpy as np
from model.node import Node

class Layer(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self.nodes = []

    def reset_node_values(self):
        self.nodes = [Node() for node in self.nodes]
            
    def get_node_array(self):
        return np.array([node.value for node in self.nodes])

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        return ', '.join([str(node) for node in self.nodes])

    def __iter__(self):
        for node in self.nodes:
            yield node