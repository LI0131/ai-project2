from abc import ABCMeta, abstractmethod
import numpy as np

class Layer(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self.nodes = []
        self.error_matrix = []

    def set_node_values(self, matrix):
        self.nodes = matrix

    def reset_node_values(self):
        self.nodes = []

    def reset_error_matrix(self):
        self.error_matrix = []
            
    def get_node_array(self):
        return np.array(self.nodes)

    def __str__(self):
        return ', '.join([str(node) for node in self.nodes])

    def __iter__(self):
        for node in self.nodes:
            yield node
