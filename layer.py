from abc import ABCMeta, abstractmethod

class Layer(object, metaclass=ABCMeta):

    def __init__(self):
        self.nodes = []

    def propagate_forward(self):
        pass

    def propagate_backward(self):
        pass

    def reset_node_values(self):
        self.nodes = [Node() for node in self.nodes]
            
