import sys
from layer import Layer
from node import Node
from config import NUM_INPUT_NODES


class InputLayer(Layer):

    def __init__(image):
        self.input_nodes = self._distribute_image(image):

    def _distribute_image(self):
        '''
            This will distribute the image over a fixed number of nodes
            Returns a list of the nodes
        '''
        return []

    def propagate_backward(self):
        sys.exit('Cannot propagate backward from Input Layer')