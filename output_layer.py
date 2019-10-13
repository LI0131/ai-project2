import sys
import random
import operator
from layer import Layer
from node import Node
from config import NUM_OUTPUT_NODES


class OutputLayer(Layer):

    def __init__(self):
        self.nodes = self._build_output_nodes()

    def _build_output_nodes(self):
        output_nodes = {}
        for i in range(NUM_OUTPUT_NODES):
            output_nodes[str(i)] = Node()
        return output_nodes

    def reset_node_values(self):
        self.build_output_nodes()

    def get_classification(self):
        return max(self.nodes.items(), key=operator.itemgetter(1))[0]

    def __str__(self):
        return ', '.join([f'{num}: {str(node)}' for (num, node) in self.nodes.items()])


def _test():
    output = OutputLayer()
    print(output)


if __name__ == '__main__':
    _test()