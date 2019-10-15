from model.input_layer import InputLayer
from model.hidden_layer import HiddenLayer
from model.output_layer import OutputLayer
from model.weight_matrix import WeightMatrix

class Model:

    def __init__(self, seed):
        self.input_layer = InputLayer(seed['image'])
        self.first_hidden_layer = HiddenLayer()
        self.second_hidden_layer = HiddenLayer()
        self.softmax = OutputLayer(seed['label'])
        self.first_weights = WeightMatrix(self.input_layer, self.first_hidden_layer)
        self.second_weights = WeightMatrix(self.first_hidden_layer, self.second_hidden_layer)
        self.third_weights = WeightMatrix(self.second_hidden_layer, self.softmax)

    def _propagate_forward(self):
        for matrix in [self.first_weights, self.second_weights, self.third_weights]:
            matrix.propagate_forward()

    def _propagate_backward(self):
        self.softmax.set_error_matrix()
        for matrix in [self.third_weights, self.second_weights, self.first_weights]:
            matrix.propagate_backward()
        
    def train(self, images):
        for image in images:
            self.input_layer.reset_image(image['image'])
            self.softmax.set_expected_output(image['label'])
            for layer in [self.first_hidden_layer, self.second_hidden_layer, self.softmax]:
                layer.reset_node_values()
            self._propagate_forward()
            self._propagate_backward()
        # for layer in [self.first_hidden_layer, self.second_hidden_layer, self.softmax]:
        #     print(layer)

    def __call__(self, image):
        self.input_layer.reset_image(image)
        for layer in [self.first_hidden_layer, self.second_hidden_layer, self.softmax]:
            layer.reset_node_values()
        self._propagate_forward()
        return self.softmax.get_classification()