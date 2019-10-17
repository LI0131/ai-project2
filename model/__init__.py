import logging
from config import NUM_EPOCHS
from model.input_layer import InputLayer
from model.hidden_layer import HiddenLayer
from model.output_layer import OutputLayer
from model.weight_matrix import WeightMatrix

logging.basicConfig(level=logging.INFO)


class Model:

    def __init__(self):
        self.input_layer = InputLayer()
        self.first_hidden_layer = HiddenLayer()
        self.second_hidden_layer = HiddenLayer()
        self.softmax = OutputLayer()
        self.first_weights = WeightMatrix(self.input_layer, self.first_hidden_layer)
        self.second_weights = WeightMatrix(self.first_hidden_layer, self.second_hidden_layer)
        self.third_weights = WeightMatrix(self.second_hidden_layer, self.softmax)

    def _propagate_forward(self):
        for matrix in [self.first_weights, self.second_weights, self.third_weights]:
            matrix.propagate_forward()
        self.softmax.set_error_matrix()

    def _propagate_backward(self):
        for matrix in [self.third_weights, self.second_weights, self.first_weights]:
            matrix.propagate_backward()

    def _reset_layer_values(self):
        for layer in [self.input_layer, self.first_hidden_layer, self.second_hidden_layer, self.softmax]:
            layer.reset_node_values()
            layer.reset_error_matrix()
        
    def train(self, images):
        logging.info(f'Beginning Training...')
        for epoch in range(NUM_EPOCHS):
            logging.info(f'Starting Epoch #{epoch}...')
            for image in images:
                self.input_layer.reset_image(image['image'])
                self.softmax.set_expected_output(image['label'])
                self._propagate_forward()
                self._propagate_backward()
                self._reset_layer_values()

    def test(self, images):
        logging.info(f'Beginning Testing...')
        num_correct = 0
        for image in images:
            self.input_layer.reset_image(image['image'])
            self.softmax.set_expected_output(image['label'])
            self._propagate_forward()
            if self.softmax.get_classification() == int(image['label']):
                num_correct += 1
            self._reset_layer_values()
        logging.info(f'Accuracy: {(num_correct/len(images)) * 100}%')

    def __call__(self, image):
        self.input_layer.reset_image(image)
        for layer in [self.first_hidden_layer, self.second_hidden_layer, self.softmax]:
            layer.reset_node_values()
        self._propagate_forward()
        return self.softmax.get_classification()
