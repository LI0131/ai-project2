import os


NUM_HIDDEN_NODES = int(os.environ.get('NUM_HIDDEN_NODES', 100))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 1000))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 0.001))
TRAINING_PERCENTAGE = float(os.environ.get('TRAINING_PERCENTAGE', .6))
TESTING_PERCENTAGE = float(os.environ.get('TESTING_PERCENTAGE', .2))
VALIDATION_PERCENTAGE = float(os.environ.get('VALIDATION_PERCENTAGE', .2))
NUM_OUTPUT_NODES = 10
IMAGE_SIZE = 784