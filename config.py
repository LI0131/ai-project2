import os


NUM_HIDDEN_NODES = int(os.environ.get('NUM_HIDDEN_NODES', 100))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 5))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 0.01))
NUM_OUTPUT_NODES = 10
IMAGE_SIZE = 784