from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras import optimizers

class NeuralNetwork(Sequential):
    def __init__(self, learning_rate=0.05):
        super().__init__()