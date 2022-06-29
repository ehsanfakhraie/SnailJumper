import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        # Initialize weights and biases
        self.W_1 = np.random.randn(layer_sizes[1], layer_sizes[0])
        self.b_1 = np.zeros((layer_sizes[1], 1))
        self.W_2 = np.random.randn(layer_sizes[2], layer_sizes[1])
        self.b_2 = np.zeros((layer_sizes[2], 1))

    def activation(self, x, function='sigmoid'):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)

        if function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif function == 'relu':
            return np.maximum(0, x)
        else:
            return x

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        # Forward propagation
        # input layer
        z_1 = np.dot(self.W_1, x) + self.b_1
        a_1 = self.activation(z_1)
        # hidden layer
        z_2 = np.dot(self.W_2, a_1) + self.b_2
        a_2 = self.activation(z_2)
        return a_2
