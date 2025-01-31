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

        layers = []
        biases = []
        for i in range((len(layer_sizes) - 1)):
            np.random.seed()
            layer = np.random.normal(0, 1, [layer_sizes[i + 1], layer_sizes[i]])
            bias = np.zeros([layer_sizes[i + 1], 1])
            layers.append(layer)
            biases.append(bias)
        self.layers = layers
        self.biases = biases

    def activation(self, x, function='sigmoid'):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param function: The activation function to be used.
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
        answer = x
        for i in range(len(self.layers)):
            if i < len(self.layers) - 1:
                answer = self.activation((self.layers[i] @ answer + self.biases[i][0]), "sigmoid")
            else:
                answer = self.activation((self.layers[i] @ answer + self.biases[i][0]), "sigmoid")
        return answer
