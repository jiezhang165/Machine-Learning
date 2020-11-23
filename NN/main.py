# Given these data point, we aim at learning a predictor
# using the support vector machine method
# We start by a simple case of separable data
import numpy as np
import math
from sklearn.model_selection import train_test_split
from numpy import exp, array, random, dot

n_samples = 200
n_features = 10
X = np.random.rand(n_samples, n_features)
y = np.zeros((n_samples, 1))
y[np.where(X[:, 0] < X[:, 5])] = 1

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
n_samples = np.size(x_train[:, 1])
n_teste_samples = np.size(x_test[:, 1])

# recall y_hat = threshold(w.X^T)
threshold = lambda x: 1 if x > 0.5 else 0
logistic = lambda x: 1 / (1 + math.exp(-x))
deriv_logistic = lambda x: (logistic(x)) / (1 - logistic(x))

x0 = np.ones((n_samples, 1))
x0_test = np.ones((n_teste_samples, 1))
samples = np.concatenate((x0, x_train), axis=1)
samples_test = np.concatenate((x0_test, x_test), axis=1)


def linear_classifier_activation(samples, labels, alpha=0.01):
    n_features = samples.shape[1]
    n_samples = samples.shape[0]

    w = 2 * np.random.rand(1, n_features) - 1  # center on 0
    cmpt = 0
    while (cmpt < 1000):
        for i in range(n_samples):
            linear_response = np.dot(w, samples[i, :].T)
            y_hat = logistic(linear_response)
            y_hat = threshold(y_hat)
            w = w + alpha * (labels[i] - y_hat) * deriv_logistic(linear_response) * samples[i, :]
            cmpt += 1
    return w


accuracy_rate = 0
w = linear_classifier_activation(samples, y_train, alpha=0.01)

y_hat = np.zeros(n_teste_samples)
for i in range(n_teste_samples):
    linear_response = np.dot(w, samples_test[i, :].T)
    y_hat[i] = logistic(linear_response)
    y_hat[i] = threshold(y_hat[i])
    if (y_test[i] == y_hat[i]):
        accuracy_rate += 1

accuracy_rate /= n_teste_samples
# print(y_hat.reshape(n_teste_samples,1)-y_test)
print(accuracy_rate)


# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='sgd',activation = 'relu',max_iter = 1000,
#                     alpha = 1e-5,hidden_layer_sizes = (100,50),random_state = 1,verbose = True)
# clf.fit(x_train, y_train)
# print(clf.predict(x_test))
# print(clf.score(x_test,y_test))
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            # layer2_error = training_set_outputs - output_from_layer_2  # E(2) = A(2) - Y
            # layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)  # dZ(2) = E(2) * g'(Z(2))
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_error = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            # layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)  # E(1) = dZ(2) * W(2).T
            # layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)  # dZ(1) = E(1) * g'(Z(1))
            layer1_error = layer2_error.dot(self.layer2.synaptic_weights.T)
            layer1_error = layer1_error * (self.__sigmoid_derivative(output_from_layer_1))

            # # Calculate how much to adjust the weights by
            # layer1_adjustment = training_set_inputs.T.dot(layer1_delta)  #dW(1) = X.T * dZ(1)
            # layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)  #dW(2) = A(1).T * dZ(2)
            layer1_delta = training_set_inputs.T.dot(layer1_error)
            layer2_delta = output_from_layer_1.T.dot(layer2_error)
            layer1_adjustment = layer1_delta
            layer2_adjustment = layer2_delta
            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (5 neurons, each with 10 inputs):")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (1 neuron, with 10 inputs):")
        print(self.layer2.synaptic_weights)


if __name__ == "__main__":

    # Seed the random number generator
    random.seed(1)

    # Create layer 1 (10 neurons, each with 10 inputs)
    layer1 = NeuronLayer(10, 10)

    # Create layer 2 (a single neuron with 10 inputs)
    layer2 = NeuronLayer(1, 10)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = x_train
    training_set_outputs = y_train.reshape(n_samples, 1)

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # # Test the neural network with a new situation.
    output = np.zeros((n_teste_samples, 1))
    for i in range(n_teste_samples):
        hidden_state, output[i] = neural_network.think(x_test[i, :])
        output[i] = threshold(output[i])

Eurror = output - y_test
score = 0
for i in range(len(Eurror)):
    if Eurror[i] == 0:
        score += 1
score = score / len(Eurror)
print("The predicted score is: ", score)
