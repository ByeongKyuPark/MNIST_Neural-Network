import numpy as np
from keras.datasets import mnist

#-------------------------------------------------functions
# Sigmoid function and its derivative
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Feed-forward function
def feed_forward(x_input, weights):
    activations = [np.vstack(([1], x_input))]  # Adding bias term to input layer
    signals = []  # Store z values

    for i, W in enumerate(weights):
        z = W.dot(activations[-1])
        signals.append(z)
        a = np.vstack(([1], sigmoid(z))) if i < len(weights) - 1 else sigmoid(z)
        activations.append(a)

    return activations, signals

# Backpropagation function
def backpropagation(activations, signals, weights, y_true):
    delta_L = activations[-1] - y_true  # Output layer error
    deltas = [None] * len(weights)
    deltas[-1] = delta_L

    for k in reversed(range(len(weights) - 2)):
        W_next_nobias = weights[k + 1][:, 1:]
        deltas[k] = W_next_nobias.T.dot(deltas[k + 1]) * sigmoid_derivative(signals[k])

    gradients = []
    for k in range(len(weights)):
        grad = deltas[k+1].dot(activations[k].T)#gW^k
        gradients.append(grad)

    return gradients

def to_vectors(y):
    one_hot = np.roll(np.eye(10), 1, axis=0)
    return one_hot[y].T

#-------------------------------------------------

# Define the network architecture
input_layer_size = 784  # 28x28 images flattened
s2 = 128  # Example hidden layer size, adjustable
output_layer_size = 10  # Digits 0-9

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data() #https://keras.io/api/datasets/mnist/
# Transform the training data to have images as columns
x_train = x_train.reshape(1, input_layer_size).T / 255.0  # (784, 60000)
x_test = x_test.reshape(1, input_layer_size).T / 255.0    # (784, 10000)

# 0. Map output values into vectors
y_train = to_vectors(y_train)
y_test = to_vectors(y_test)

# Initialize weights for the layers
# Weights between input layer and hidden layer
W1 = np.random.randn(s2, input_layer_size + 1) * 0.01  # Adding 1 for bias
# Weights between hidden layer and output layer
W2 = np.random.randn(output_layer_size, s2 + 1) * 0.01  # Adding 1 for bias

weights = [W1, W2]

# Training parameters
learning_rate = 0.1
epochs = 10  # Number of iterations over the entire training dataset

# Training loop
for epoch in range(epochs):
    for i in range(x_train.shape[1]):
        x = x_train[:, [i]]
        y = y_train[:, [i]]

        # Forward pass
        activations, signals = feed_forward(x, weights)

        # Backward pass
        gradients = backpropagation(activations, signals, weights, y)

        # Update weights
        for j in range(len(weights)):
            weights[j] -= learning_rate * gradients[j]

    print(f"Epoch {epoch + 1} complete")

# Testing the network
correct_predictions = 0
for i in range(x_test.shape[1]):
    x = x_test[:, [i]]
    y = y_test[:, [i]]

    activations, _ = feed_forward(x, weights)
    prediction = np.argmax(activations[-1], axis=0)
    true_label = np.argmax(y, axis=0)

    if prediction == true_label:
        correct_predictions += 1

accuracy = correct_predictions / x_test.shape[1]
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

#------------------------------------------------------
