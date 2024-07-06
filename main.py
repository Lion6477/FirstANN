import random
import matplotlib.pyplot as plt
import numpy as np
import utils

# my first ANN

# load dataset mnist.npz
images, labels = utils.load_mnist_dataset()

# Инициализация весов
weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))

# Инициализация смещения (bias)
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

learning_rate = 0.01
epochs = 5

for epoch in range(epochs):
    e_loss = 0  # Reset loss for each epoch
    e_correct = 0  # Reset accuracy for each epoch
    print(f"Epoch №{epoch + 1}")

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # forward propagation (to hidden layer)
        hidden_raw = weights_input_to_hidden @ image + bias_input_to_hidden
        # sigmoid activation function
        hidden = 1 / (1 + np.exp(-hidden_raw))

        # forward propagation (to output layer)
        output_raw = weights_hidden_to_output @ hidden + bias_hidden_to_output
        # sigmoid activation function
        output = 1 / (1 + np.exp(-output_raw))

        # Loss / Error calculation
        # MSE
        e_loss += np.sum((output - label) ** 2) / len(output)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # Backpropagation

        # Output layer
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        # Hidden layer
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

    # Print some debug info between epochs in percentages
    epoch_loss = (e_loss / images.shape[0]) * 100
    epoch_accuracy = (e_correct / images.shape[0]) * 100
    print(f"Loss: {round(epoch_loss, 3)}%")
    print(f"Accuracy: {round(epoch_accuracy, 3)}%")

# CHECK
test_image = random.choice(images)

# Predict
test_image = np.reshape(test_image, (-1, 1))

# forward propagation (to hidden layer)
hidden_raw = weights_input_to_hidden @ test_image + bias_input_to_hidden
# sigmoid activation function
hidden = 1 / (1 + np.exp(-hidden_raw))

# forward propagation (to output layer)
output_raw = weights_hidden_to_output @ hidden + bias_hidden_to_output
# sigmoid activation function
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"ANN suggests the number is: {output.argmax()}")
plt.show()
