import numpy as np


def load_dataset():
    with np.load("mnist.npz") as f:
        # Конвертация изображений из RGB в единичный RGB
        x_train = f['x_train'].astype(np.float32) / 255
        x_train = x_train.reshape(x_train.shape[0], -1)

        # Метки
        y_train = f['y_train']
        y_train = np.eye(10)[y_train]

        return x_train, y_train


def load_model(filepath):
    data = np.load(filepath)
    weights_input_to_hidden = data['weights_input_to_hidden']
    weights_hidden_to_output = data['weights_hidden_to_output']
    bias_input_to_hidden = data['bias_input_to_hidden']
    bias_hidden_to_output = data['bias_hidden_to_output']
    return weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output
