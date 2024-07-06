import random

import matplotlib.pyplot as plt
import numpy as np

import utils

# Загрузка натренированной модели
weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output = utils.load_model(
    "trained_model.npz")

# Загрузка датасета mnist.npz
images, labels = utils.load_dataset()

# Выбор случайного изображения из датасета
test_image = random.choice(images)

# Прогнозирование
test_image = np.reshape(test_image, (-1, 1))

# Прямое распространение (к скрытому слою)
hidden_raw = weights_input_to_hidden @ test_image + bias_input_to_hidden
hidden = 1 / (1 + np.exp(-hidden_raw))  # Сигмоидная функция активации

# Прямое распространение (к выходному слою)
output_raw = weights_hidden_to_output @ hidden + bias_hidden_to_output
output = 1 / (1 + np.exp(-output_raw))  # Сигмоидная функция активации

# Визуализация результата
plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"ANN suggests the number is: {output.argmax()}")
plt.show()
