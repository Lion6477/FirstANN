import numpy as np

import utils

# Загрузка датасета mnist.npz
images, labels = utils.load_dataset()

# Инициализация весов
weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))

# Инициализация смещения (bias)
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

learning_rate = 0.01
epochs = 5
final_loss = 0
final_accuracy = 0

print("Start training")

for epoch in range(epochs):
    e_loss = 0  # Сброс потерь для каждой эпохи
    e_correct = 0  # Сброс точности для каждой эпохи
    print(f"Epoch №{epoch + 1}")

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # Прямое распространение (к скрытому слою)
        hidden_raw = weights_input_to_hidden @ image + bias_input_to_hidden
        hidden = 1 / (1 + np.exp(-hidden_raw))  # Сигмоидная функция активации

        # Прямое распространение (к выходному слою)
        output_raw = weights_hidden_to_output @ hidden + bias_hidden_to_output
        output = 1 / (1 + np.exp(-output_raw))  # Сигмоидная функция активации

        # Расчет потерь / ошибок
        e_loss += np.sum((output - label) ** 2) / len(output)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # Обратное распространение

        # Выходной слой
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        # Скрытый слой
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

    # Вывод отладочной информации между эпохами в процентах
    epoch_loss = (e_loss / images.shape[0]) * 100
    epoch_accuracy = (e_correct / images.shape[0]) * 100
    print(f"Loss: {round(epoch_loss, 3)}%")
    print(f"Accuracy: {round(epoch_accuracy, 3)}%")
    final_loss = epoch_loss
    final_accuracy = epoch_accuracy

print("Training completed")
print("Save model into file \"trained_model.npz\"? y/n")

if input().lower() == "y":
    name = ("trained_model" +
            "_loss-" + str(round(final_loss, 3)) +
            "_acc-" + str(round(final_accuracy, 3)) + ".npz")
    print("Saving as ")

    # Сохранение натренированной модели
    np.savez("model_loss_.npz",
             weights_input_to_hidden=weights_input_to_hidden,
             weights_hidden_to_output=weights_hidden_to_output,
             bias_input_to_hidden=bias_input_to_hidden,
             bias_hidden_to_output=bias_hidden_to_output)
    print("Model saved")
else:
    print("Model not saved")
