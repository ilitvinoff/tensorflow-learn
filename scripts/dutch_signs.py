import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage.io import imread, imsave
from tensorflow.keras.layers import (Conv2D,
                                     Input,
                                     Dense,
                                     MaxPool2D,
                                     BatchNormalization,
                                     GlobalAvgPool2D)

from scripts.utils import show_random_images_on_plot


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm") or f.endswith(".png")]
        for f in file_names:
            images.append(imread(f))
            labels.append(int(d))
    return np.array(images), np.array(labels)


def save_data(root, data, labels):
    for i, img in enumerate(data):
        path = os.path.join(root, str(labels[i]))
        if not os.path.exists(path):
            os.makedirs(path)
        imsave(os.path.join(path, f"{i}.png"), img)


seq_model = tf.keras.Sequential(
    [
        Input(shape=(28, 28, 3)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(62, activation='softmax')
    ]
)

seq_model_gray = tf.keras.Sequential(
    [
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(62, activation='softmax')
    ]
)

ROOT_PATH = Path(__file__).resolve().parent.parent
train_colored_data_directory = os.path.join(ROOT_PATH, 'sign_dataset/prepared/colored/training')
test_colored_data_directory = os.path.join(ROOT_PATH, 'sign_dataset/prepared/colored/test')

train_gray_data_directory = os.path.join(ROOT_PATH, 'sign_dataset/prepared/gray/training')
test_gray_data_directory = os.path.join(ROOT_PATH, 'sign_dataset/prepared/gray/test')


def print_shape(loaded_data: dict):
    for k, v in loaded_data.items():
        print(f"{k}.shape: {v.shape}")


def model_assembly(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(x_train, y_train, batch_size=62, epochs=30, validation_split=0.2)
    model.evaluate(x_test, y_test, batch_size=62)


if __name__ == '__main__':
    images28_colored, colored_labels = load_data(train_colored_data_directory)
    images28_colored_test, colored_test_labels = load_data(test_colored_data_directory)

    # images28_gray, gray_labels = load_data(train_gray_data_directory)
    # images28_gray_test, gray_test_labels = load_data(test_gray_data_directory)

    # images28_colored = np.expand_dims(images28_colored, -1)
    # images28_colored_test = np.expand_dims(images28_colored_test, -1)

    images28_colored = images28_colored.astype('float32') / 255
    images28_colored_test = images28_colored_test.astype('float32') / 255

    # images28_colored = np.expand_dims(images28_colored, -1)
    # images28_colored_test = np.expand_dims(images28_colored_test, -1)

    show_random_images_on_plot(images28_colored, colored_labels, 4)

    print_shape({'images28_gray': images28_colored, 'gray_labels': colored_labels,
                 'images28_gray_test': images28_colored_test, 'gray_test_labels': colored_test_labels})

    evaluate = True
    if evaluate:
        model_assembly(seq_model, images28_colored, colored_labels, images28_colored_test, colored_test_labels)
