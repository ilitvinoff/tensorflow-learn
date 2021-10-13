import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage import transform
from skimage.color import rgb2gray
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
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(imread(f))
            labels.append(int(d))
    return np.array(images), np.array(labels)

def save_data(root,data,labels):
    for i,img in enumerate(data):
        path = os.path.join(root,str(labels[i]))
        if not os.path.exists(path):
            os.makedirs(path)
        imsave(os.path.join(path,f"{i}.png"),img)



def model_assembly(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(x_train, y_train, batch_size=62, epochs=3, validation_split=0.2)
    model.evaluate(x_test, y_test, batch_size=62)


seq_model = tf.keras.Sequential(
    [
        Input(shape=(28, 28, 3)),
        Conv2D(576, (3, 3), activation='relu'),
        Conv2D(1152, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(2304, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(1152, activation='relu'),
        Dense(62, activation='softmax')
    ]
)

seq_model_gray = tf.keras.Sequential(
    [
        Input(shape=(28, 28, 1)),
        Conv2D(192, (3, 3), activation='relu'),
        Conv2D(384, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(768, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(384, activation='relu'),
        Dense(62, activation='softmax')
    ]
)

ROOT_PATH = Path(__file__).resolve().parent.parent
train_data_directory = os.path.join(ROOT_PATH, "sign_dataset/Training")
test_data_directory = os.path.join(ROOT_PATH, "sign_dataset/Testing")

if __name__ == '__main__':
    images, labels = load_data(train_data_directory)
    images28 = np.array([transform.resize(image, (28, 28)) for image in images])
    images28_gray = rgb2gray(images28)
    images28_gray = np.expand_dims(images28_gray, -1)

    images28 = images28.astype('float32') / 255
    images28_gray = images28_gray.astype('float32') / 255

    images_test, labels_test = load_data(test_data_directory)
    images28_test = np.array([transform.resize(image, (28, 28)) for image in images_test])
    images28_gray_test = rgb2gray(images28_test)
    images28_gray_test = np.expand_dims(images28_gray_test, -1)

    images28_test = images28_test.astype('float32') / 255
    images28_gray_test = images28_gray_test.astype('float32') / 255

    save_data(os.path.join(ROOT_PATH,'sign_dataset/prepared/colored/training'),images28,labels)
    save_data(os.path.join(ROOT_PATH,'sign_dataset/prepared/colored/test'),images28_test,labels_test)
    save_data(os.path.join(ROOT_PATH,'sign_dataset/prepared/gray/training'),images28_gray,labels)
    save_data(os.path.join(ROOT_PATH,'sign_dataset/prepared/gray/test'),images28_gray_test,labels_test)

    print("images.shape = ", images.shape)
    print("labels.shape = ", labels.shape)
    print("images28.shape = ", images28.shape)
    print("images28_gray.shape = ", images28_gray.shape)
    print()
    print("images_test.shape = ", images_test.shape)
    print("labels_test.shape = ", labels_test.shape)
    print("images28_test.shape = ", images28_test.shape)
    print("images28_gray_test.shape = ", images28_gray_test.shape)

    show_random_images_on_plot(images28, labels, 4)
    show_random_images_on_plot(images28_gray, labels, 4)

    evaluate = False
    if evaluate:
        model_assembly(seq_model, images28, labels, images28_test, labels_test)
        model_assembly(seq_model_gray, images28_gray, labels, images28_gray_test, labels_test)
