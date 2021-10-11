import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from skimage.io import imread
from skimage.color import rgb2gray
import tensorflow as tf

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


def show_random_images_on_plot(array, limit):
    for i, index in enumerate([random.randint(0, len(array)) for i in range(limit)]):
        plt.subplot(1, limit, i + 1)
        plt.axis('off')
        plt.imshow(array[index])
        plt.subplots_adjust(wspace=0.5)
    plt.show()


ROOT_PATH = Path(__file__).resolve().parent.parent
train_data_directory = os.path.join(ROOT_PATH, "sign_dataset/Training")
test_data_directory = os.path.join(ROOT_PATH, "sign_dataset/Testing")

images, labels = load_data(train_data_directory)
images28 = np.array([transform.resize(image, (28, 28)) for image in images])
images28_gray = rgb2gray(images28)

show_random_images_on_plot(images28, 4)
show_random_images_on_plot(images28_gray, 4)

