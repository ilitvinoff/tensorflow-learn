import random

import matplotlib.pyplot as plt


def show_random_images_on_plot(array, labels, limit):
    for i, index in enumerate([random.randint(0, len(array) - 1) for i in range(limit)]):
        plt.subplot(1, limit, i + 1)
        plt.axis('off')
        plt.title(labels[index])
        plt.imshow(array[index])
        plt.subplots_adjust(wspace=0.5)
    plt.show()
