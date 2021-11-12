from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_unique_combinations(x1_list, x2_list):
    ret = []
    for x1 in x1_list:
        for x2 in x2_list:
            ret.append((x1, x2))

    return ret


def show_scatter_matrix(X, y, alpha=0.1):
    for x_comb in combinations(X, 2):
        x1_uniques = X[x_comb[0]].unique()
        x2_uniques = X[x_comb[1]].unique()

        color = {0: 'red', 1: 'green'}
        plt.xlim(x1_uniques.min(), x1_uniques.max())
        plt.ylim(x2_uniques.min(), x2_uniques.max())
        plt.xlabel(x_comb[0])
        plt.ylabel(x_comb[1])
        unique_combinations = get_unique_combinations(x1_uniques, x2_uniques)

        for comb in unique_combinations:
            indices = X.index
            condition = (X[x_comb[0]] == comb[0]) & (X[x_comb[1]] == comb[1])
            y_list_indices = indices[condition]

            for i in y_list_indices:
                plt.plot(comb[0], comb[1], color[y[i]], marker='s', alpha=alpha)
        plt.show()


def plot_decision_regions(X_full, y, classifier, resolution=1):
    x_headers = X_full.columns
    colors = ('red', 'green')

    x_min = [X_full[x].to_numpy().min() for x in X_full]
    x_max = [X_full[x].to_numpy().max() + 1 for x in X_full]

    features_grid = np.array(np.meshgrid(*[np.arange(x_min, x_max, resolution) for x_min, x_max in zip(x_min, x_max)]))
    features_grid = features_grid.reshape((len(x_min), int(features_grid.size / len(x_min)))).T

    Z = classifier.predict(features_grid)
    df = pd.DataFrame(features_grid, columns=x_headers)
    pd.plotting.scatter_matrix(df, c=([colors[v] for v in Z]), alpha=0.1)
    plt.show()

    show_scatter_matrix(df, Z)
