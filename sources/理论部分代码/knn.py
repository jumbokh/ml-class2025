# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:31:05 2020

@author: ThinkPad
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt


def plot_decision_boundary(model, xlim: [int, int], ylim: [int, int], fill=False):
    x, y = np.meshgrid(
        np.linspace(*xlim, 101),
        np.linspace(*ylim, 101),
        )
    data = np.c_[x.ravel(), y.ravel()]
    pred = model.predict(data)
    z = pred.reshape(x.shape)
    if fill: plt.contourf(x, y, z)
    else: plt.contour(x, y, z)


if __name__ == '__main__':
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data[:, [1,3]], iris.target)
    model = KNN(1)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print("train score:", train_score)
    print("test score:", test_score)
    
    # plot_decision_boundary(model, xlim=[1.5, 5], ylim=[0, 3])
    for target in range(3):
        plt.scatter(
            iris.data[iris.target == target, 1], 
            iris.data[iris.target == target, 3],
            marker=['o', '+', 'x'][target],
            )
    plt.show()
