# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:28:08 2020

@author: ThinkPad
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


if __name__ == '__main__':
    wine = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        wine.data, wine.target)
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print("train score:", train_score)
    print("test score:", test_score)