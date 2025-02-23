# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:39:29 2020

@author: ThinkPad
"""

from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor as GBDT
from sklearn.model_selection import train_test_split, validation_curve
import matplotlib.pyplot as plt


def validate():
    boston = load_boston()
    param_range = range(20, 150, 5)
    train_scores, val_scores = validation_curve(
        GBDT(max_depth=3), boston.data, boston.target, 
        param_name='n_estimators', 
        param_range=param_range, 
        cv=5,
        )
    train_mean = train_scores.mean(axis=-1)
    train_std = train_scores.std(axis=-1)
    val_mean = val_scores.mean(axis=-1)
    val_std = val_scores.std(axis=-1)
    
    _, ax = plt.subplots(1, 2)
    ax[0].plot(param_range, train_mean)
    ax[1].plot(param_range, val_mean)
    ax[0].fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax[1].fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.show()


def train():
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target)
    model = GBDT(n_estimators=50)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(train_score, test_score)


if __name__ == '__main__':
    validate()
    # train()
    
    