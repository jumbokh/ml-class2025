# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:34:07 2020

@author: ThinkPad
"""

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def score(pred, gt):
    assert len(pred) == len(gt)
    m = len(pred)
    
    map_ = {}
    for c in set(pred):
        map_[c] = stats.mode(gt[pred == c])[0]
    score = sum([map_[pred[i]] == gt[i] for i in range(m)])
    return score[0] / m


if __name__ == '__main__':
    iris = load_iris()
    model = KMeans(n_clusters=3)
    pred = model.fit_predict(iris.data)
    print(score(pred, iris.target))
        
    _, axes = plt.subplots(1, 2)
    axes[0].set_title("ground truth")
    axes[1].set_title("prediction")
    for target in range(3):
        axes[0].scatter(
            iris.data[iris.target == target, 1], 
            iris.data[iris.target == target, 3],
            )
        axes[1].scatter(
            iris.data[pred == target, 1], 
            iris.data[pred == target, 3],
            )
    plt.show()
    