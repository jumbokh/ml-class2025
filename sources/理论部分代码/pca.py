# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:38:14 2020

@author: ThinkPad
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


iris = load_iris()
data, targets = scale(iris.data), iris.target
pca = PCA(n_components=2)
y = pca.fit_transform(data)
plt.scatter(y[targets==0][:,0], y[targets==0][:,1])
plt.scatter(y[targets==1][:,0], y[targets==1][:,1])
plt.scatter(y[targets==2][:,0], y[targets==2][:,1])
plt.show()