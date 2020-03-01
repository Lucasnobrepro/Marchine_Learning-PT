#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:15:07 2020

@author: lucas
"""

# Bibliotecas
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

# Gerando banco de dados aleatorio com make_blobs
X , y = make_blobs(n_samples=200, centers=4, random_state=0)

# plot 1
plt.scatter(X[:,0],X[:,1])

# Criando modelo
k_means = KMeans(n_clusters=4)
k_means.fit(X)
y_pred = k_means.predict(X)

# plot 2
plt.scatter(X[:,0],X[:,1],c=y_pred)