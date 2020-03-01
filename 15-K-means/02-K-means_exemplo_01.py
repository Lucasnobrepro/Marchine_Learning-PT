#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:04:42 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# X = idade, y = Salario
x=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]  

# plot 1
plt.scatter(x,y)

# Criando base de dados com 15 registros
df = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                 [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                 [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])

# Fazendo escalonamento da base de dados
std = StandardScaler()
df = std.fit_transform(df)

# Criando modelo
k_means = KMeans(n_clusters=3)
k_means.fit(df)

# Verificando centroids e labels
centers = k_means.cluster_centers_
cluster_labels = k_means.labels_

# plot 2
cores = ["g.", "r.", "b."]
for i in range(len(x)):
    plt.plot(df[i][0], df[i][1], cores[cluster_labels[i]], markersize = 15)
plt.scatter(centers[:,0], centers[:,1], marker = "x")
