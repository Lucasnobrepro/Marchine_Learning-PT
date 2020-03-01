#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:32:38 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Criando banco de dados
X=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]
plt.scatter(X,y)

df = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                 [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                 [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])

# Escalonamento
std = StandardScaler()
df = std.fit_transform(df)

# criando dendrograma
dendrograma = dendrogram(linkage(df, method = 'ward'))
plt.title('Dendrograma')
plt.xlabel('Pessoas')
plt.ylabel('Distância Euclidiana')

# Criando modelo
model = AgglomerativeClustering(n_clusters = 3,         # Numero de clusters
                                affinity = 'euclidean', # Calculo da distacia
                                linkage = 'ward')       # Lincagem de dados
y_pred = model.fit_predict(df)

# Visualização
plt.scatter(df[y_pred == 0, 0], df[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(df[y_pred == 1, 0], df[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(df[y_pred == 2, 0], df[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.legend()














