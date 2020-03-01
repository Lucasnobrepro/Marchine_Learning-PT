#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:43:21 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# lendo banco de dados
df = pd.read_csv('Datasets/credit-card-clients.csv', header=1)

# Fazendo somatorio da divida
df['BILL_TOTAL'] = df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']

# Pegando features
X = df.iloc[:,[1,25]].values

# Escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)


# criando dendrograma
dendrograma = dendrogram(linkage(df, method = 'ward'))

# Criando modelo
model = AgglomerativeClustering(n_clusters = 3,         # Numero de clusters
                                affinity = 'euclidean', # Calculo da distacia
                                linkage = 'ward')       # Lincagem de dados
y_pred = model.fit_predict(df)

# Visualização
plt.scatter(df[y_pred == 0, 0], df[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(df[y_pred == 1, 0], df[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(df[y_pred == 2, 0], df[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.legend()















