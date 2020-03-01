#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:57:18 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# lendo banco de dados
df = pd.read_csv('Datasets/credit-card-clients.csv', header=1)

# Fazendo somatorio da divida
df['BILL_TOTAL'] = df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']

# Pegando features
X = df.iloc[:,[1,2,3,4,5,25]].values

# Escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Criando o modelo e testando o numero de clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')

# modelo final
kmeans = KMeans(n_clusters = 4, random_state = 0)
y_pred = kmeans.fit_predict(X)


# Lista final de clientes
lista_clientes = np.column_stack((df, y_pred))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]














