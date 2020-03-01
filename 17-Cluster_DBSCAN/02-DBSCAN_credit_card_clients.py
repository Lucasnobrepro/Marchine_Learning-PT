#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:51:36 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# lendo banco de dados
df = pd.read_csv('Datasets/credit-card-clients.csv', header=1)

# Fazendo somatorio da divida
df['BILL_TOTAL'] = df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']

# Pegando features
X = df.iloc[:,[1,25]].values

# Escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = DBSCAN(eps = 0.37, 
                min_samples = 4)
y_pred = model.fit_predict(X)
unicos, quantidade = np.unique(y_pred, return_counts = True)

plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'orange', label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

lista_clientes = np.column_stack((df, y_pred))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]



