#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 07:48:46 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lendo banco de dados
df = pd.read_csv('Datasets/iris.data',header= None)

df.head()


# Separando features de labels
X = df.iloc[:,:-1].values
y = df.iloc[:,4:5].values

# codificando label
label_enconder = LabelEncoder()
y = label_enconder.fit_transform(y)

# Escalanamento
std = StandardScaler()
X = std.fit_transform(X)

# Criando modelo
k_means = KMeans(n_clusters=3,  # Numero de clusters.
                 max_iter=100,  # Quantidade de interações maxima.
                 random_state=1,# Semente randomica.
                 n_jobs=2)      # Quantidae de theads.
k_means.fit(X)#------------------ Treinamento do modelo.
y_pred = k_means.predict(X)#----- Predição.

# accuracy
acc = accuracy_score(y,y_pred)
print('Acertos: ', acc * 100)


# Visualização
# 01 - Visualizando X e y em 3D
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')

img = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.viridis());
fig.colorbar(img);

# 02 - Visualizando X e y_pred em 3D
fig = plt.figure()
ax = fig.add_subplot(222, projection='3d')

img = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, cmap=plt.viridis());
fig.colorbar(img);

# 03 - Visualizando X e y
plt.subplot(221), plt.scatter(X[:,0],X[:,1],c=y)
plt.title('01 - X[:,0] e X[:,1] com y')
plt.subplot(222), plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.title('01 - X[:,0] e X[:,1] com y_pred')










