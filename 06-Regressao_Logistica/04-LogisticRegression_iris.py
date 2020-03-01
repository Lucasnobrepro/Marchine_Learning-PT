#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:50:06 2020

@author: lucas
"""

# Biblotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

# Lendo banco de dados
df = pd.read_csv('Datasets/iris.data',header=None)
df.head()

# Conversão das variaveis categoricas em variaveis numericas, utilizando mapa.
label_map = {'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2}
df[4] = df[4].map(label_map)

# Separando a label das features
X = df.values[:, :-1]
y = df.values[:,-1]

# Estratificação de dados e divisão de teste e treino
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, 
                                                    stratify=y,random_state=1)


# Projeção no grafico de 4 dimensões (cor é a 4º Dimensão apresentada)
# -------Execute ao proximas 4 linhas juntas-------#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 3], cmap=plt.viridis());
fig.colorbar(img);
# -------------------------------------------------#


# MODELO CLASSIFICADOR
# A acuracia vai varias de qual o melhor dependendo da quantidade de vizinhos
# analisada.

#--modelo sem scale
# --kernel serve para classificar com multiclass
model = LogisticRegression(random_state=1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
matriz = confusion_matrix(y_test,y_pred)
print("Acerto sem scale: ",acc * 100,'%')
print(matriz)


