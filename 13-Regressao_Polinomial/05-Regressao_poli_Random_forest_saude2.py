#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:53:25 2020

@author: lucas
"""

# Bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


# Lendo banco de dados
df = pd.read_csv('Datasets/plano-saude2.csv')   

# Separando dados
X = df.iloc[:,0:1].values
y = df.iloc[:,1:2].values


std = StandardScaler()

X = std.fit_transform(X)
y = std.fit_transform(y)

# Criando modelo
model = SVR(kernel='rbf')
model.fit(X,y)
score = model.score(X,y)

# Criando teste para melhor visualização
X_test = np.arange(min(X),max(X),0.1)
X_test = X_test.reshape(-1,1)
y_pred = model.predict(X_test)

# Visualização
plt.plot(X_test,y_pred,color='red')        # plot da regressão
plt.scatter(x=X,y=y)                  # plot dos pontos
plt.title("Regressão por arvore") # titulo
plt.xlabel("Idade")                   # eixo X
plt.ylabel("Custo");                  # eixo Y
