#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 11:25:41 2020

@author: lucas
"""

# Bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt

# Lendo banco de dados
df = pd.read_csv('Datasets/plano-saude2.csv')

# Separando dados
X = df.values[:,0]
y = df.values[:,1]

X = X.reshape(-1,1)

# Criando modelo
model = DecisionTreeRegressor()
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
