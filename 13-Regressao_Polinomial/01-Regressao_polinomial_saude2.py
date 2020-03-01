#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 05:25:58 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
from sklearn.preprocessing import PolynomialFeatures

# Lendo banco de dados
df = pd.read_csv('Datasets/plano-saude2.csv')

# Separando dados
X = df.values[:,0]
y = df.values[:,1]

X = X.reshape(-1,1)


poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

# Criando Modelo
model = LinearRegression()
model.fit(X_poly,y)
y_pred = model.predict(poly.transform(X))

model.score(X_poly,y)

# Visualização
plt.plot(X,y_pred,color='red')        # plot da regressão
plt.scatter(x=X,y=y)                  # plot dos pontos
plt.title("Regressão Linear Simples") # titulo
plt.xlabel("Idade")                   # eixo X
plt.ylabel("Custo");                  # eixo Y




