#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 19:30:40 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

# Lendo banco de dados
df = pd.read_csv('Datasets/plano-saude.csv')

# Separando dados
X = df.values[:,0]
y = df.values[:,1]

X = X.reshape(-1,1)

# Criando Modelo
model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)


# Visualização
plt.plot(X,y_pred,color='red')        # plot da regressão
plt.scatter(x=X,y=y)                  # plot dos pontos
plt.title("Regressão Linear Simples") # titulo
plt.xlabel("Idade")                   # eixo X
plt.ylabel("Custo");                  # eixo Y

visual = ResidualsPlot(model)
visual.fit(X,y)
visual.poof()

# Valor de corelação ou score
model.score(X,y)




