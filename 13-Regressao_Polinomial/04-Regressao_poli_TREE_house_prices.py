#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:15:34 2020

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
df = pd.read_csv('Datasets/house-prices.csv')

# Separando features de labels
X = df.values[:,3:19]
y = df.values[:,2]

# Separando em treinamento e teste
X_train,X_test , y_train, y_test =  train_test_split(X, y, 
                                                     test_size=0.3,
                                                     random_state=0)

# Criando modelo
model = DecisionTreeRegressor()
model.fit(X_train,y_train)
score = model.score(X_train,y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)




