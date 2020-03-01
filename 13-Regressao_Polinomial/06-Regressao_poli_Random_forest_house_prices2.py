#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 07:04:31 2020

@author: lucas
"""

# Bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train)
score_train = model.score(X_train,y_train)

score_test = model.score(X_test,y_test)



