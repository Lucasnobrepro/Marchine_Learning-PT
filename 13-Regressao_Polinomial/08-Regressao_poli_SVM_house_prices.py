#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 08:37:13 2020

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
df = pd.read_csv('Datasets/house-prices.csv')

# Separando features de labels
X = df.values[:,3:19]
y = df.values[:,2:3]

# Fazendo escalonamento
std = StandardScaler()
X = std.fit_transform(X)
y = std.fit_transform(y)

# Separando em treinamento e teste
X_train,X_test , y_train, y_test =  train_test_split(X, y, 
                                                     test_size=0.3,
                                                     random_state=0)

# Criando modelo
model = SVR()
model.fit(X_train,y_train)
y_pred = std.inverse_transform(model.predict(X_test))
score_train = model.score(X_train,y_train)

score_test = model.score(X_test,y_test)

mae = mean_absolute_error(y_test,y_pred)
