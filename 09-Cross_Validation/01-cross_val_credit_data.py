#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 07:10:56 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

# Lendo dados
base = pd.read_csv('Datasets/credit-data.csv')

# Substituindo valores negativos pela media da idade.
base.loc[base.age < 0, 'age'] = 40.92
               
# Separando dados.
train = base.iloc[:, 1:4].values
test = base.iloc[:, 4].values

# Substituindo valores faltantes pela media.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(train[:, 1:4])
train[:, 1:4] = imputer.transform(train[:, 1:4])

# Fazendo escalonamento dos dados.
scaler = StandardScaler()
train = scaler.fit_transform(train)

# Criando Modelo.
model = GaussianNB()
# treinamento com validação cruzada e predição dos casos de teste.
y_pred = cross_val_score(model, train, test, cv = 10)
y_pred.mean()
y_pred.std()


