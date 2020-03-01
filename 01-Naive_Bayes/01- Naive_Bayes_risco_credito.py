#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 06:21:12 2019

@author: lucas nobre
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Lendo o banco de dados
data = pd.read_csv('Datasets/risco-credito.csv')

# Dividindo o banco de dados
X = data.values[:,:-1]
y = data.values[:,-1]

# Pre-processamento dos dados
labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])
X[:,1] = labelEncoder.fit_transform(X[:,1])
X[:,2] = labelEncoder.fit_transform(X[:,2])
X[:,3] = labelEncoder.fit_transform(X[:,3])


# Criando modelo
model = GaussianNB()
model.fit(X,y)

# CASO DE TESTE
# --Historia ruim, Divida alta,Garantias nenhuma, renda >35
# --Historia ruim, Divida alta,Garantias adequada, renda < 15
teste = [[0,0,1,2],[3,0,0,0]]
# --Resultado da previsão
resultado = model.predict(teste)

# Classes que o modelo prever
print(model.classes_)
# Probabilidade para cada classe
print(model.class_prior_)
# Qunatidade de cada classe
print(model.class_count_)






















