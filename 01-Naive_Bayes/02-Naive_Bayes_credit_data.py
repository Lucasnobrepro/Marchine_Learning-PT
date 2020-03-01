#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:22:26 2019

@author: lucas nobre
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Lendo dataframe
df = pd.read_csv('Datasets/credit-data.csv')

# PRE-PROCESSAMENTO DE DADOS
# --Corpo do dataFrame
df.shape

# --Procuro inconsistências nas idades
df.loc[df['age'] < 0]

# --Informações do dataFrame
df.info()

# --Media das idades
ageMean = df.age.mean()

# --Aplico a media nas inconsistências 
df.loc[df.age < 0, 'age'] = ageMean

# SEPARANDO DADOS
# --Separando features de labels
X = df.values[:,1:-1] # features
y = df.values[:,-1]  # labels

# --Substituição de valores nulos(nan)
impute_mean = SimpleImputer(missing_values=np.nan,strategy="mean")
X = impute_mean.fit_transform(X)

# Padronizando os dados
Scaler = StandardScaler()
X = Scaler.fit_transform(X)

# --Separando em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=0)
# TREINAMENTO
# --Criando modelo
model = GaussianNB()
model.fit(X_train,y_train)
# --fazendo previsões dos dados de teste
y_pred = model.predict(X_test)

# --Medidas de Acertos
score_ = accuracy_score(y_test,y_pred)
print("Porcentagem de acertos: ",score_ * 100, "%")

# Matriz de confusão
matriz = confusion_matrix(y_test,y_pred)

# FIM