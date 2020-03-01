#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:50:47 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense

# Lendo banco de dados
df = pd.read_csv('Datasets/credit-data.csv')

coluns = df.columns
#  ANALISE DE DADOS
# --informações do banco
df.info()
# --descrição dos dados
df.describe()
# --dimensões dos dados
df.shape

# PRE-PROCESSAMENTO
# --Procurando por: age < 0.
df.loc[df['age'] < 0]
# --substituindo as idade que são menores que 0, pela media das idades.
df.loc[df.age < 0, 'age'] = df.age.mean()

# Trocando valores nulos pela media dos valores
impute_mean = SimpleImputer(missing_values= np.nan, strategy='mean')
df = impute_mean.fit_transform(df)

# Padronizando os dados, deixando na mesma escala
Scaler = StandardScaler()
df[:,:-1] = Scaler.fit_transform(df[:,:-1])

# SEPARANDO DADOS
# --Separando features de labels
X = df[:,1:-1] # features
y = df[:,-1]  # labels

# --Separando em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=0)


# criando rede neural.
model = Sequential()
# Camadas ocultas
model.add(Dense(units=2, activation='relu', input_dim=3))
model.add(Dense(units=2, activation='relu'))
# Camada de saida
model.add(Dense(units = 1, activation = 'sigmoid'))
# copilando rede neural
model.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['accuracy'])


model.fit(X_train,y_train,batch_size = 10, nb_epoch = 100)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
acc = accuracy_score(y_test, y_pred)
print('Taxa de acertos: ', acc * 100,'%')

matriz = pd.DataFrame(confusion_matrix(y_test, y_pred))
print('Matriz de confusão: \n',matriz)