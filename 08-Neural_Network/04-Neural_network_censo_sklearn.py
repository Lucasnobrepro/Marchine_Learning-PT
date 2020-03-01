#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:15:54 2020

@author: lucas
"""

# BIBLIOTECAS
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neural_network import MLPClassifier

# Lendo banco de dados
df = pd.read_csv('Datasets/census.csv')
#  ANALISE DE DADOS
# --informações do banco
df.info()
# --descrição dos dados
df.describe()
# --dimensões dos dados
df.shape

# SEPARANDO DADOS
# --Separando features de labels
X = df.values[:,:-1] # features
y = df.values[:,-1]  # labels

# PRE-PROCESSAMENTO
# --Trocando variaveis categoricas para inteiros
labelencoder_previsores = LabelEncoder()
X[:, 1] = labelencoder_previsores.fit_transform(X[:, 1])
X[:, 3] = labelencoder_previsores.fit_transform(X[:, 3])
X[:, 5] = labelencoder_previsores.fit_transform(X[:, 5])
X[:, 6] = labelencoder_previsores.fit_transform(X[:, 6])
X[:, 7] = labelencoder_previsores.fit_transform(X[:, 7])
X[:, 8] = labelencoder_previsores.fit_transform(X[:, 8])
X[:, 9] = labelencoder_previsores.fit_transform(X[:, 9])
X[:, 13] = labelencoder_previsores.fit_transform(X[:, 13])

onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
X = onehotencoder.fit_transform(X).toarray()

y = labelencoder_previsores.fit_transform(y)
# Padronizando os dados, deixando na mesma escala
Scaler = StandardScaler()
X = Scaler.fit_transform(X)

# --Separando em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=0)
    
# MODELO DE CLASSIFICADOR
model = MLPClassifier(activation='relu',
                      solver='adam',
                      max_iter=1000,
                      random_state=1,
                      tol=0.00001,
                      verbose=True)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Taxa de acertos: ', round(acc * 100,2),'%')

matriz = pd.DataFrame(confusion_matrix(y_test, y_pred))
print('Matriz de confusão: \n',matriz)




