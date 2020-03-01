#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:59:16 2020

@author: lucas
"""
# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# --models imports
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# --biblioteca para salvar modelos
import pickle

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

# Criando modelo SVM
model_SVM = SVC(kernel = 'rbf', C = 2.0)
model_SVM.fit(X, y)

# Criando modelo de Random Forest
model_RandomForest = RandomForestClassifier(n_estimators = 40, criterion = 'entropy')
model_RandomForest.fit(X, y)

# Criando modelo de Rede Neural
model_MLP = MLPClassifier(verbose = True, max_iter = 1000,
                                 tol = 0.000010, solver = 'adam',
                                 hidden_layer_sizes=(100), activation = 'relu',
                                 batch_size = 200, learning_rate_init = 0.001)
model_MLP.fit(X, y)


# Salvando modelos
pickle.dump(model_SVM, open('svm_finalizado.sav', 'wb'))
pickle.dump(model_RandomForest, open('random_forest_finalizado.sav', 'wb'))
pickle.dump(model_MLP, open('mlp_finalizado.sav', 'wb'))

pickle.dump()