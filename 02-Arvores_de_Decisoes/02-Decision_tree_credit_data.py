#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 08:44:45 2019

@author: lucas
"""

# BIBLIOTECAS
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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
    
# MODELO DE CLASSIFICADOR
model = model = DecisionTreeClassifier(criterion='entropy', random_state=1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

matriz = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print('accuracy: ',acc * 100,'%')
print(matriz)

f_importance = np.array([coluns[:-1], model.feature_importances_])
f_importance = pd.DataFrame(f_importance)
print('features mais importantes\n',f_importance.head(2))
















