#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:53:56 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Lendo dados
base = pd.read_csv('Datasets/credit-data.csv')

# Substituindo valores negativos pela media da idade.
base.loc[base.age < 0, 'age'] = 40.92
               
# Separando dados.
X = base.iloc[:, 1:4].values
y = base.iloc[:, 4].values

# Substituindo valores faltantes pela media.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:4])
X[:, 1:4] = imputer.transform(X[:, 1:4])

# Fazendo escalonamento dos dados.
scaler = StandardScaler()
X = scaler.fit_transform(X)

X.shape[0]
a = np.zeros(5)
b = np.zeros(shape=(X.shape[0],1))

kfold = StratifiedKFold(n_splits=10,
                        shuffle=True,
                        random_state=0  )

result = []

# Criando Modelo com K-folds.
for id_train, id_test in kfold.split(X,
                              np.zeros(shape= (X.shape[0], 1))):
    
    model = GaussianNB()
    model.fit(X[id_train],y[id_train])
    y_pred = model.predict(X[id_test])
    acc = accuracy_score(y[id_test],y_pred)
    result.append(acc)


media_acc = np.mean(result)
print("acc: ", media_acc * 100)








