#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:25:05 2020

@author: lucas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:09:39 2020

@author: lucas
"""

# BIBLIOTECAS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import KernelPCA

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
# --Trocando variaveis categoricas para inteiros, usando 
#   apply e lambda para fazer a codificação das variaveis.
labelencoder_previsores = LabelEncoder()
X[:, 1] = labelencoder_previsores.fit_transform(X[:, 1])
X[:, 3] = labelencoder_previsores.fit_transform(X[:, 3])
X[:, 5] = labelencoder_previsores.fit_transform(X[:, 5])
X[:, 6] = labelencoder_previsores.fit_transform(X[:, 6])
X[:, 7] = labelencoder_previsores.fit_transform(X[:, 7])
X[:, 8] = labelencoder_previsores.fit_transform(X[:, 8])
X[:, 9] = labelencoder_previsores.fit_transform(X[:, 9])
X[:, 13] = labelencoder_previsores.fit_transform(X[:, 13])

# Padronizando os dados, deixando na mesma escala
Scaler = StandardScaler()
X = Scaler.fit_transform(X)

# --Separando em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=0)


# MODELO
kpca = KernelPCA(n_components = 6, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)


# MODELO DE CLASSIFICADOR
model = RandomForestClassifier(n_estimators=40,criterion='entropy', random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

matriz = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test, y_pred)
print('Taxa de acertos: ', acc * 100,'%')













