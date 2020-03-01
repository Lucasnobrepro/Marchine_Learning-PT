#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 08:06:58 2019

@author: lucas
"""

# BIBLIOTECAS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Lendo banco de dados
df = pd.read_csv('Datasets/risco-credito2.csv')

#  ANALISE DE DADOS
# --informações do banco
df.info()
# --descrição dos dados
df.describe()
# --dimensões dos dados
df.shape

# PRE-PROCESSAMENTO
# --Trocando variaveis categoricas para inteiros, usando 
#   apply e lambda para fazer a codificação das variaveis.
coluns = ['historia', 'divida', 'garantias', 'renda']
df[coluns] = df[coluns].apply(lambda coluns: LabelEncoder().fit_transform(coluns))

# DIVISÃO DOS DADOS
# --Dividindo as features e as labels
X = df.values[:,:-1]
y = df.values[:,-1]

# --Dividindo em treino e teste

# MODELO CLASSIFICADOR
model = LogisticRegression(random_state=1)
model.fit(X,y)
# Valor parametro B0
print(model.intercept_)
# Coeficientes para cada atributo
print(model.coef_)

# CASO DE TESTE
# --Historia ruim, Divida alta,Garantias nenhuma, renda >35
# --Historia ruim, Divida alta,Garantias adequada, renda < 15
teste = [[0,0,1,2],[3,0,0,0]]
# --Resultado da previsão
resultado = model.predict(teste)

print(resultado)
print(model.predict_proba(teste))




















