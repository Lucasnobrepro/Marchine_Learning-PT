#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:39:47 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
from apyori import apriori

# Lendo banco de dados
df = pd.read_csv('Datasets/mercado2.csv',header=None)

# Convertendo Dataframe para Lista
transacoes = []
for i in range(0, 7501):
    transacoes.append([str(df.values[i,j]) for j in range(0, 20)])
    
# Regras
rules = apriori(transacoes, 
                min_support=0.003, 
                min_confidence=0.2,
                min_lift=3,
                min_leght=2)    

resultados = list(rules)

# Organizando regras
resultados2 = [list(x) for x in resultados]
resultados_formart = []
for j in range(0,5):
    resultados_formart.append([list(x) for x in resultados2[j][2]])

resultados_formart

