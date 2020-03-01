#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:40:44 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# Criando banco de dados
X=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]
plt.scatter(X,y)

df = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                 [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                 [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])

# Escalonamento
std = StandardScaler()
df = std.fit_transform(df)


# Criando modelo
model = DBSCAN(eps=0.95,         #threshold     
               min_samples=2)  #amostras minimas

y_pred = model.fit_predict(df)

cores = ["g.", "r.", "b."]
for i in range(len(df)):
    plt.plot(df[i][0], df[i][1], cores[y_pred[i]], markersize = 15)

