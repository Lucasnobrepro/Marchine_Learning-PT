#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 05:11:08 2020

@author: lucas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 04:20:49 2020

@author: lucas
"""

# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

# Lendo banco de dados
df = pd.read_csv('Datasets/house-prices.csv')

df.info()

df.shape
df.isnull().count()

# Separando dados em features e labels.
X = df.iloc[:,3:19].values
y = df.iloc[:,2].values

# Separando dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state= 0)

# Criação do modelo
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

model.score(X_train,y_train)

mean_absolute_error(y_test, y_pred)






