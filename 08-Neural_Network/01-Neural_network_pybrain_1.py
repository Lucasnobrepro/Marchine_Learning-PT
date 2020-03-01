#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:10:53 2020

@author: lucas
"""

# Bibliotecas
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

# structure - Definine estruturas.
# FeedForwardNetwork - Tipo da rede neural.
# LinearLayer, SigmoidLayer, BiasUnit - Tipos de camada e o BIAS.
# FullConnection - Servira para fazer as ligações entre as classes.

# Criando Rede neural
network = FeedForwardNetwork()

# Criando camada de entrada.
# parametros: (-quantidade de neuronios na camada)
inputLayer = LinearLayer(dim=2)

# Criando camada oculta.
# parametros: (-quantidade de neuronios na camada)
hiddenLayer = SigmoidLayer(dim=3)

# Criando camada de saida.
# parametros: (-quantidade de neuronios na camada)
outLayer = SigmoidLayer(dim=1)

# Criando BIAS, 1 para camada oculta e 1 para camada de saida.
bias1 = BiasUnit()
bias2 = BiasUnit()

# Adicionar camadas e BIAS a rede.
# ---Adicionando camadas
network.addModule(inputLayer)
network.addModule(hiddenLayer)
network.addModule(outLayer)
# ---Adicionando BIAS
network.addModule(bias1)
network.addModule(bias2)

# Ligação de camadas:
input_Hidden = FullConnection(inputLayer, hiddenLayer)# Entrada para oculta.
hidden_Out = FullConnection(hiddenLayer, outLayer)    # Oculta para saida.
bias_Hidden = FullConnection(bias1, hiddenLayer)      # Bias para oculta
bias_out = FullConnection(bias2, outLayer)            # Bias para saida

# Contrução
network.sortModules()

print(network) # Mostra o tipo de camadas
print(input_Hidden.params) # mostras os pesos.
print(hidden_Out.params) # mostras os pesos.
print(bias_Hidden.params) # mostras os pesos.
print(bias_out.params) # mostras os pesos.










