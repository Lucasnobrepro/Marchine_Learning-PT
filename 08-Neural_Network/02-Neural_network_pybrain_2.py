#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:45:25 2020

@author: lucas
"""

# Bibliotcas
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer

# Criando rede neural
# ---Parametros: (-Camadas de entrada, -Camadas olcutas, Camadas de saida)
# ---Outros Parametros: outclass = SoftmaxLayer, hiddenclass = SigmoidLayer, bias = False
network = buildNetwork(2,3,1)

print('Camada de Entrada: ',network['in'])
print('Camada Oculta: ',network['hidden0'])
print('Camada de Saida: ',network['out'])
print('BIAS:', network['bias'])

# Criando base de dados 
base = SupervisedDataSet(2,1)
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

# Treinando
trainamento = BackpropTrainer(network, dataset = base, learningrate = 0.01,
                        momentum = 0.06)
# Epocas
for i in range(30000):
    erro = trainamento.train()
    if i % 1000 == 0:
        print("Erro: %s" % erro)

# Classificando
print(network.activate([0, 0]))
print(network.activate([1, 0]))
print(network.activate([0, 1]))
print(network.activate([1, 1]))















