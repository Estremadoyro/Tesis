# -*- coding: utf-8 -*-
import tensorflow as tf 
import pandas as pd
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Input
from numpy import loadtxt

dataset = pd.read_csv('dataset.csv')
#X atributos de entrada 
X = dataset[:, 0:3]
#y es el target que queremos predecir  
y = dataset[:, 3]

#---------------------------#
#        Topologia          #
#         [3-2-1]           #
#---------------------------#

model = Sequential([
    Input(input_shape = X.shape)
    BatchNormalization(),
    #glorot_normal = Xaviers Initialization como tecnica inicializar los pesos de las conexiones
    Dense(3, kernel_initializer="glorot_normal", use_bias=False),
    #Ayuda a evitar el desvanecimiento de la grandiente obtenida de backpropagation
    BatchNormalization(),
    #Funcion de actviacion escogida fue TangenteHiperbolica [-1,+1]
    Activation("tanh"),
    #glorot_normal = Xaviers Initialization como tecnica inicializar los pesos de las conexiones
    Dense(2, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    #Capa de salida con 1 neurona 
    Dense(1)
])

#Utilizacion de Stochastic Gradient Deschent para la optimizacion de la topologia 
optimizer = SGD()

#MSE como funcion de costo, para determinar el error. 
model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_error'])

#Entrenamiento del modelo, dividido en 150 epocas, con un batch de tamano 10
model.fit(X, y, epochs=150, batch_size=10)

#Se evalua el modelo utilizando mse 
score = model.evaluate(X, y)
print(f"Score: {score:.2f}")

"""# New Section"""