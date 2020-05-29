# -*- coding: utf-8 -*-
"""Tesis

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zXQIEF2z9kYjHnrbhgaHdU8Su8qGSQq-
"""

import tensorflow as tf 
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Input
from numpy import loadtxt

##### Etapa 1 - Extracion y pre procesamiento de datos #####

#Extraccion de datos 
data = pd.read_csv("RetailDataIII.csv") 
#Seleccion de productos que se van a pronosticar para posteriormente generar una base con menos carga de datos 
#Se seleccionaron los productos con mayor cantidad de registros, para ver los registrios se ejecuto la siguiente sentencia 
data.Product.value_counts().head(10)
productos =  ["White hanging heart t-light holder", "Regency cakestand 3 tier", "Jumbo bag red retrospot"]

dataset = data[data.Product.isin(productos)]

dataset.to_csv("3productos.csv", index=False)

#Se elimina la base anterior para liberar memoria RAM
del data

#Se genera una variable con el valor de la nueva base 
data3prod = pd.read_csv("3productos.csv")

#Se realiza un brief todo los datos y se encuentran missing en CustomerID
#Pero esta columna no es critica para realizar el pronostico, por lo tanto no se busca una forma 
#De depurar esta variable de datos
data3prod.info()

#Estandarizar la columna BillDate, dividiendola en Year, Month, Day, DayOfWeek (Dia de semana), IsWeekend (Booleano SI, NO fin de semana)
#isHoliday (Booleano SI, NO es feriado)
data3prod.BillDate = pd.to_datetime(data3prod.BillDate)

cal = calendar() 
holidays = cal.holidays(start=data3prod.BillDate.min(), end=data3prod.BillDate.max())

data3prod["Year"] = data3prod.BillDate.dt.year
data3prod["Month"] = data3prod.BillDate.dt.month
data3prod["Day"] = data3prod.BillDate.dt.day
data3prod["dayOfWeek"] = data3prod.BillDate.dt.dayofweek
# 1 si es fin de semana (S-D)
data3prod.loc[data3prod.dayOfWeek >= 5, "isWeekend"] = 1
# 0 si es semana (L-V)
data3prod.loc[data3prod.dayOfWeek < 5, "isWeekend"] = 0
data3prod["isHoliday"] = data3prod.BillDate.isin(holidays)
data3prod["isHoliday"] = data3prod.isHoliday.map({False: 0, True: 1})

data3prod.head(5)

#Agrupar cada producto de ventas diarias a ventas semanales 
def get_products(df, products):
    result = []
    aggfuncs = {
        "Quota": 'sum',
        "Amount": 'mean',
        "Product": 'first',
        "isHoliday": "max",
        "nHolidayInWeek": "sum"
    }
    offset = pd.offsets.timedelta(days=-6)
    for prod in products:
        dataset = df[df.Product == prod]
        dataset = dataset.set_index("BillDate").resample("W" , loffset=offset).apply(aggfuncs)
        result.append(dataset)
    return result

dataprod_1, dataprod_2, dataprod_3 = get_products(data3prod, productos)

dataprod_1
#data3prod[data3prod.Quota<=0]

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
model.compile(loss='mean_square_error', optimizer=sgd, metrics=['mean_squared_error'])

#Entrenamiento del modelo, dividido en 150 epocas, con un batch de tamano 10
model.fit(X, y, epochs=150, batch_size=10)

#Se evalua el modelo utilizando mse 
score = model.evaluate(X, y)
print(f"Score: {score:.2f}")


#---------------------------#
#        Topologia          #
#         [3-3-2-1]         #
#---------------------------#

model = Sequential([
    Input(input_shape = X.shape)
    BatchNormalization(),
    Dense(3, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    Dense(3, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    Dense(2, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    #Capa de salida con 1 neurona 
    Dense(1)
])
optimizer = SGD()
model.compile(loss='mean_square_error', optimizer=sgd, metrics=['mean_squared_error'])
model.fit(X, y, epochs=150, batch_size=10)

score = model.evaluate(X, y)
print(f"Score: {score:.2f}")


#---------------------------#
#         Topologia         #
#        [3-4-5-5-1]        #
#---------------------------#

model = Sequential([
    Input(input_shape = X.shape)
    BatchNormalization(),
    Dense(3, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    Dense(4, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    Dense(5, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    Dense(5, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    #Capa de salida con 1 neurona 
    Dense(1)
])
optimizer = SGD()
model.compile(loss='mean_square_error', optimizer=sgd, metrics=['mean_squared_error'])
model.fit(X, y, epochs=150, batch_size=10)

score = model.evaluate(X, y)
print(f"Score: {score:.2f}")

#---------------------------#
#         Topologia         #
#        [3-6-5-6-6-5]      #
#---------------------------#

model = Sequential([
    Input(input_shape = X.shape)
    BatchNormalization(),
    Dense(6, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    Dense(5, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    Dense(6, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    Dense(6, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    Dense(5, kernel_initializer="glorot_normal", use_bias=False),
    BatchNormalization(),
    Activation("tanh"),
    #Capa de salida con 1 neurona 
    Dense(1)
])
optimizer = SGD()
model.compile(loss='mean_square_error', optimizer=sgd, metrics=['mean_squared_error'])
model.fit(X, y, epochs=150, batch_size=10)

score = model.evaluate(X, y)
print(f"Score: {score:.2f}")

"""# New Section"""