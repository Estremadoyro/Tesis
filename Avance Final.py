# -*- coding: utf-8 -*-
# Librerias
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import tensorflow as tf
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.chdir('C:\\Users\\speed\\OneDrive\\Escritorio\\TESIS\\Seminario 1\\Testing')
os.getcwd()
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
#Parametros de ploteos
plt.style.use('seaborn')
mpl.rcParams.update({'axes.titlesize': 24,
                     'axes.labelsize': 20,
                     'lines.linewidth': 2,
                     'lines.markersize': 10,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'figure.figsize': (12, 8),
                     'legend.fontsize': 13,
                     'legend.handlelength': 2})

#Determinar la GPU como hardware de computacion 
tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

#Data set https://www.kaggle.com/coldperformer/online-retail-data-v3?select=RetailDataIII.csv
data = pd.read_csv("RetailDataIII.csv")

data.head()

data.Product.value_counts().head(10)

productos = ['White hanging heart t-light holder', 'Regency cakestand 3 tier', 'Jumbo bag red retrospot']
#, 'Regency cakestand 3 tier', 'Jumbo bag red retrospot'
dataset = data[data.Product.isin(productos)]

# del data
#Extracion de datos
dataset.to_csv("final_data.csv", index=False)

dataset.head()

dataset.BillDate = pd.to_datetime(dataset.BillDate)

cal = calendar()
holidays = cal.holidays(start=dataset.BillDate.min(), end=dataset.BillDate.max())
#Creacion de variables adicionales 
dataset["Month"] = dataset.BillDate.dt.month
dataset["Day"] = dataset.BillDate.dt.day
dataset["nameofday"] = dataset.BillDate.dt.day_name()
dataset["dayofweek"] = dataset.BillDate.dt.dayofweek
dataset.loc[dataset.dayofweek >= 5, "isWeekend"] = 1
dataset.loc[dataset.dayofweek < 5, "isWeekend"] = 0
dataset["isHoliday"] = dataset.BillDate.isin(holidays)
dataset.isHoliday = dataset.isHoliday.map({True: 1, False: 0})
dataset["Year"] = dataset.BillDate.dt.year
dataset["WeekOfYear"] = dataset.BillDate.dt.weekofyear

dataset.head()

"""## Con los 13K datos"""

# Label product names
dataset.Product, values = pd.factorize(dataset.Product)

dataset.head()

# Quitar Quota negativos y outliers
LOWER = dataset.Quota.quantile(q=0.05)
UPPER = dataset.Quota.quantile(q=0.99)

dataset = dataset[(dataset.Quota >= 0) & (dataset.Quota <= UPPER)]

# product_cols = pd.get_dummies(dataset.Product, prefix="Product")

# dataset = pd.concat([dataset, product_cols], axis=1)
# dataset.drop(columns="Product", inplace=True)

dataset.reset_index(drop=True ,inplace=True)
data = pd.DataFrame()
print(dataset.count())
dataset = dataset.loc[:len(dataset)-5, :]
#5
data
#Obteniendo los valores de venta de dias consiguientes
data['D1'] = dataset.loc[::5, "Quota"]
data['D2'] = dataset.loc[1::5, "Quota"].values
data['D3'] = dataset.loc[2::5, "Quota"].values
data['D4'] = dataset.loc[3::5, "Quota"].values
data['D5'] = dataset.loc[4::5, "Quota"].values
data
X_full = data.drop(columns="D5")
y_full = data["D5"]

#X_full = dataset.drop(labels=['Bill', 'MerchandiseID', 'BillDate', 'CustomerID', 'Country', 'Product','Quota','nameofday'], axis=1)
#y_full = dataset['Quota']

dataset.columns

X_full.columns

#Especificacion de datos de entrenamiento y validacion 
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train.shape

X_train.head()

#Estructuracion de la arquitectura de red neuronal, en este caso RN4
#glorot_uniform = Xavier Initialization 

# -------- RN 1 -------- #
# model = tf.keras.models.Sequential(
#     [
#       tf.keras.layers.InputLayer(input_shape=X_train.iloc[0, :].shape),
#       tf.keras.layers.Dense(2, kernel_initializer="glorot_uniform", use_bias=False),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Activation("tanh"),
#       tf.keras.layers.Dense(1)
     
#     ]
# )

# -------- RN 2 -------- #
# model = tf.keras.models.Sequential(
#     [
#       tf.keras.layers.InputLayer(input_shape=X_train.iloc[0, :].shape),
#       tf.keras.layers.Dense(3, kernel_initializer="glorot_uniform", use_bias=False),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Activation("tanh"),
#       tf.keras.layers.Dense(2, kernel_initializer="glorot_uniform", use_bias=False),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Activation("tanh"),
#       tf.keras.layers.Dense(1)
     
#      ]
# )

# -------- RN 3 -------- #
model = tf.keras.models.Sequential(
    [
      tf.keras.layers.InputLayer(input_shape=X_train.iloc[0, :].shape),
      tf.keras.layers.Dense(4, kernel_initializer="glorot_uniform", use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("tanh"),
      tf.keras.layers.Dense(5, kernel_initializer="glorot_uniform", use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("tanh"),
      tf.keras.layers.Dense(5, kernel_initializer="glorot_uniform", use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("tanh"),
      tf.keras.layers.Dense(1)
     
    ]
)

# # -------- RN 4 -------- #
# model = tf.keras.models.Sequential(
#     [
#       tf.keras.layers.InputLayer(input_shape=X_train.iloc[0, :].shape),
#       tf.keras.layers.Dense(6, kernel_initializer="glorot_uniform", use_bias=False),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Activation("tanh"),
#       tf.keras.layers.Dense(5, kernel_initializer="glorot_uniform", use_bias=False),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Activation("tanh"),
#       tf.keras.layers.Dense(6, kernel_initializer="glorot_uniform", use_bias=False),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Activation("tanh"),
#       tf.keras.layers.Dense(5, kernel_initializer="glorot_uniform", use_bias=False),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Activation("tanh"),
#       tf.keras.layers.Dense(1)
     
#     ]
# )

# Metricas
metrics = [
           tf.keras.metrics.MAE,
           tf.keras.metrics.MAPE,
           tf.keras.metrics.MSE
]

# Optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005)
# Model compile
model.compile(
    optimizer=optimizer,
    loss=['mape','mse'],
    metrics=metrics,
)

"""# RNN"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
#Medicion de tiempo de ejecucion
start_time = time.time()
#Inicio de entrenamiento
history = model.fit(
            x=X_train.values,
            y=y_train,
            epochs=200,
            validation_data=(X_valid.values, y_valid),
            #callbacks=[tf.keras.callbacks.EarlyStopping(patience=50)],
            batch_size=500
        )
print("Time: ", time.time() - start_time)

# %load_ext tensorboard
# %tensorboard --logdir=./my_logs --port=6006

y_pred = model.predict(X_test.values)

model.evaluate(X_test.values, y_test)

#Calculo de MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

#plt.figure(figsize=(32, 10))
#sns.scatterplot(y=y_pred.flatten(), x=range(len(y_pred.flatten())), alpha=0.5, label='Prediction')
#sns.scatterplot(y=y_test, x=range(len(y_test)), alpha=0.5, label='Test Values')

plt.plot(history.history['mean_squared_error'], label='Entrenamiento')
plt.plot(history.history['val_mean_squared_error'], label='Validacion')
plt.legend()
plt.title('Aprendizaje de la Red')
plt.ylabel('Ventas')
plt.xlabel('# Dia')

"""## SVM"""

from sklearn.svm import SVR

svm_reg = SVR()

start_time = time.time()
svm_reg.fit(X_train, y_train)
print("Time:", time.time() - start_time)
y_pred = svm_reg.predict(X_test)

time.time()

print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

"""# Gradient Boosted Trees"""

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

lgbm_reg = LGBMRegressor(
    n_estimators=1000,
    random_state=42,
    objective='mape',
    num_iterations=5000,
)

start_time = time.time()
#Entrenamiento de GBT
lgbm_reg.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    eval_metric='mape',
    early_stopping_rounds=200
)
print("Time: ", time.time() - start_time)

y_pred = lgbm_reg.predict(X_test)

print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))