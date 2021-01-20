import random
import time
import itertools
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.svm import SVR
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from deap import base
from deap import creator
from deap import tools
from itertools import chain
from lightgbm import LGBMRegressor

if not tf.test.gpu_device_name():
    print("asd")
else:
    print(tf.test.gpu_device_name())
    
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

#os.chdir('C:\\Users\\speed\\OneDrive\\Escritorio\\TESIS\\Seminario 1\\Testing')
os.chdir('C:\\Users\\speed\\OneDrive\\Escritorio\\TESIS\\Seminario 2\\Implementacion')
os.getcwd()

np.set_printoptions(precision=3, suppress=True)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

#Create individual
class Agente(object):
    def __init__(self, num_input, max_num_hidden):
        self.num_input = num_input
        self.max_num_hidden = max_num_hidden

    def gen_hidden(self):
        return random.randint(1, self.max_num_hidden)

    def gen_weights(self, num_hidden, type_):
        if type_ == "input":
            num_weights = self.num_input * num_hidden
        elif type_ == "output":
            num_weights = num_hidden
        result = [random.random() for i in range(num_weights)]
        #result = [random.randint(0,1) for i in range(num_weights)]
        return result

    def get_random_agent(self):
        num_hidden = self.gen_hidden()
        hidden_weights = self.gen_weights(num_hidden, type_="input")
        output_weights = self.gen_weights(num_hidden, type_="output")
        #print([[num_hidden], hidden_weights, output_weights])
        return [num_hidden] + hidden_weights + output_weights
#num_input, max_num_hidden
def newIndividual(num_input, max_num_hidden): 
    agent = Agente(num_input, max_num_hidden).get_random_agent()
    #print(agent)
    agent = creator.Individual(agent)
    return agent
print(newIndividual(16,20))
#with tf.device('/GPU:0'):
def agent_SVM(): 
    ind = []
    gamma = random.uniform(0.0001, 0.001)
    epsilon = random.uniform(0.0001, 0.001)
    C = random.uniform(0, 1)
    ind.append(gamma)
    ind.append(epsilon)
    ind.append(C)
    ind = creator.Individual(ind)
    return ind
  
print(agent_SVM())

def agent_GBT():
    ind = []
    learningRate = round(random.uniform(0.01, 1), 2)
    nEstimators = random.randrange(10, 1500, step = 25)
    maxDepth = int(random.randrange(1, 10, step= 1))
    minChildWeight = round(random.uniform(0.01, 10.0), 2)
    subSample = round(random.uniform(0.01, 1.0), 2)
    colSampleByTree = round(random.uniform(0.01, 1.0), 2)
    gammaValue = round(random.uniform(0.01, 10.0), 2)
    
    ind.append(learningRate)
    ind.append(nEstimators)
    ind.append(maxDepth)
    ind.append(minChildWeight)
    ind.append(subSample)
    ind.append(colSampleByTree)
    ind.append(gammaValue)
    ind = creator.Individual(ind)
    return ind

print(agent_GBT())

toolbox = base.Toolbox()
#Asignar class type deap.creator.Individual a nuestro individuo
#num_input, max_num_hidden
toolbox.register("individual", newIndividual, 17, 20)
#toolbox.register("individual", agent_SVM)
#toolbox.register("individual", agent_GBT)
print(type(toolbox.individual()))
toolbox.individual()

#Crear population de tipo list con invdividuals de tipo deap.creator.Individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

population = toolbox.population(n=20)

population

def neuralNetwork():
  dataset = pd.read_csv("DatasetFinal.csv")

  data = dataset.copy()

  X_full = data.drop(columns="Quota")
  y_full = data["Quota"]

  X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
  X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
  
  metrics = [tf.keras.metrics.MSE]
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

  return metrics, optimizer, X_train, X_valid, X_test, y_train, y_valid, y_test

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

metrics, optimizer, X_train, X_valid, X_test, y_train, y_valid, y_test = neuralNetwork()
print(y_test)

"""# Support Vector Machine"""
def SVM(individual):
  gamma = individual[0]
  epsilon = individual[1]
  C = individual[2]
  print(gamma)
  print(epsilon)
  print(C)
  if (gamma <= 0 or epsilon <= 0 or C <= 0):
      gamma = 0.001
      epsilon = 0.001 
      C = 1 
  svm_reg = SVR(kernel='rbf', gamma=gamma, epsilon=epsilon, C=C) 
  svm_reg.fit(X_train, y_train)
  y_pred = svm_reg.predict(X_test)

  print("MSE: ", (mean_squared_error(y_test, y_pred)))
  print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
  print("MAE: ", mean_absolute_error(y_test, y_pred))
  print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

  return mean_absolute_percentage_error(y_test, y_pred),
#print(SVM([0.00019401519155119352, 0.00043450554083697895, 1.0]))

print("MSE: ", 2638.149823742)
print("RMSE: ", np.sqrt(2638.149823742))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

"""# Gradient Boosted Trees"""
def GBT(individual):
    learning_rate = individual[0]
    n_estimators = individual[1]
    max_depth = individual[2]
    min_child_weight = individual[3]
    subsample = individual[4]
    colsample_bytree = individual[5]
    gamma = individual[6]
    
    if (learning_rate <= 0 or subsample <= 0 or colsample_bytree <= 0 or n_estimators <= 0 or gamma <= 0 or min_child_weight <= 0 or max_depth <= 0):
        learning_rate = 0.01
        subsample = 0.01
        colsample_bytree = 0.01
        n_estimators = 10
        gamma = 0.01
        min_child_weight = 0.01
        max_depth = 1
        
    print(learning_rate)
    print(subsample)
    print(colsample_bytree) 
    
    lgbm_reg = LGBMRegressor(
        random_state=42,
        objective='mape',
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma
    )
    
    #Entrenamiento de GBT
    lgbm_reg.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        eval_metric='mape'
        #early_stopping_rounds=200
    )
    
    y_pred = lgbm_reg.predict(X_test)
    
    print("MSE: ", (mean_squared_error(y_test, y_pred)))
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))
    
    print(len(y_pred))
    print(y_test)

    return mean_absolute_percentage_error(y_test, y_pred), 

print(GBT([1.0, 0, 1, 0.0, 0.0, 0.0, 0.0]))

"""# Artificial Neural Network"""
def runNeuralNetwork(weight_set_1, weight_set_2, neurons, metrics, optimizer, X_train, X_valid, X_test, y_train, y_valid, y_test) : 
  #9 columnas de entrada, sin el target
  pesos_1 = np.array(weight_set_1)
  pesos_1 = [np.reshape(pesos_1, (16,neurons))] 

  #pesos_2 = np.array(weight_set_2)
  #pesos_2 = np.array(pesos_2, 'float32')
  #pesos_2 = [np.reshape(pesos_2, (neurons,1))]
  #print(pesos_2.shape)
  #pesos_2 = np.array([pesos_2, np.array([0], 'float32')])

  pesos_1 = tf.convert_to_tensor(pesos_1)
  model = tf.keras.models.Sequential(
    [
      tf.keras.layers.InputLayer(input_shape=X_train.iloc[0, :].shape),
      #neurons --> Neuronas capa invisible
      tf.keras.layers.Dense(neurons, kernel_initializer="glorot_uniform", use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("tanh"),
      tf.keras.layers.Dense(1)
     
    ]
  )
  print(type(model.layers[3].get_weights()))
  #Actualizar pesos capa 1 
  print("Pesos capa Input")
  print(model.layers[0].get_weights()[0])
  model.layers[0].set_weights(pesos_1)
  print("Nuevos pesos capa Input")
  print(model.layers[0].get_weights()[0])
  print("Pesos capa Output")
  print(model.layers[3].get_weights()[0])
  #Actualizar pesos capa 2
  #model.layers[3].set_weights(pesos_2)

  model.compile(
    optimizer=optimizer,
    loss=['mape'],
    metrics=metrics,
  )

  start_time = time.time()
  #Inicio de entrenamiento
  history = model.fit(
            x=X_train.values,
            y=y_train,
            epochs=100,
            validation_data=(X_valid.values, y_valid),
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)],
            batch_size=500
        )

  y_pred = model.predict(X_test.values)

  model.evaluate(X_test.values, y_test)

  #Calculo de MAPE
  def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

  print("MSE: ", mean_squared_error(y_test, y_pred))
  print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
  print("MAE: ", mean_absolute_error(y_test, y_pred))
  print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

  return mean_squared_error(y_test, y_pred)

n_inputs = 16
#individuo_test = [2, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
metrics, optimizer, X_train, X_valid, X_test, y_train, y_valid, y_test = neuralNetwork()

def evalNeuralNetwork(individual): 
  individuo_length = len(individual) - 1
  neurons = individual[0]
  set_1_length = neurons * n_inputs
  print(neurons)
  weight_set_1 = individual[1:set_1_length+1]
  weight_set_2 = individual[set_1_length+1::]
  costo = runNeuralNetwork(weight_set_1, weight_set_2, neurons, metrics, optimizer, X_train, X_valid, X_test, y_train, y_valid, y_test)
  print("Weight set 1: {}".format(weight_set_1))
  print("Weight set 2: {}".format(weight_set_2))
  return int(costo),
evalNeuralNetwork([1, 0.6012968962057692, 0.3288730558052064, 0.7268140298689721, 0.8327367794820845, 0.236145084963732, 0.38318481944450833, 0.43378997724989865, 0.5186766987110594, 0.5738685344541565, 0.6754617175509781, 0.8460970519086625, 0.4177079271220986, 0.6581872555038365, 0.365328228990503, 0.64619864090001, 0.17396688007706473, 0.8174058238215702])
#Operadores Geneticos
toolbox.register("evaluate", evalNeuralNetwork)
#toolbox.register("evaluate", SVM)
#toolbox.register("evaluate", GBT)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

#population = toolbox.population(n=4)
print(population)
fitnesses = list(map(toolbox.evaluate, population))
print(fitnesses)

for ind, fit in zip(population, fitnesses):
      print(ind)
      print(fit)
      ind.fitness.values = fit

# CXPB = % probabilidad de Crossover
# MUTPB = % probabilidad de Mutation
CXPB, MUTPB = 0.5, 0.2

print("Population")
print(population)
fits = [ind.fitness.values[0] for ind in population]
print(fits)

generation = 0 
maxGenerations = 20
individuoAceptable = 5

while max(fits) > individuoAceptable and generation < maxGenerations: 
  generation = generation + 1 
  print("-- Generation %i --" % generation)
  print("-- Generation %i --" % generation)
  print("-- Generation %i --" % generation)
  print("-- Generation %i --" % generation)
  print("-- Generation %i --" % generation)
  # Select the next generation individuals
  offspring = toolbox.select(population, len(population))
  # Clone the selected individuals
  offspring = list(map(toolbox.clone, offspring))
  for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CXPB:
      #Crossover
      toolbox.mate(child1, child2)
      del child1.fitness.values
      del child2.fitness.values

    for mutant in offspring:
      if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    print("invalid_ind: {}".format(invalid_ind))
    print("fitnesses: {}".format(fitnesses))
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit

    print("Evaluated %i individuals" % len(invalid_ind))
    #Reemplazar poblacion antigua por nueva con los nuevos fitness 
    #de los individuos
    population[:] = offspring
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in population]
        
    length = len(population)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
        
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

  print("-- End of (successful) evolution --")

print(population)

best_ind = tools.selBest(population, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
