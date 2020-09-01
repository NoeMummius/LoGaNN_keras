# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:27:37 2020

@author: Noé Muñoz Pérez
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
 
# cargamos las 4 combinaciones de las compuertas lógicas
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# y estos son los resultados que se obtienen de la compuerta XOR, en el mismo orden
XOR_target = np.array([[0],[1],[1],[0]], "float32")
# y estos son los resultados que se obtienen de la compuerta OR, en el mismo orden
OR_target = np.array([[0],[1],[1],[1]], "float32")
# y estos son los resultados que se obtienen de la compuerta NOR, en el mismo orden
NOR_target = np.array([[1],[0],[0],[0]], "float32")
# y estos son los resultados que se obtienen de la compuerta AND, en el mismo orden
AND_target = np.array([[0],[0],[0],[1]], "float32")
# y estos son los resultados que se obtienen de la compuerta NAND, en el mismo orden
NAND_target = np.array([[1],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

while(1):
 print("Logic gates neural networks\n1. XOR\n2. OR\n3. NOR\n4. AND\n5. NAND\nE. Exit\nWhich model do you choose? ")
 answer = input()
 if(answer == 'E'):
  break
 elif(int(answer) == 1):
 
  model.fit(training_data, XOR_target, epochs=200)
 
  # evaluamos el modelo
  scores = model.evaluate(training_data, XOR_target)
 
  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print (model.predict(training_data).round())
 elif(int(answer) == 2):

  model.fit(training_data, OR_target, epochs=200)
 
  # evaluamos el modelo
  scores = model.evaluate(training_data, OR_target)
 
  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print (model.predict(training_data).round())
 elif(int(answer) == 3):

  model.fit(training_data, NOR_target, epochs=200)
 
  # evaluamos el modelo
  scores = model.evaluate(training_data, NOR_target)
 
  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print (model.predict(training_data).round())
 elif(int(answer) == 4):

  model.fit(training_data, AND_target, epochs=200)
 
  # evaluamos el modelo
  scores = model.evaluate(training_data, AND_target)
 
  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print (model.predict(training_data).round())
 elif(int(answer) == 5):

  model.fit(training_data, NAND_target, epochs=200)
 
  # evaluamos el modelo
  scores = model.evaluate(training_data, NAND_target)
 
  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print (model.predict(training_data).round())
 else:
  print("Invalid answer")
