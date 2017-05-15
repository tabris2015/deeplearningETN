#This code is beerware; if you use it, please buy me 
#a cold beverage next time you run into one of
#us at the local.
#abril de 2017 - Jose Laruta
#Code developed in python 2.7
#****************************************************************/

#modulos necesarios
import os

####keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

from keras.models import model_from_json
#############
import numpy as np 						
from matplotlib import pyplot as plt
import time
import cv2
import random

import affinity, multiprocessing 	# para todos los nucleos (np)

affinity.set_process_affinity_mask(0,2**multiprocessing.cpu_count()-1)


print "importando dataset..."

examples = []

# CARGAR EL DATASET EN LA MEMORIA
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
# PREPROCESAR EL DATASET PARA ENTRENAMIENTO Y TEST
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# extraer 12 imagenes aleatorias del dataset
for i in range(100):
	examples.append(random.choice(X_train).reshape((28,28)))

examples = np.array(examples)
print examples.shape

#imagen de prueba 10x10
ex_matrix = np.zeros((28,308))

for i in range(10):
	aux_row = np.zeros((28,28))
	for j in range(10):
		aux_row = np.concatenate((aux_row,examples[j+10*i]), axis=1)
	ex_matrix = np.concatenate((ex_matrix, aux_row), axis=0)

print "tamano matriz de prueba: " + str(ex_matrix.shape)

plt.imshow(ex_matrix)
plt.show()
print "dataset importado!"

#---------------------------
# ARQUITECTURA DEL MODELO
#---------------------------
model = Sequential()
 
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
# 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# ENTRENAMIENTO DEL MODELO
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=1, verbose=1)
 
# PRUEBA DEL MODELO
score = model.evaluate(X_test, Y_test, verbose=0)
print " precision: "  + str(score[1] * 100)

#guardamos el clasificador en un archivo
# serialize model to JSON
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")
