#This code is beerware; if you use it, please buy me 
#a cold beverage next time you run into one of
#us at the local.
#abril de 2017 - Jose Laruta
#Code developed in python 2.7
#****************************************************************/

#modulos necesarios
import os

####keras
import tensorflow as tf
from keras.utils import np_utils
from keras.datasets import mnist

from keras.models import model_from_json
#############

import numpy as np 	
from sklearn.model_selection import train_test_split					
from matplotlib import pyplot as plt
import time
import cv2
import random

import affinity, multiprocessing 	# para todos los nucleos (np)

affinity.set_process_affinity_mask(0,2**multiprocessing.cpu_count()-1)

#####

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#####


print "importando dataset..."

examples = []

# CARGAR EL DATASET EN LA MEMORIA
(X_train, y_train), (X_test_base, y_test_base) = mnist.load_data()
 

# Splitting the dataset into the Training set and Test set
X_test, X_valid, y_test, y_valid = train_test_split(X_test_base, y_test_base, test_size = 0.30, random_state = 0)

FLATTENED_SIZE = X_train.shape[1] * X_train.shape[2]
NUM_LABELS = 10
# PREPROCESAR EL DATASET PARA ENTRENAMIENTO Y TEST
X_train = X_train.reshape(X_train.shape[0], FLATTENED_SIZE)
X_test = X_test.reshape(X_test.shape[0], FLATTENED_SIZE)
X_valid = X_valid.reshape(X_valid.shape[0], FLATTENED_SIZE)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_valid = X_valid.astype('float32')
X_train /= 255
X_test /= 255
X_valid /= 255

Y_train = np_utils.to_categorical(y_train, NUM_LABELS)
Y_test = np_utils.to_categorical(y_test, NUM_LABELS)
Y_valid = np_utils.to_categorical(y_valid, NUM_LABELS)
print ('train:', X_train.shape, ' test: ', X_test.shape, ' valid: ', X_valid.shape)

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

#plt.imshow(ex_matrix)
#plt.show()
print "dataset importado!"

#---------------------------
# ARQUITECTURA DEL MODELO
#---------------------------

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
batch_size = 128

HIDDEN_SIZE = 32
graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_input_data = tf.placeholder(tf.float32, shape=(None, FLATTENED_SIZE), name="input_placeholder")
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))
    tf_valid_dataset = tf.constant(X_valid)
    tf_test_dataset = tf.constant(X_test)
    
    # Variables.
    weights1 = tf.Variable(tf.truncated_normal([FLATTENED_SIZE, HIDDEN_SIZE]), name='W1')
    biases1 = tf.Variable(tf.zeros([HIDDEN_SIZE]),name='b1')
    tf.add_to_collection('vars', weights1)
    tf.add_to_collection('vars', biases1)

    weights2 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, HIDDEN_SIZE]), name='W2')
    biases2 = tf.Variable(tf.zeros([HIDDEN_SIZE]),name='b2')
    tf.add_to_collection('vars', weights2)
    tf.add_to_collection('vars', biases2)
    
    weights3 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, NUM_LABELS]), name='W3')
    biases3 = tf.Variable(tf.zeros([NUM_LABELS]),name='b3')
    tf.add_to_collection('vars', weights3)
    tf.add_to_collection('vars', biases3)
    

    # Training computation.
    logits1 = tf.matmul(tf_input_data, weights1) + biases1
    layer1 = tf.nn.relu(logits1)
    
    logits2 = tf.matmul(layer1, weights2) + biases2
    layer2 = tf.nn.relu(logits2)
    
    logits3 = tf.matmul(layer2, weights3) + biases3
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits3))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits3, name='prediction')

    #valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    #test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    # SAVER
    saver = tf.train.Saver()
    #----
num_steps = 15001



with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (Y_train.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = X_train[offset:(offset + batch_size), :]
        batch_labels = Y_train[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_input_data
 : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            val_predictions = session.run(train_prediction, feed_dict={tf_input_data
    :X_valid})
            print("Validation accuracy: %.1f%%" % accuracy(val_predictions, Y_valid))
    
    #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), Y_test))
    print "guardando modelo..."
    saver.save(session, 'fully2')
    print "guardado!"

#------------------------------
 