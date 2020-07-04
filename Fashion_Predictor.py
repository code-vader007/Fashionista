
#Import the libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#Import the fashion dataset from keras
fashion_mnist = keras.datasets.fashion_mnist

#Load the data into train and test sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Plot the figure
plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()

#Scale the train and test data
train_images = train_images / 255.0
test_images = test_images / 255.0

#Craete a keras model with softmax and relu functions
model = keras.Sequential([ keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])

#Compile the model
model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Train the model
model.fit(train_images, train_labels, epochs=5)

#Testing the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)

predictions[0]

numpy.argmax(predictions[0])