'''
Dharma Hoy
05/01/22
Physics 305 Creative Project
Neural Network

This program creates a neural network 
model for the galaxy data 
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras

# print out the different of packages used versions
print ("system:", sys.version)
print ("pandas:", pd.__version__)
print ("tf:", tf.__version__)

# load data
stars = pd.read_csv("data/normalized_star_classification.csv")

# define independent variables x and response variable y
X = stars[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', \
'redshift', 'MJD']]
y = stars['class']

# split into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# create the model
neuralNet = Sequential()
neuralNet.add(Dense(7, input_shape = (9, ), activation = 'relu'))
neuralNet.add(Dense(7, activation = 'relu'))
neuralNet.add(Dense(4, activation = 'relu'))
neuralNet.add(Dense(1, activation = 'relu'))

# compile model and print hyperparameters
neuralNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("hyperparameters:", neuralNet.optimizer.get_config())

# print model summary
print("summary:", neuralNet.summary())

# train the model and view the output
history = neuralNet.fit(X_train, y_train,  batch_size=5, epochs=10, verbose=0, validation_data=(X_test, y_test))
results = neuralNet.evaluate(X_test, y_test)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

'''
output shown below
system: 3.8.12 (default, Oct 12 2021, 03:01:40) [MSC v.1916 64 bit (AMD64)]
pandas: 1.4.2
tf: 2.3.0

hyperparameters: {'name': 'Adam', 'learning_rate': 0.001, 'decay': 0.0, 
'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 7)                 70        
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 56        
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 32        
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5         
=================================================================
Total params: 163
Trainable params: 163
Non-trainable params: 0
_________________________________________________________________
summary: None

Final test set loss:  nan
Final test set accuracy: 0.592550
'''

# add another layer to the neural net and see if it works better
# create model
neuralNet = Sequential()
neuralNet.add(Dense(7, input_shape = (9, ), activation = 'relu'))
neuralNet.add(Dense(7, activation = 'relu'))
neuralNet.add(Dense(4, activation = 'relu'))
neuralNet.add(Dense(2, activation = 'relu'))
neuralNet.add(Dense(1, activation = 'relu'))

# compile model and print hyperparameters
neuralNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("hyperparameters:", neuralNet.optimizer.get_config())

# print model summary
print("summary:", neuralNet.summary())

# train the model and view the output
history = neuralNet.fit(X_train, y_train,  batch_size=5, epochs=10, verbose=0, validation_data=(X_test, y_test))
results = neuralNet.evaluate(X_test, y_test)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

'''
output is shown below
hyperparameters: {'name': 'Adam', 'learning_rate': 0.001, 'decay': 0.0, 
'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 7)                 70        
_________________________________________________________________
dense_5 (Dense)              (None, 7)                 56        
_________________________________________________________________
dense_6 (Dense)              (None, 4)                 32        
_________________________________________________________________
dense_7 (Dense)              (None, 2)                 10        
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 3         
=================================================================
Total params: 171
Trainable params: 171
Non-trainable params: 0
_________________________________________________________________
summary: None

Final test set loss: 0.000000
Final test set accuracy: 0.592550
'''
# there was not a significant difference in the models after adding the layer
