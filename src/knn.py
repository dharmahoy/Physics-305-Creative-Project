'''
Dharma Hoy
04/24/22
Physics 305 Creative Project
K Nearest Neighbors

This program creates a K nearest neighbors 
model for the galaxy data 
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load data
stars = pd.read_csv("data/normalized_star_classification.csv")

# define independent variables x and response variable y
X = stars[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', \
'redshift', 'MJD']]
y = stars['class']

# split into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# create the model
knnModel = KNeighborsClassifier(n_neighbors=5)
knnModel.fit(X_train,y_train)

# view the coefficent of determination of the model
score = knnModel.score(X_train, y_train)
print("score:", score)

# show the parameters of the model
print("parameters:", knnModel.get_params())

# test the model on the test data
for i in range(5):
  y_pred = knnModel.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Trial", i, "accuracy:", accuracy)

'''
output is shown below
score: 0.948725

parameters: {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 
'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}

Trial 0 accuracy: 0.93215
Trial 1 accuracy: 0.93215
Trial 2 accuracy: 0.93215
Trial 3 accuracy: 0.93215
Trial 4 accuracy: 0.93215
'''


# now create a second KNN model using 10 neighbors
# create the model
knnModel = KNeighborsClassifier(n_neighbors=10)
knnModel.fit(X_train,y_train)

# view the coefficent of determination of the model
score = knnModel.score(X_train, y_train)
print("score:", score)

# show the parameters of the model
print("parameters:", knnModel.get_params())

# test the model on the test data
for i in range(5):
  y_pred = knnModel.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Trial", i, "accuracy:", accuracy)


'''
output is shown below
score: 0.933775

parameters: {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 
'metric_params': None, 'n_jobs': None, 'n_neighbors': 10, 'p': 2, 'weights': 'uniform'}

Trial 0 accuracy: 0.9233
Trial 1 accuracy: 0.9233
Trial 2 accuracy: 0.9233
Trial 3 accuracy: 0.9233
Trial 4 accuracy: 0.9233
'''

# the accuracy decreses when the trials were doubled
