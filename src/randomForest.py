'''
Dharma Hoy
05/01/22
Physics 305 Creative Project
Random Forest

This program creates a random forest 
model for the galaxy data and looks
at feature importance.
'''
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

# load data
stars = pd.read_csv("data/normalized_star_classification.csv")

# define independent variables x and response variable y
X = stars[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', \
'redshift', 'MJD']]
y = stars['class']

# split into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# set up your model
randomForest = RandomForestClassifier(n_estimators = 100)
randomForest.fit(X_train, y_train)

# print parameters
params = randomForest.get_params()
print("hyperparameters", params)

# make predictions
y_pred = randomForest.predict(X_test)
 
# find the accuracy 
accuracy = accuracy_score(y_test, y_pred)
print ("accuracy:", accuracy)

# find the feature importance
feature_importances = pd.DataFrame(randomForest.feature_importances_,index = X_train.columns, \
columns=['importance']).sort_values('importance',ascending=False)

fig, ax = plt.subplots()
feature_importances.plot(kind = 'barh', color = 'paleturquoise')
plt.title("Feature Importance") 
plt.xlabel("features")
plt.ylabel("Importance")
plt.savefig("output/featureImportance.pdf")
plt.show()
plt.close()

'''
The output is shown below
hyperparameters {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 
'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 
'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0,
'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0,
'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 
'verbose': 0, 'warm_start': False}

accuracy: 0.9777

The feature importance graph is in the output folder.
'''

