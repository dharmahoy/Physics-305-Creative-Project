'''
Dharma Hoy
05/01/22
Physics 305 Creative Project
Boosted Decision tree

This program creates a boosted decision tree
model for the galaxy data 
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

# load data
stars = pd.read_csv("data/normalized_star_classification.csv")

# define independent variables x and response variable y
X = stars[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', \
'redshift', 'MJD']]
y = stars['class']

# split into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, \
random_state=0)

# create the model
decisionTree = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,\
max_depth=3, random_state=0)
decisionTree.fit(X_train, y_train)

# view the mean accuracy on given test data and labels
score = decisionTree.score(X_train, y_train)
print("score:", score)

# show the parameters of the model
print("parameters:", decisionTree.get_params())

# test the model on the test data
for i in range(5):
  y_pred = decisionTree.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Trial", i, "accuracy:", accuracy)

# plot the tree
# defining feature and class 
fn = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', \
'redshift', 'MJD']
cn = ['GALAXY', 'QSR', 'STAR']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(decisionTree.estimators_[10,0],feature_names = fn, \
class_names=cn,filled = True, proportion = True)
fig.savefig('output/boostedDecisonTree10.png')

'''
output shown below
score: 0.98175

parameters: {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 
'learning_rate': 1.0, 'loss': 'deviance', 'max_depth': 3, 'max_features': None, 
'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 
'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 
'n_iter_no_change': None, 'random_state': 0, 'subsample': 1.0, 'tol': 0.0001, 
'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

Trial 0 accuracy: 0.971
Trial 1 accuracy: 0.971
Trial 2 accuracy: 0.971
Trial 3 accuracy: 0.971
Trial 4 accuracy: 0.971

The tenth decision tree is shown in the output folder
'''
