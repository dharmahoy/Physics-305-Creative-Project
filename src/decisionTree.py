'''
Dharma Hoy
05/01/22
Physics 305 Creative Project
Decision Tree

This program creates a Decision tree 
model for the galaxy data
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz

# load data
stars = pd.read_csv("data/normalized_star_classification.csv")

# define independent variables x and response variable y
X = stars[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', \
'redshift', 'MJD']]
y = stars['class']

# split into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# create the model
decisionTree = DecisionTreeClassifier(max_depth = 3)
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
export_graphviz(decisionTree, out_file =  "output/decisionTree.dot",feature_names = list(X.columns), \
class_names = ['Galaxy', 'Star', 'Quasar'], filled = True, rounded = True)

# the dot file created was plugged into https://dreampuf.github.io/GraphvizOnline
# and the resulting jpeg was added to the output folder

'''
output is shown below
score: 0.9477

parameters: {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 
'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 
'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 
'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'}

Trial 0 accuracy: 0.9465
Trial 1 accuracy: 0.9465
Trial 2 accuracy: 0.9465
Trial 3 accuracy: 0.9465
Trial 4 accuracy: 0.9465

The dot file and jpeg of the decision tree are in the output folder.
'''

