#Neural Network
import pandas as pd
import numpy as np

# Reading data 
Concrete = pd.read_csv("C:/Users/Aswini Cherukuri/Desktop/Data Science Assignments/Python Codes/Neural Networks/concrete.csv")
Concrete.head()

colnames = list(Concrete.columns)
colnames
predictors = colnames[:8]
target = colnames[8]
Y = np.asarray(Concrete[target], dtype="|S6")
X = np.asarray(Concrete[predictors])

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, ), random_state=1)
clf.fit(X,Y)

#### Predicted scores
clf.score(X,Y)
# 10 hidden layers suits best for predicting the strength of the 
# concrete
# as the hidden layers increases the accuracy of the model decreases

