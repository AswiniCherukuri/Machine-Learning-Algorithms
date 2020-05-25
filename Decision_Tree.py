import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/Aswini Cherukuri/Desktop/Data Science Assignments/Python Codes/Decision Tree/iris.csv")
data.head()
data['Species'].unique()
data.Species.value_counts()
colnames = list(data.columns)
colnames
predictors = colnames[:4]
predictors
target = colnames[4]
target

# Splitting data into training and testing data set

import numpy as np


from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)
from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])
preds
pd.Series(preds).value_counts()

pd.crosstab(test[target],preds)


# Accuracy 
np.mean(preds==test.Species) # 1


