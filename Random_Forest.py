import pandas as pd
import numpy as np


# Reading the Diabetes Data #################
Diabetes = pd.read_csv("C:/Users/Aswini Cherukuri/Desktop/Data Science Assignments/Python Codes/Random Forests/Diabetes_RF.csv")
Diabetes.head()
Diabetes.columns
colnames = list(Diabetes.columns)

predictors = colnames[:8]
predictors
target = colnames[8]
target
X = Diabetes[predictors]
Y = Diabetes[target]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(Diabetes) # 768,9 => Shape 

#### Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.
rf.n_outputs_ # Number of outputs when fit performed
rf.oob_score_  # 0.72916
rf.predict(X)
##############################

Diabetes['rf_pred'] = rf.predict(X)
cols = ['rf_pred',' Class variable']
Diabetes[cols].head()
Diabetes[" Class variable"]


from sklearn.metrics import confusion_matrix
confusion_matrix(Diabetes[' Class variable'],Diabetes['rf_pred']) # Confusion matrix

print("Accuracy",(497+268)/(497+268+0+3)*100)
# Accuracy is 99.609375
Diabetes["rf_pred"]

