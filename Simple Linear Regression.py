
# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# reading a csv file using pandas library
calories=pd.read_csv("C:/Users/Aswini Cherukuri/Downloads/calories_consumed.csv")
calories.columns
plt.hist(calories.wg)
plt.boxplot(calories.wg,0,"rs",0)

plt.hist(calories.cc)
plt.boxplot(calories.cc)

plt.plot(calories.wg,calories.cc,"bo");plt.xlabel("WEIGHT");plt.ylabel("CALORIES_CONSUMED")

calories.cc.corr(calories.wg) # # correlation value between X and Y


# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("cc~wg",data=calories).fit()

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary()

print (model.conf_int(0.05)) # 95% confidence interval

pred = model.predict(calories.iloc[:,0]) # Predicted values of AT using the model

# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=calories['wg'],y=calories['cc'],color='red');plt.plot(calories['wg'],pred,color='black');plt.xlabel('WG');plt.ylabel('CC')

pred.corr(calories.cc) # 0.81

# Transforming variables for accuracy
model2 = smf.ols('cc~np.log(wg)',data=calories).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(pd.DataFrame(calories['wg']))
pred2.corr(calories.cc)
# pred2 = model2.predict(wcat.iloc[:,0])
pred2
plt.scatter(x=calories['wg'],y=calories['cc'],color='red');plt.plot(calories['wg'],pred2,color='black');plt.xlabel('WG');plt.ylabel('CC')


# Exponential transformation
model3 = smf.ols('np.log(cc)~wg',data=calories).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(calories['wg']))
pred_log
pred3=np.exp(pred_log)  # as we have used log(AT) in preparing model so we need to convert it back
pred3
pred3.corr(calories.cc)
plt.scatter(x=calories['wg'],y=calories['cc'],color='red');plt.plot(calories['wg'],pred3,color='black');plt.xlabel('WG');plt.ylabel('CC')



# Quadratic model
calories["wg_sq"] = calories.wg*calories.wg
model_quad = smf.ols("cc~ wg+wg*wg",data=calories).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(pd.DataFrame(calories['wg']))

model_quad.conf_int(0.05) # 
plt.scatter(x=calories['wg'],y=calories['cc'],color='red');plt.plot(calories['wg'],pred_quad,color='black');plt.xlabel('WG');plt.ylabel('CC')
plt.scatter(calories.wg,calories.cc,c="b");plt.plot(calories.wg,pred_quad,"r")


plt.hist(model_quad.resid_pearson) # histogram plt.scatter(np.arange(109),model_quad.resid_pearson);plt.axhline(y=0 residual values 


# Choose highest R^2 value model then find residuals
student_resid = model_quad.resid_pearson 
student_resid
plt.plot(model_quad.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred_quad,y=calories.cc);plt.xlabel("Predicted");plt.ylabel("Actual")

Residuals=calories.cc-pred_quad
import statistics
x=statistics.mean(Residuals)    
print(x)
