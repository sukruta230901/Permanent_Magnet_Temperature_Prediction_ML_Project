#!/usr/bin/env python
# coding: utf-8

# # Rotor Temperature Prediction
# 

# The goal of this project is to predict the Permanent Magnet Temperature of Electric Motor using various Machine Learning Regression Algorithms.

# ## Importing Libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## Importing Dataset 

# In[2]:


df = pd.read_csv("electric_motor_temperature.csv") 
df.head()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.info()


# ## Descriptive Data Analysis 

# In[6]:


df.describe()


# In[7]:


# There are no missing values in the dataset.
df.isnull().sum()


# In[8]:


df_test = df[(df['profile_id'] == 65) | (df['profile_id'] == 72)]
df = df[(df['profile_id'] != 65) & (df['profile_id'] != 72)]


# In[9]:


plt.figure(figsize=(15,6))
df['profile_id'].value_counts().sort_values().plot(kind = 'bar')


# As we can see, session ids 66, 6 and 20 have the most number of measurements recorded.

# In[10]:


plt.figure(figsize=(20,5))
df[df['profile_id'] == 20]['stator_yoke'].plot(label = 'stator yoke')
df[df['profile_id'] == 20]['stator_tooth'].plot(label = 'stator tooth')
df[df['profile_id'] == 20]['stator_winding'].plot(label = 'stator winding')
plt.legend()


# As we can see from the plot, all three stator components follow a similar measurment variance.
# 
# As the dataset author mentioned, the records in the same profile id have been sorted by time, we can assume that these recordings have been arranged in series of time.
# 
# Due to this we can infer that there has not been much time given for the motor to cool down in between recording the sensor data as we can see that initially the stator yoke temperature is low as compared to temperature of stator winding but as we progress in time, the stator yoke temperature goes above the temperature of stator winding.

# In[11]:


# As profile_id is an id for each measurement session, we can remove it from any furthur analysis and model building.
df.drop('profile_id',axis = 1,inplace=True)
df_test.drop('profile_id',axis = 1,inplace=True)


# In[12]:


df.head()


# ## Data Visualization 

# In[13]:


plt.figure(figsize=(20,30))
plt.subplot(5,2,1)
sns.scatterplot(data=df, x = 'ambient', y = 'pm')
plt.subplot(5,2,2)
sns.scatterplot(data=df, x = 'coolant', y = 'pm')
plt.subplot(5,2,3)
sns.scatterplot(data=df, x = 'motor_speed', y = 'pm')
plt.subplot(5,2,4)
sns.scatterplot(data=df, x = 'u_q', y = 'pm')
plt.subplot(5,2,5)
sns.scatterplot(data=df, x = 'u_d', y = 'pm')
plt.subplot(5,2,6)
sns.scatterplot(data=df, x = 'i_q', y = 'pm')
plt.subplot(5,2,7)
sns.scatterplot(data=df, x = 'i_d', y = 'pm')
plt.show()


# ## Extract the independent (input) and dependent (output) variable 

# In[14]:


# extracting independent variable
X = df.iloc[:,:-1].values
# extracting dependent variable
Y = df.iloc[:,-1].values 
print(X.shape)
print(Y.shape)


# ## Splitting the dataset into the Training and Testing sets 

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 100)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# ## Normalization of Dataset 

# In[16]:


# bringing all the features into same range to perform valid predictions
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler() # 0 : 1
X_train =mm.fit_transform(X_train)
X_test = mm.fit_transform(X_test)
print(X_train, X_test)


# In[17]:


print(Y_train, Y_test)


# ## Training dataset with Regression Models

# Here, 4 regression models were used to predict the PM temperature:
# 1. Linear Regression Model
# 2. K-Nearest Neighbour Regressor 
# 3. XGBoost Regressor
# 4. AdaBoost Regressor

# ## Importing Libraries 

# In[18]:


import sklearn
import numpy as np
import pandas as pd
from math import sqrt
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold


# ## Linear Regression Model

# In[19]:


l_reg = LinearRegression()
l_reg.fit(X_train,Y_train)

l_train_acc = (l_reg.score(X_train,Y_train))*100
l_test_acc = (l_reg.score(X_test,Y_test))*100

print(f"Train accuracy: {l_train_acc}")
# printf("Train accuracy is %f", l_train_acc)
print(f"Test accuracy: {l_test_acc}")


# In[20]:


Y_pred = l_reg.predict(X_test)
print(Y_test.shape, Y_pred.shape)


# In[21]:


r2_l = r2_score(Y_test, Y_pred)*100
rms_l = sqrt(mean_squared_error(Y_test, Y_pred))
mae_l = mean_absolute_error(Y_test, Y_pred)
print(f"R^2 score of model is {r2_l} %")
print(f"Root mean squared error is {rms_l}")
print(f"Mean absolute error is {mae_l}")


# ### Performing KFold Cross-Validation (CV) 

# In[22]:


# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=100, shuffle=True)
# create model
l_reg = LinearRegression()
l_reg.fit(X_train, Y_train)
# evaluate model
scores = cross_val_score(l_reg, X_train, Y_train, scoring='r2', cv=cv)
print(f'Score Array list: {scores}') 
print('\n')
# report performance
Y_pred = l_reg.predict(X_test)
r2_l_cv = sklearn.metrics.r2_score(Y_test, Y_pred)*100
print(f'R^2 Score: {r2_l_cv} %')


# ### Evaluation Table 

# In[23]:


calculation = pd.DataFrame(np.c_[Y_test,l_reg.predict(X_test)], columns = ["Original Temperature","Predicted Temperature"])
calculation.head(5)


# ### Visualizing the test results 

# In[24]:


plt.style.use('ggplot') 
plt.figure(figsize=(19,9))
sns.histplot(Y_test, color="red", kde=True, stat="density", linewidth=0, label = 'Original Temperature')
sns.histplot(Y_pred, color="blue", kde=True, stat="density", linewidth=0, label = 'Predicted Temperature')
plt.legend(loc = 'upper right') 
plt.title("PM temperature Prediction for Linear Regressor") 
plt.xlabel("PM Temperature")
plt.ylabel("Density")
plt.show()


# ## K-Nearest Neighbour Regressor 

# In[26]:


k_reg = KNeighborsRegressor(n_neighbors=10,p=2,metric='minkowski')
k_reg.fit(X_train,Y_train)

k_train_acc = (k_reg.score(X_train,Y_train))*100
k_test_acc = (k_reg.score(X_test,Y_test))*100

print(f"Train accuracy: {k_train_acc}")
print(f"Test accuracy: {k_test_acc}")


# In[27]:


Y_pred = k_reg.predict(X_test)
print(Y_test.shape, Y_pred.shape)


# In[28]:


r2_k = r2_score(Y_test, Y_pred)*100
rms_k = sqrt(mean_squared_error(Y_test, Y_pred))
mae_k = mean_absolute_error(Y_test, Y_pred)
print(f"R^2 score of model is {r2_k} %")
print(f"Root mean squared error is {rms_k}")
print(f"Mean absolute error is {mae_k}")


# ### Performing KFold Cross-Validation (CV) 

# In[29]:


# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=100, shuffle=True)
# create model
k_reg = KNeighborsRegressor()
k_reg.fit(X_train, Y_train)
# evaluate model
scores = cross_val_score(k_reg, X_train, Y_train, scoring='r2', cv=cv)
print(f'Score Array list: {scores}') 
print('\n')
# report performance
Y_pred = k_reg.predict(X_test)
r2_k_cv = sklearn.metrics.r2_score(Y_test, Y_pred)*100
print(f'R^2 Score: {r2_k_cv} %')


# ### Evaluation and Visualization

# In[30]:


calculation = pd.DataFrame(np.c_[Y_test,k_reg.predict(X_test)], columns = ["Original Temperature","Predicted Temperature"])
calculation.head(5)


# In[31]:


plt.style.use('ggplot') 
plt.figure(figsize=(19,9))
sns.histplot(Y_test, color="red", kde=True, stat="density", linewidth=0, label = 'Original Temperature')
sns.histplot(Y_pred, color="blue", kde=True, stat="density", linewidth=0, label = 'Predicted Temperature')
plt.legend(loc = 'upper right') 
plt.title("PM temperature Prediction for KNN Regressor") 
plt.xlabel("PM Temperature")
plt.ylabel("Density")
plt.show()


# ## XGBoost Regressor 

# In[32]:


x_reg = XGBRegressor()
x_reg.fit(X_train,Y_train)

x_train_acc = (x_reg.score(X_train,Y_train))*100
x_test_acc = (x_reg.score(X_test,Y_test))*100

print(f"Train accuracy: {x_train_acc}")
print(f"Test accuracy: {x_test_acc}")


# In[33]:


Y_pred = x_reg.predict(X_test)
print(Y_test.shape, Y_pred.shape)


# In[34]:


r2_x = r2_score(Y_test, Y_pred)*100
rms_x = sqrt(mean_squared_error(Y_test, Y_pred))
mae_x = mean_absolute_error(Y_test, Y_pred)
print(f"R^2 score of model is {r2_x} %")
print(f"Root mean squared error is {rms_x}")
print(f"Mean absolute error is {mae_x}")


# ### Performing KFold Cross-Validation (CV)

# In[35]:


# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=100, shuffle=True)
# create model
x_reg = XGBRegressor()
x_reg.fit(X_train, Y_train)
# evaluate model
scores = cross_val_score(x_reg, X_train, Y_train, scoring='r2', cv=cv)
print(f'Score Array list: {scores}') 
print('\n')
# report performance
Y_pred = x_reg.predict(X_test)
r2_x_cv = sklearn.metrics.r2_score(Y_test, Y_pred)*100
print(f'R^2 Score: {r2_x_cv} %')


# ### Evaluation and Visualization 

# In[36]:


calculation = pd.DataFrame(np.c_[Y_test,x_reg.predict(X_test)], columns = ["Original Temperature","Predicted Temperature"])
calculation.head(5)


# In[37]:


plt.style.use('ggplot') 
plt.figure(figsize=(19,9))
sns.histplot(Y_test, color="red", kde=True, stat="density", linewidth=0, label = 'Original Temperature')
sns.histplot(Y_pred, color="blue", kde=True, stat="density", linewidth=0, label = 'Predicted Temperature')
plt.legend(loc = 'upper right') 
plt.title("PM temperature Prediction for XGBoost Regressor") 
plt.xlabel("PM Temperature")
plt.ylabel("Density")
plt.show()


# ## AdaBoost Regressor 

# In[38]:


dtree = DecisionTreeRegressor()
a_reg = AdaBoostRegressor(n_estimators=100, base_estimator=dtree,learning_rate=1)
a_reg.fit(X_train, Y_train)

a_train_acc = (a_reg.score(X_train,Y_train))*100
a_test_acc = (a_reg.score(X_test,Y_test))*100

print(f"Train accuracy: {a_train_acc}")
print(f"Test accuracy: {a_test_acc}")


# In[39]:


Y_pred = a_reg.predict(X_test)
print(Y_test.shape, Y_pred.shape)


# In[40]:


r2_a = r2_score(Y_test, Y_pred)*100
rms_a = sqrt(mean_squared_error(Y_test, Y_pred))
mae_a = mean_absolute_error(Y_test, Y_pred)
print(f"R^2 score of model is {r2_a} %")
print(f"Root mean squared error is {rms_a}")
print(f"Mean absolute error is {mae_a}")


# ### Performing KFold Cross-Validation (CV) 

# In[43]:


# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=100, shuffle=True)
# create model
a_reg = AdaBoostRegressor()
a_reg.fit(X_train, Y_train)
# evaluate model
scores = cross_val_score(a_reg, X_train, Y_train, scoring='r2', cv=cv)
print(f'Score Array list: {scores}') 
print('\n')
# report performance
Y_pred = a_reg.predict(X_test)
r2_a_cv = sklearn.metrics.r2_score(Y_test, Y_pred)*100
print(f'R^2 Score: {r2_a_cv} %')


# ### Evaluation and Visualization 

# In[44]:


calculation = pd.DataFrame(np.c_[Y_test,a_reg.predict(X_test)], columns = ["Original Temperature","Predicted Temperature"])
calculation.head(5)


# In[45]:


plt.style.use('ggplot') 
plt.figure(figsize=(19,9))
sns.histplot(Y_test, color="red", kde=True, stat="density", linewidth=0, label = 'Original Temperature')
sns.histplot(Y_pred, color="blue", kde=True, stat="density", linewidth=0, label = 'Predicted Temperature')
plt.legend(loc = 'upper right') 
plt.title("PM temperature Prediction for AdaBoost Regressor") 
plt.xlabel("PM Temperature")
plt.ylabel("Density")
plt.show()


# ## Evaluation Table 

# In[53]:


models = pd.DataFrame({
    'Algorithm': ['Linear Regression','XGBoost Regressor', 
             'AdaBoost Regressor',  'K-Nearest Neighbours Regressor'],
    'Training Accuracy' : [l_train_acc, x_train_acc, a_train_acc, k_train_acc],
    'Testing Auracy' : [l_test_acc, x_test_acc, a_test_acc, k_test_acc],
    'RMS Score' : [rms_l, rms_x, rms_a, rms_k],
    'MAE Score' : [mae_l, mae_x, mae_a, mae_k],
    'R^2 Score': [ r2_l, r2_x, r2_a, r2_k],
    'CV R^2 Score': [r2_l_cv, r2_x_cv, r2_a_cv, r2_k_cv]
})

models.sort_values(by = ['Training Accuracy', 'Testing Auracy', 'RMS Score', 'MAE Score', 'R^2 Score','CV R^2 Score'], 
                   ascending = True)


# ## Comaprison Graphs 

# ### Algorithm vs. R^2 Score  

# In[54]:


plt.style.use('ggplot') 
plt.figure(figsize=(12,8))
sns.barplot(x='Algorithm',y='R^2 Score',data=models)
plt.title("Best Model Prediction w.r.t. R^2 Score") 
plt.show()


# ### Algorithm vs. CV R^2 Score  

# In[55]:


plt.style.use('ggplot') 
plt.figure(figsize=(12,8))
sns.barplot(x='Algorithm',y='CV R^2 Score',data=models)
plt.title("Best Model Prediction w.r.t. Cross-Validation R^2 Score") 
plt.show()


# ### Algorithm vs. Training Accuracy

# In[56]:


plt.style.use('ggplot') 
plt.figure(figsize=(12,8))
sns.barplot(x='Algorithm',y='Training Accuracy',data=models)
plt.title("Model Prediction w.r.t. Training accuracy")
plt.show()


# ### Algorithm vs. Testing Accuracy 

# In[57]:


plt.style.use('ggplot') 
plt.figure(figsize=(12,8))
sns.barplot(x='Algorithm',y='Testing Auracy',data=models)
plt.title("Model Prediction w.r.t. Testing accuracy")
plt.show()

