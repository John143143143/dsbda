#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Create a Linear Regression Model using Python/R to predict home prices using Boston Housing 
# Dataset (https://www.kaggle.com/c/boston-housing). The Boston Housing dataset contains 
# information about various houses in Boston through different parameters. There are 506 samples 
# and 14 feature variables in this dataset. 
# The objective is to predict the value of prices of the house using the given features.


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[3]:


df = pd.read_csv('housing.csv')


# In[4]:


df.columns


# In[5]:


x=df[['RM', 'LSTAT', 'PTRATIO']]
y=df[['MEDV']]


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=29)


# In[7]:


model=LinearRegression()


# In[12]:


model.fit(x_train,y_train)


# In[21]:


y_pred=model.predict(x_test)
y_pred


# In[22]:


model.score(x_train,y_train)


# In[23]:


model.score(x_test,y_test)


# In[24]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:




