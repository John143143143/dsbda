#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1. Implement logistic regression using Python/R to perform classification on 
# Social_Network_Ads.csv dataset.
# 2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, 
# Recall on the given dataset.


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score


# In[4]:


df=pd.read_csv('Social_Network_Ads.csv')


# In[6]:


df.columns


# In[7]:


x=df[['Age', 'EstimatedSalary']]
y=df[['Purchased']]


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=29)


# In[9]:


model=LogisticRegression()


# In[11]:


model.fit(x_train,y_train)


# In[12]:


y_pred=model.predict(x_test)
y_pred


# In[13]:


model.score(x_train,y_train)


# In[14]:


model.score(x,y)


# In[16]:


cm=confusion_matrix(y_test,y_pred)
cm


# In[20]:


ac=accuracy_score(y_test,y_pred)
ac


# In[22]:


rc=recall_score(y_test,y_pred)
rc


# In[23]:


pc=precision_score(y_test,y_pred)
pc


# In[26]:


e=1-ac


# In[27]:


e


# In[29]:


print(confusion_matrix.__doc__)


# In[30]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()


# In[31]:


tn,fp,fn,tp


# In[ ]:




