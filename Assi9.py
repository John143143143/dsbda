#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for 
# distribution of age with respect to each gender along with the information about whether 
# they survived or not. (Column names : 'sex' and 'age')
# 2. Write observations on the inference from the above statistics


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


dataset = sns.load_dataset('titanic')


# In[3]:


dataset.isnull().sum()


# In[4]:


dataset.drop(columns=['deck','embark_town'],axis=1,inplace=True)
dataset['age'].fillna(dataset['age'].mean(),inplace=True)
dataset['embarked'].fillna(dataset['embarked'].mode()[0],inplace=True)


# In[5]:


dataset.isnull().sum()


# In[6]:


sns.boxplot(x='sex',y='age',data=dataset)


# In[8]:


sns.boxplot(x='sex',y='age',hue='survived',data=dataset)


# In[ ]:




