#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Perform the following operations on any open source dataset (e.g., data.csv)
# 1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for 
# a dataset (age, income etc.) with numeric variables grouped by one of the qualitative 
# (categorical) variable. For example, if your categorical variable is age groups and 
# quantitative variable is income, then provide summary statistics of income grouped by 
# the age groups. Create a list that contains a numeric value for each response to the 
# categorical variable. 
# 2. Write a Python program to display some basic statistical details like percentile, mean, 
# standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Irisversicolor’ of iris.csv dataset.
# Provide the codes with outputs and explain everything that you do in this step


# In[1]:


import numpy as  np
import pandas as pd
import seaborn as sns


# In[2]:


tip_df=sns.load_dataset('tips')


# In[3]:


tip_df.head()


# In[4]:


tip_df.info()


# In[5]:


tip_df.describe()


# In[6]:


tip_df.isnull().sum()


# In[7]:


tip_df['total_bill'].value_counts()


# In[8]:


tip_df['sex'].value_counts()


# In[10]:


tip_df.groupby('sex').describe()


# In[11]:


tip_df.groupby('sex')['total_bill'].describe()


# In[12]:


tip_df.groupby('sex')['tip'].describe()


# In[13]:


tip_df.groupby('day')['total_bill'].describe()


# In[14]:


tip_df.groupby('day')['tip'].describe()


# In[15]:


tip_df.groupby('time')['total_bill'].describe()


# In[18]:


tip_df.groupby('time')[['tip','size']].describe()


# In[19]:


tip_df.groupby(['sex','day'])['total_bill'].describe()


# In[20]:


tip_df.groupby(['smoker','day'])['total_bill'].describe()


# In[21]:


tip_df.groupby(['smoker','day','sex'])['total_bill'].describe()


# In[22]:


tip_df.groupby(['smoker','day','sex','time'])['total_bill'].describe()


# In[23]:


iris_df = sns.load_dataset('iris')


# In[24]:


iris_df.shape


# In[25]:


iris_df.info()


# In[26]:


iris_df.describe()


# In[27]:


iris_df.isnull().sum()


# In[28]:


iris_df['species'].value_counts()


# In[30]:


iris_df.groupby(['species'])['sepal_length'].describe()


# In[32]:


iris_df.groupby(['species'])['sepal_width'].describe()


# In[34]:


iris_df.groupby(['species'])[['petal_length','petal_width']].describe()


# In[ ]:




