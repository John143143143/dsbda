#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ownload the Iris flower dataset or any other dataset into a DataFrame. (e.g.,
# https://archive.ics.uci.edu/ml/datasets/Iris ). Scan the dataset and give the inference as:
# 1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
# 2. Create a histogram for each feature in the dataset to illustrate the feature distributions. 
# 3. Create a box plot for each feature in the dataset. 
# 4. Compare distributions and identify outliers.


# In[1]:


import pandas as pd
import numpy as snsnp
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=sns.load_dataset('iris')


# In[3]:


df.head()


# In[11]:


fig,axes=plt.subplots(2,2,figsize=(16,9))
sns.histplot(df['sepal_length'],ax=axes[0,0])
sns.histplot(df['sepal_width'],ax=axes[0,1])
sns.histplot(df['petal_length'],ax=axes[1,0])
sns.histplot(df['petal_width'],ax=axes[1,1])


# In[13]:


fig,axes = plt.subplots(2,2,figsize=(16,9))
sns.boxplot(y='petal_length',x='species',data=df,ax=axes[0,0])
sns.boxplot(y='petal_width',x='species',data=df,ax=axes[0,1])
sns.boxplot(y='sepal_length',x='species',data=df,ax=axes[1,0])
sns.boxplot(y='sepal_width',x='species',data=df,ax=axes[1,1])


# In[ ]:




