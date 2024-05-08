#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information 
# about the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library 
# to see if we can find any patterns in the data. 
# 2. Write a code to check how the price of the ticket (column name: 'fare') for each 
# passenger is distributed by plotting a histogram.


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


df = sns.load_dataset('Titanic')


# In[5]:


df.isnull().sum()


# In[6]:


df.drop(columns=['deck','embark_town'],axis=1,inplace=True)


# In[7]:


df.isnull().sum()


# In[8]:


df['age'].fillna(df['age'].mean(),inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df['embarked'].fillna(df['embarked'].mode()[0],inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


df


# In[20]:


fig,axes = plt.subplots(1,3)
sns.histplot(x=df['sex'],hue=df['sex'],multiple='dodge',shrink=0.8,data=df,ax=axes[0])
sns.histplot(x='pclass',hue='pclass',multiple='dodge',shrink=0.8,data=df,ax=axes[1])
sns.histplot(x='survived',hue='survived',multiple='dodge',shrink=0.8,data=df,ax=axes[2])
plt.show()


# In[21]:


fig,axes = plt.subplots(1,2)
sns.histplot(x='age',kde=True,multiple='dodge',shrink=0.8,data=df,ax=axes[0])
sns.histplot(x='fare',kde=True,multiple='dodge',shrink=0.8,data=df,ax=axes[1])

plt.show()


# In[22]:


fig,axes = plt.subplots(2,2)
sns.histplot(x='age',hue='sex',multiple='dodge',shrink=0.8,data=df,ax=axes[0,0])
sns.histplot(x='fare',hue='sex',multiple='dodge',shrink=0.8,data=df,ax=axes[0,1])
sns.histplot(x='age',hue='survived',multiple='dodge',shrink=0.8,data=df,ax=axes[1,0])
sns.histplot(x='fare',hue='survived',multiple='dodge',shrink=0.8,data=df,ax=axes[1,1])
plt.show()


# In[ ]:




