#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score,fbeta_score,classification_report


# In[3]:


df = pd.read_csv('iris.csv')


# In[4]:


df.columns


# In[6]:


x=df[['5.1', '3.5', '1.4', '0.2']]
y=df[['Iris-setosa']]


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=29)


# In[8]:


classifier  = GaussianNB()


# In[9]:


classifier.fit(x_train,y_train)


# In[10]:


y_pred=classifier.predict(x_test)


# In[11]:


classifier.score(x_train,y_train)


# In[12]:


classifier.score(x_test,y_test)


# In[15]:


cm=confusion_matrix(y_test,y_pred)
cm


# In[16]:


accuracy_score(y_test,y_pred)


# In[18]:


precision_score(y_test,y_pred,average='micro')


# In[19]:


recall_score(y_test,y_pred,average='micro')


# In[21]:


fbeta_score(y_test,y_pred,beta=0.5,average='micro')


# In[22]:


print(classification_report(y_test,y_pred))


# In[ ]:




