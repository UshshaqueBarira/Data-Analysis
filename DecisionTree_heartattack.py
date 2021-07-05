#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[49]:


heart=pd.read_csv("./heart.csv")
heart.head()


# In[50]:


sns.set_style('white')


# In[52]:


sns.relplot(x='age',y='chol',data=heart,color='green',hue='sex')


# In[54]:


sns.relplot(x='age',y='cp',data=heart,hue='sex')


# In[68]:


feature_cols=['age','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall','output']
feature_cols


# In[115]:


X=heart[feature_cols]
y=heart.sex
y1=heart.chol


# In[116]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[117]:


clf=DecisionTreeClassifier()
clf=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)


# In[118]:


print("Accuracy:(Gender)",(metrics.accuracy_score(y_test,y_pred))*100)


# In[122]:


x_train,x_test,y_train,y_test=train_test_split(X,y1,test_size=0.4,random_state=1)


# In[123]:


clf1=DecisionTreeClassifier()
clf1=clf1.fit(x_train,y_train)
y_pred=clf1.predict(x_test)


# In[124]:


print("Accuracy:(Cholestrol)",(metrics.accuracy_score(y_test,y_pred)*100))


# In[ ]:




