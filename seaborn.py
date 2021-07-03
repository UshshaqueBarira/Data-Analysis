#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib.pyplot', '% inline')


# In[11]:


sns.get_dataset_names()


# In[12]:


attention=sns.load_dataset('attention')
attention.head()


# In[13]:


sns.relplot(x='subject',y='score',data=attention,hue='attention',size='subject')


# In[14]:


tips=sns.load_dataset('tips')
tips.head()


# In[15]:


sns.scatterplot(x='total_bill',y='tip',data=tips)


# In[16]:


# using linear regression technique----one independent variable and one dependent variable(total_bill, tip)
import sklearn.linear_model
tips.head()


# In[17]:


x=tips['total_bill']
y=tips['tip']


# In[29]:


x.train=x[:100]
x.test=x[-100:]
y.train=y[:100]
y.test=y[-100:]


# In[18]:


plt.scatter(x.test,y.test,color='blue')


# In[19]:


regr=linear_model.LinearRegression()
regr.fit(x.train,y.train)
plt.plot(x.test,regr.predict(x.test),color='green',linewidth=2)


# In[20]:


sns.set_style('dark')
sns.regplot(x,y,data=tips,color='green')


# In[24]:


#using the different dataset as car_crashes
car_crashes=sns.load_dataset('car_crashes')
car_crashes.head()


# In[25]:


penguins=sns.load_dataset('penguins')
penguins.head()


# In[29]:


#cross dimensional features correlation graph
sns.pairplot(penguins,hue='species',height=2.5);


# In[31]:


sns.relplot(x='bill_length_mm',y='bill_depth_mm',data=penguins,hue='sex')


# In[35]:


sns.set_style('white')
sns.scatterplot(x='bill_length_mm',y='species',data=penguins,color='green')


# In[37]:



sns.scatterplot(x='bill_length_mm',y='sex',data=penguins,color='orange')


# In[ ]:




