#!/usr/bin/env python
# coding: utf-8

# In[3]:


#titanic data set is all manipulated thus we have an accuracy level of 1.0 that is 100 matching as trained and test data

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns


# In[4]:


sns.set_style('dark')


# In[51]:


titanic=sns.load_dataset('titanic')
titanic.head()


# In[66]:


feature_cols=['survived','pclass','sibsp','parch','fare']


# In[78]:


X=titanic[feature_cols]
#y=titanic.pclass
y1=titanic.survived
#print(X.isnull())


# In[79]:


x_train,x_test,y_train,y_test=train_test_split(X,y1,test_size=0.4,random_state=1)#test 30% and 70% train data


# In[80]:


clf=DecisionTreeClassifier()
clf=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)


# In[81]:


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[ ]:




