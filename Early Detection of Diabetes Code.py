#!/usr/bin/env python
# coding: utf-8

# In[7]:



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[8]:


balance_data = pd.read_csv('diabetes_data.csv',sep= ',', header = 0)


# In[9]:


print ("Dataset Lenght:: "), len(balance_data)
print ("Dataset Shape:: "), balance_data.shape


# In[12]:


print ("Dataset:: ")
balance_data.head()


# In[14]:


#get all categorical columns
cat_columns = balance_data.select_dtypes(['object']).columns
#convert all categorical columns to numeric
balance_data[cat_columns] = balance_data[cat_columns].apply(lambda x: pd.factorize(x)[0])


# In[15]:


X = balance_data.values[:, 12:15]
Y = balance_data.values[:,16]


# In[16]:


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[17]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


# In[18]:


y_pred_en = clf_entropy.predict(X_test)
y_pred_en


# In[19]:


print ("Accuracy is "), accuracy_score(y_test,y_pred_en)*100


# In[ ]:





# In[ ]:




