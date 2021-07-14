#!/usr/bin/env python
# coding: utf-8

# ### Import Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ### Read data from csv file and convert to dataframe

# In[2]:


df=pd.read_csv(r'C:\Users\Amaan\OneDrive\Documents\Projects\Fake News Detection\\news.csv')


# In[3]:


#Get shape and head
df.shape
df.head()


# In[4]:


#Get the labels
labels=df.label
labels.head()


# ### Split data into test and train sets

# In[5]:


x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# #### Initialize a TfidfVectorizer

# In[6]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# ### Fit and transform train set, transform test set

# In[7]:


tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# ### Initialize a PassiveAggressiveClassifier

# In[8]:


pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# #### Predict on the test set and calculate accuracy

# In[9]:


y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# #### Build confusion matrix

# In[10]:


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# The confusion matrix tells us that we obtained 589 true positives, 587 true negatives, 42 false positives, and 49 false negatives

# In[ ]:




