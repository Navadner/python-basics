#!/usr/bin/env python
# coding: utf-8

# # Day 1

# ## Importing the data 
# 
# Link to data - https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification?resource=download

# In[2]:


import pandas as pd 
import numpy as np 


# In[3]:


df = pd.read_csv('bbc_data.csv')


# In[4]:


df


# In[5]:


df['labels'].unique()


# In[7]:


df['labels'].nunique()


# In[8]:


df.isna().sum()


# In[9]:


df['data'].isna().sum()


# In[10]:


df['labels'].value_counts()


# In[ ]:




