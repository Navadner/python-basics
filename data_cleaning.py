#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
dataset = pd.read_csv("tweets.csv",encoding = "ISO-8859-1")
dataset.head()


# In[3]:


import re

def clean_text(text):
   
    text = re.sub(r'RT', '', text)
    
    text = re.sub(r'&amp:', '', text)
    
    text = re.sub(r'[?.;:,#@-]', '', text)
    
    text = text.lower()
    
    return text


# In[9]:


pip install wordcloud


# In[10]:


from wordcloud import STOPWORDS
print(STOPWORDS)


# In[ ]:




