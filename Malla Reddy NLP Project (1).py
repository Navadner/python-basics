#!/usr/bin/env python
# coding: utf-8

# # Day 1

# ### Importing the data 
# 
# Link to data - https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification?resource=download

# In[16]:


import pandas as pd 
import numpy as np 


# In[17]:


df = pd.read_csv('bbc_data.csv')


# In[18]:


df


# In[ ]:





# In[19]:


df


# In[20]:


df['labels'].unique()


# In[21]:


df['labels'].nunique()


# In[22]:


df['data'].nunique()


# In[23]:


df[['data','labels']].drop_duplicates()


# In[24]:


df.isna().sum()


# In[25]:


df['labels'].value_counts()


# In[26]:


l = [1,2.,1,8,9]


# In[27]:


dir(l)


# In[28]:


l.remove()


# In[29]:


dir(df)


# # Day 2

# ## Prepping the data 

# ### Converting all the text column to lower case

# In[30]:


df['data'] = df['data'].str.lower()


# In[31]:


df


# ### Import spacy for other text cleaning purposes

# In[ ]:


get_ipython().system('pip install spacy')


# In[ ]:


get_ipython().system(' python -m spacy download en_core_web_sm')


# In[32]:


import spacy
nlp = spacy.load('en_core_web_sm')


# ### Remove Punctuations

# In[33]:


df['data'] = df['data'].str.replace(',', ' ')
df['data'] = df['data'].str.replace('.', ' ')
df['data'] = df['data'].str.replace('-', ' ')
df['data'] = df['data'].str.replace('"', ' ')
df['data'] = df['data'].str.replace('  ',' ')


# In[34]:


df


# ### Remove stop words 

# In[38]:


stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 
              'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
               'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing',
               'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn',
               "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
               'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me',
               'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'now', 'o', 'of',
               'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan',
               "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that',
               "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
               'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what',
               'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y',
               'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']


# In[39]:


def stop_word_removal(text):
    tokens = text.split(' ')
    words_without_stopwords = [x for x in tokens if x not in stop_words]
    final_sentence = ' '.join(words_without_stopwords)
    return final_sentence


# In[43]:


df['clean_data'] = df['data'].apply(stop_word_removal)


# In[44]:


df


# ### Get the vector embeddings

# In[45]:


def get_embeddings(text):
    doc = nlp(text)
    return doc.vector


# In[46]:


df["embeddings"] = df["clean_data"].apply(get_embeddings)


# In[47]:


df


# In[50]:


np.vstack(df["embeddings"].values).shape


# In[ ]:





# In[ ]:




