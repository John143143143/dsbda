#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


# In[ ]:


from nltk import word_tokenize, sent_tokenize


# In[ ]:


corpus = "Sachin was the GOAT of the previous generation. Virat is the GOAT of this generation. Shubman will be the GOAT of the next generation"


# In[ ]:



print(word_tokenize(corpus))
print(sent_tokenize(corpus))


# In[ ]:


#tagging
from nltk import pos_tag


# In[ ]:


tokens = word_tokenize(corpus)
print(pos_tag(tokens))


# In[ ]:


#stopwordsremoval
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))


# In[ ]:


tokens = word_tokenize(corpus)
cleaned_tokens = []
for token in tokens:
  if (token not in stop_words):
    cleaned_tokens.append(token)
print(cleaned_tokens)
     


# In[ ]:


#stemming 
from nltk.stem import PorterStemmer


# In[ ]:


stemmer = PorterStemmer()


# In[ ]:


stemmed_tokens = []
for token in cleaned_tokens:
  stemmed = stemmer.stem(token)
  stemmed_tokens.append(stemmed)
print(stemmed_tokens)


# In[ ]:


#Lemmatization

from nltk.stem import WordNetLemmatizer


# In[ ]:


lemmatizer = WordNetLemmatizer()


# In[ ]:


lemmatized_tokens = []
for token in cleaned_tokens:
  lemmatized = lemmatizer.lemmatize(token)
  lemmatized_tokens.append(lemmatized)
print(lemmatized_tokens)


# In[ ]:


#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:



corpus = [
    "Sachin was the GOAT of the previous generation",
    "Virat is the GOAT of the this generation",
    "Shubman will be the GOAT of the next generation"
]
     


# In[ ]:


vectorizer = TfidfVectorizer()


# In[ ]:



matrix = vectorizer.fit(corpus)
matrix.vocabulary_


# In[ ]:


tfidf_matrix = vectorizer.transform(corpus)
print(tfidf_matrix)


# In[ ]:


tfidf_matrix = vectorizer.transform(corpus)
print(tfidf_matrix)

