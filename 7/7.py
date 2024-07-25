#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# a) E.g. (4-to-1 RNN) to show that the quantity of rain on a certain day also depends on the values of the previous day


# In[1]:


import numpy as np 
import tensorflow as tf


# In[2]:


# Generate synthetic data
data = np.random.rand(100, 1) 
X, y = [], []
for i in range(len(data) - 4): 
    X.append(data[i:i+4]) 
    y.append(data[i+4])
X, y = np.array(X), np.array(y)


# In[3]:


# Build and train the RNN model 
model = tf.keras.Sequential([
tf.keras.layers.SimpleRNN(50, input_shape=(4, 1)), 
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse') 
model.fit(X, y, epochs=200, verbose=0)


# In[4]:


# Predict the next value
print('Predicted Rainfall for the next day: ',model.predict(data[-4:].reshape(1, 4, 1)))


# In[ ]:


# b) LSTM for sentiment analysis on datasets like UMICH SI650 for similar.


# In[5]:


import tensorflow as tf
from sklearn.model_selection import train_test_split 
import nltk
from nltk.corpus import movie_reviews 
nltk.download('movie_reviews')


# In[6]:


# Load and preprocess data
sentences = [" ".join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()] 
labels = [1 if fileid.startswith('pos') else 0 for fileid in movie_reviews.fileids()]


# In[7]:


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000) 
X = tokenizer.texts_to_sequences(sentences)
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=100) 
y = np.array(labels)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


# In[9]:


# Build and train the LSTM model
model = tf.keras.Sequential([
tf.keras.layers.Embedding(5000, 128, input_length=100), 
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))


# In[10]:


# Evaluate the model
print('Test Accuracy: ',model.evaluate(X_test, y_test)[1])


# In[ ]:




