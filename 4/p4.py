#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense 
from tensorflow.keras.models import Model 
from tensorflow.keras.datasets import mnist 
import matplotlib.pyplot as plt


# In[2]:


# Load and Prepare Data
(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0


# In[3]:


# Define the Autoencoder Model 
input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) 
encoded = Dense(32, activation='relu')(encoded) 
decoded = Dense(64, activation='relu')(encoded) 
decoded = Dense(784, activation='sigmoid')(decoded)


# In[4]:


autoencoder = Model(input_img, decoded) 


# In[5]:


# Compile and Train the Autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))


# In[6]:


# Predict and Display Results 
decoded_imgs = autoencoder.predict(x_test)


# In[10]:


# Plotting original and reconstructed images 
plt.figure(figsize=(20, 4))
for i in range(10):
# Original images
    ax = plt.subplot(2, 10, i + 1) 
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray') 
    ax.axis('off')
    # Reconstructed images
    ax = plt.subplot(2, 10, i + 11) 
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray') 
    ax.axis('off')

plt.suptitle('Original and Reconstructed Images')
plt.show()


# In[ ]:





# In[ ]:




