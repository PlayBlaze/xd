#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install tensorflow matplotlib 
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# In[2]:


# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)


# In[3]:


# Build the model with regularization and dropout 
model = Sequential([
    
    Dense(128, activation='relu', input_shape=(784,), 
          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
Dropout(0.5),
Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)), Dropout(0.5),
Dense(10, activation='softmax')
])


# In[4]:


# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), verbose=2)


# In[5]:


# Plot training and validation accuracy 
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.title('Training and Validation Accuracy with Regularization') 
plt.xlabel('Epochs')
plt.ylabel('Accuracy') 
plt.legend() 
plt.show()


# In[ ]:




