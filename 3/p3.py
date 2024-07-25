#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install tensorflow matplotlib 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt 
import numpy as np


# In[2]:


# Load and preprocess the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# In[4]:


# Build the CNN model 
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
MaxPooling2D((2, 2)),
Conv2D(32, (3, 3), activation='relu'),
MaxPooling2D((2, 2)), Flatten(),
Dense(10, activation='softmax') ])


# In[5]:


# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)


# In[6]:


# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test) 
print(f'Test accuracy: {test_acc:.2f}')


# In[7]:


# Predict and visualize some test images 
predictions = model.predict(X_test[:10]) 
plt.figure(figsize=(20, 4))
for i in range(10): 
    plt.subplot(2, 10, i + 1) 
    plt.imshow(X_test[i]) 
    plt.xticks([])
    plt.yticks([]) 
    plt.title(classes[np.argmax(predictions[i])])
plt.show()


# In[ ]:




