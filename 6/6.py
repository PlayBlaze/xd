#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense 
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


# In[2]:


# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[3]:


# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) 
x = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# In[4]:


# Freeze base model layers 
base_model.trainable = False


# In[5]:


# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[6]:


# Train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


# In[7]:


# Predict and visualize some test images 
predictions = model.predict(x_test[:9]) 
plt.figure(figsize=(10, 10))
for i in range(9): 
    plt.subplot(3, 3, i+1) 
    plt.imshow(x_test[i])
    plt.title(f"True: {y_test[i][0]}, Pred: {predictions[i].argmax()}") 
    plt.axis('off')
plt.show()


# In[ ]:




