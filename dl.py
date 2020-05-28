#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras.backend import clear_session
import tensorflow
import keras
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adam
import numpy


# In[2]:


dataset = mnist.load_data('mymnist.db')


# In[3]:


epoch=4
train , test = dataset


# In[4]:


X_train , y_train = train


# In[5]:


X_test , y_test = test


# In[6]:


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[7]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[8]:
# In[9]:


y_train_cat = to_categorical(y_train)


# In[10]:


model = Sequential()
model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fitting = model.fit(X_train, y_train_cat, epochs=epoch,verbose=False)
accuracy=fitting.history['accuracy'][-1] *100
model.summary()


# In[11]:

if (accuracy < 97.0):
    execfile("/root/print.py")
else:
    b=("the model has been trained sussesfully with the accuracy of ",accuracy,"%")    
    f=open("/root/model.txt","w")
    f.write(b)
    f.close()
    model.save("/root/mnist.h5")