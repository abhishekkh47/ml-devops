#!/usr/bin/env python
from keras.datasets import mnist
from keras.backend import clear_session
#import tensorflow
#import keras
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adam
import numpy
dataset = mnist.load_data('mymnist.db')

epoch=4
train , test = dataset

X_train , y_train = train

X_test , y_test = test

X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)

X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')

y_train_cat = to_categorical(y_train)

model = Sequential()
model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fitting = model.fit(X_train, y_train_cat, epochs=epoch,verbose=False)
accuracy=fitting.history['accuracy'][-1] *100
model.summary()

if (accuracy < 80.0):
	b=("the model has been trained sussesfully with the accuracy of ",accuracy,"%")    
	f=open("/root/accuracy.txt","w")
	f.write(b)
	f.close()
	print("less than 80")
    #	execfile("/root/print.py")
else:
    b=("the model has been trained sussesfully with the accuracy of ",accuracy,"%")    
    f=open("/root/accuracy.txt","w")
    f.write(b)
    f.close()
    model.save("/root/mnist.h5")
