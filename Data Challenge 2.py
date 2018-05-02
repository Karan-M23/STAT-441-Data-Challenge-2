
# coding: utf-8

# Karan Mehta 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot
import scipy.io as sio
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras.layers import Convolution2D


# In[ ]:


train = sio.loadmat('/Users/Karan_M/Documents/STAT 441/Data Challenge 2/train.mat', squeeze_me=True)
X_train = train['X']
y_train = train['y']
test = sio.loadmat('/Users/Karan_M/Documents/STAT 441/Data Challenge 2/test.mat', squeeze_me=True)
X_test = test['X']
X_test.shape


# In[ ]:


X_train = np.transpose(X_train)
X_test = np.transpose(X_test)


# In[ ]:


input_shape = (3,32,32)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = np.true_divide(X_train, 255)
X_test = np.true_divide(X_test, 255)


# In[ ]:


y_train[y_train == 10] = 0
y_cat = np_utils.to_categorical(y_train)


# In[ ]:


def CNN(): 
    model = Sequential()
    model.add(Convolution2D(32,(3,3),activation='relu',data_format='channels_first',input_shape=input_shape))
    model.add(Convolution2D(32,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# In[ ]:


classifier = CNN()
batch_size = 32
epochs = 20
classifier.fit(X_train, y_cat, batch_size=batch_size, epochs=epochs, verbose=1)


# In[ ]:


y_pred = classifier.predict_classes(X_test)


# In[ ]:


y_pred[y_pred == 0] = 10


# In[ ]:


submission = pd.DataFrame(y_pred, columns=['class'])
submission.head()
submission.to_csv('/Users/Karan_M/Documents/STAT 441/Data Challenge 2/sub_3.csv')

