#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.autograd as ag
import tensorflow as tf
from tensorflow.python.client import device_lib

import pandas as pd
import cv2
import glob
import pickle
from matplotlib import pyplot as plt
import scipy.misc


# In[2]:


import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras import backend as K
from keras import applications

import os


# In[3]:


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


# In[4]:


x_train = np.load('train_data.npy')
y_train = np.load('train_lbl.npy')


# In[5]:


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# In[6]:


print(device_lib.list_local_devices())


# In[7]:


K.tensorflow_backend._get_available_gpus()


# In[21]:


model = applications.vgg16.VGG16(include_top=True,weights=None,input_shape=(64,64,3),classes=5)


# In[22]:


model.compile(loss='categorical_crossentropy',
optimizer=Adam(),
metrics=['accuracy'])

model.summary()


# In[23]:


checkpoint = ModelCheckpoint(filepath='.',monitor='val_acc',
                             verbose=1,save_best_only=True)
def lr_sch(epoch):
    #200 total
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5
lr_scheduler = LearningRateScheduler(lr_sch)
lr_reducer = ReduceLROnPlateau(monitor='val_acc',factor=0.2,patience=5,
                               mode='max',min_lr=1e-3)
callbacks = [checkpoint,lr_scheduler,lr_reducer]


# In[24]:


history = model.fit(x_train,y_train,batch_size=64,epochs=200,validation_split=0.3,validation_data=None,verbose=1,callbacks=callbacks)


# In[ ]:


# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = 'history_VGG16.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv: 
hist_csv_file = 'history_VGG16.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# In[ ]:





# In[ ]:




