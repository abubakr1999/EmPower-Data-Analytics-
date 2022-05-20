#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\\Users\user\Desktop\\FYP")

from sklearn.model_selection import train_test_split
import cv2 

#keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import sklearn.metrics as metrics

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image

import cv2 as cv
import numpy as np
import os
from PIL import Image

import My_module
from matplotlib import pyplot as plt

import scipy 
#!pip install detecta
from detecta import detect_peaks
from scipy.signal import find_peaks

# !pip install malaya
import malaya
# !pip3 install Spylls


# In[4]:


train = pd.read_csv("emnist-balanced-train.csv",delimiter = ',')
test = pd.read_csv("emnist-balanced-test.csv", delimiter = ',')
label_map = pd.read_csv("emnist-balanced-mapping.txt", delimiter = ' ',                    index_col=0, header=None, squeeze=True)


# In[5]:


unique_values=pd.unique(train.iloc[:,0])
print(unique_values)


# In[6]:


#writing out the labels explicitly in terms of numbers and alphabets

label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)

label_dictionary


# In[7]:


# Constants
HEIGHT = 28
WIDTH = 28


# In[8]:


# Split x and y
train_x = train.iloc[:,1:]
train_y = train.iloc[:,0]
del train

test_x = test.iloc[:,1:]
test_y = test.iloc[:,0]
del test


# In[9]:


# Flip and rotate image
train_x = np.asarray(train_x)
train_x = np.apply_along_axis(My_module.rotate, 1, train_x)

test_x = np.asarray(test_x)
test_x = np.apply_along_axis(My_module.rotate, 1, test_x)


# In[10]:


# Normalise
train_x = train_x.astype('float32')
train_x /= 255
test_x = test_x.astype('float32')
test_x /= 255


# In[11]:


# plot random images from the training set
from random import sample
fig, axs = plt.subplots(3,3)
indices = sample(range(train_x.shape[0]),9)
for row in range(3):
    for col in range(3):
        index = indices[3*row + col]
        axs[row, col].imshow(train_x[index], cmap=plt.get_cmap('gray'))
        axs[row, col].set_title(label_dictionary[train_y[index]])
        axs[row, col].set_xticks([0, 14, 27])
        axs[row, col].set_yticks([0, 14, 27])
fig.tight_layout(pad=1)


# In[12]:


# plot random images from the training set
from random import sample
v=[]
for i in range(len(train_x)):
    if train_y[i]==21:
        v.append(i)
        
print(v)
indices = sample(v,100)
fig, axs = plt.subplots(10,10)
plt.figure(figsize=(100000, 100000))

for row in range(10):
    for col in range(10):
        index = indices[10*row + col]
        axs[row, col].imshow(train_x[index], cmap=plt.get_cmap('gray'))
        axs[row, col].set_title(label_dictionary[train_y[index]])
        axs[row, col].set_xticks([0, 14, 27])
        axs[row, col].set_yticks([0, 14, 27])
fig.tight_layout(pad=1)


# In[13]:


# number of classes
num_classes = 47


# In[14]:


# to categories one hot encoding
train_y_cat = np_utils.to_categorical(train_y, num_classes)
test_y_cat = np_utils.to_categorical(test_y, num_classes)


# In[15]:


# Reshape image for CNN
#Tensorflow (batch, width, height, channel)
train_x = train_x.reshape(-1, WIDTH, HEIGHT, 1)
test_x = test_x.reshape(-1, WIDTH, HEIGHT, 1)


# In[16]:


# partition to train and val
train_x, val_x, train_y_cat, val_y_cat = train_test_split(train_x, train_y_cat, test_size= 0.10, random_state=7)


# In[17]:


#cnn model
model = Sequential()

model.add(Conv2D(filters=168, kernel_size=(3,3), padding = 'same', activation='relu',                 input_shape=(HEIGHT, WIDTH,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters=168, kernel_size=(3,3) , padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=168, kernel_size=(3,3) , padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(units=num_classes, activation='softmax'))

model.summary()


# In[18]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


history = model.fit(train_x, train_y_cat, epochs=10, batch_size=512, verbose=1,validation_data=(val_x, val_y_cat))


# In[ ]:


#%%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
print(max(acc))
print(max(val_acc))
print(max(loss))
print(max(val_loss))
# Accuracy curve
My_module.plotgraph(epochs, acc, val_acc)
My_module.plotgraph1(epochs,loss,val_loss)


# In[ ]:


score = model.evaluate(test_x, test_y_cat, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(test_x)
cm = metrics.confusion_matrix(test_y_cat.argmax(axis=1), y_pred.argmax(axis=1))
df_cm=pd.crosstab(test_y_cat.argmax(axis=1), y_pred.argmax(axis=1),rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[ ]:



# from IPython.display import Image
# Image('Address.jpeg')
os.chdir('C:\\Users\\user\\Desktop\\FYP')
im = Image.open('Address.jpeg')
im.show()


# In[ ]:


I = np.asarray(Image.open('Address.jpeg'))


# In[ ]:


img1 = im.convert("L")
img1.show()
img2=np.asarray(img1)
img3=255-img2


# In[ ]:


im = Image.fromarray(np.uint8(img3))
im.show('img3',img3)


# In[ ]:


from statistics import mean
row_means = img3.mean(axis=1)
column_means=img3.mean(axis=0)


# In[ ]:


plt.plot(row_means)
plt.title('Plot of row means against row index')
plt.xlabel('row index')
plt.ylabel('row mean')


# In[ ]:


plt.plot(column_means)
plt.title('Plot of column means against column index')
plt.xlabel('column index')
plt.ylabel('column mean')


# In[ ]:


boolArr = (row_means>10)
result = np.where(boolArr)


# In[ ]:


img4=img3[308:430,]
print(img4)
img4_pil=Image.fromarray(np.uint8(img4))
img4_pil.show()


# In[ ]:


img5= np.asarray(img4_pil)
column_means=img5.mean(axis=0)
column_plt=plt.plot(column_means)
plt.title('Plot of column means against column index')
plt.xlabel('column index')
plt.ylabel('column mean')


# In[ ]:


peaks=detect_peaks(-column_means,show=True)
column_plt=plt.plot(peaks)
gray_intensity_values_of_peaks = column_means[peaks]


# In[ ]:


new1=column_means[peaks]
# plt.scatter(new1[new1<20],new1[new1<20])
position=np.where(column_means<20)
difference=np.diff(np.where(column_means<20))


# In[ ]:


storing_list=[]
for i in range(position[0].shape[0]-1):
    if (position[0][i+1]-position[0][i]>1):
        img6=img5[:,position[0][i]:position[0][i+1]]
        img6_pil=Image.fromarray(np.uint8(img6))
        img6_pil_resize = img6_pil.resize((28,28))
        storing_list.append(img6_pil_resize)


# In[ ]:


nine=storing_list[0]
# nine=nine.save('nine.jpg')


# In[ ]:


zero=storing_list[1]
# zero=zero.save('zero.jpg')


# In[ ]:


j=storing_list[2]
# j=j.save('j.jpg')


# In[ ]:


a=storing_list[3]
# a=a.save('a.jpg')


# In[ ]:


l1=storing_list[4]
# l1=l1.save('l1.jpg')


# In[ ]:


l=img5[:,515:590]
l_pil=Image.fromarray(np.uint8(l))
l_pil_resize = l_pil.resize((28,28))
# l_pil=l_pil.save('new_l.jpg')


# In[ ]:


a2=storing_list[5]
# a2=a2.save('a2.jpg')


# In[ ]:


n=storing_list[6]
# n=n.save('n.jpg')


# In[ ]:


g=storing_list[7]
# g=g.save('g.jpg')


# In[ ]:


a3=storing_list[8]
# a3=a3.save('a3.jpg')


# In[ ]:


s1=storing_list[9]
# s1=s1.save('s1.jpg')


# In[ ]:


i=storing_list[10]
# i=i.save('i.jpg')


# In[ ]:


n1=storing_list[11]
# n1=n1.save('n1.jpg')


# In[ ]:


g1=storing_list[12]
# g1=g1.save('g1.jpg')


# In[ ]:


im = Image.open('nine.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num1 = position[1][0]


# In[ ]:


im = Image.open('zero.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num2 = position[1][0]


# In[ ]:


im = Image.open('j2.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num3 = position[1][0]


# In[ ]:


im = Image.open('a.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num4 = position[1][0]


# In[ ]:


im = Image.open('l1.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num5 = position[1][0]


# In[ ]:


im = Image.open('a2.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num6 = position[1][0]


# In[ ]:


im = Image.open('n.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num7 = position[1][0]


# In[ ]:


im = Image.open('g.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num8 = position[1][0]


# In[ ]:


im = Image.open('a3.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num9 = position[1][0]


# In[ ]:


im = Image.open('s.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num10 = position[1][0]


# In[ ]:


im = Image.open('i.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num11 = position[1][0]


# In[ ]:


im = Image.open('n1.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num12 = position[1][0]


# In[ ]:


im = Image.open('g1.jpg')
plt.imshow(im)
plt.show()
im=np.asarray(im)
im= im.reshape(-1, WIDTH, HEIGHT, 1)
y_pred = model.predict(im)
position=np.where(y_pred)
num13 = position[1][0]


# In[ ]:


# print("{},{}".format(label_dictionary[num13],label_dictionary[num12]))
address1=label_dictionary[9]+""+label_dictionary[0]
address2=label_dictionary[num3]+""+label_dictionary[num4]+""+label_dictionary[num5]+""+label_dictionary[num6]+""+label_dictionary[num7]
address3=f"{label_dictionary[num8]}{label_dictionary[num9]}{label_dictionary[num10]}{label_dictionary[num11]}{label_dictionary[num12]}{label_dictionary[num13]}"


# In[ ]:


model1 = malaya.spell.spylls()


# In[ ]:


address2=model1.correct_text('JALAN')
address2


# In[ ]:


address3=model1.correct_text('GA5ING')
address3


# In[ ]:


full_address=f"{address1} {address2} {address3}"
full_address


# In[ ]:




