#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:53:31 2018

@author: lucagaegauf
"""

#import os
from glob import glob
import scipy.misc
import numpy as np

from keras.applications import vgg19
import sklearn.metrics
import scipy.spatial

import matplotlib.pyplot as plt

#%%
data_type = 'train'
dataset = 'edges2handbags'

path = glob('./datasets/%s/%s/*' % (dataset, data_type))

#%%

# I will probably need to add an input_tensor when combining this with the GAN
#vgg = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(256, 256, 3), pooling=None)
vgg = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3), pooling='max')

vgg.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#%%
imgs = []
for i in range(5):
    img = scipy.misc.imread(path[i], mode='RGB').astype(np.float)
    img = img/127.5 - 1
    imgs.append(img)

imgs = np.stack(imgs)

img_embed = vgg.predict(imgs)
#img_embed = img_embed.flatten()
print(img_embed.shape)

#%%
def cos_sim(x, y):
    return sklearn.metrics.pairwise.cosine_similarity(x.reshape(1, -1),
                                                      y.reshape(1, -1)).item()

def euclid(x, y):
    return scipy.spatial.distance.euclidean(x, y)

print(cos_sim(img_embed[1], img_embed[4]))
print(euclid(img_embed[1], img_embed[4]))

#%%
for i in range(20):
    print('cosine similarity of image 1 with image {}: {}'.format(i,
          cos_sim(img_embed[1], img_embed[i])))

#%%
plt.imshow(imgs[0])
#%%
plt.imshow(imgs[1])
#%%
plt.imshow(imgs[2])
#%%
plt.imshow(imgs[3])
#%%
plt.imshow(imgs[4])