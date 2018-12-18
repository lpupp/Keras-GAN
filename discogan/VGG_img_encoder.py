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
n = 50

#%%
data_type = 'train'
dataset = 'edges2handbags'

path = glob('./datasets/%s/%s/*' % (dataset, data_type))

#%%
def cos_sim(x, y):
    return sklearn.metrics.pairwise.cosine_similarity(x.reshape(1, -1),
                                                      y.reshape(1, -1)).item()

def euclid(x, y):
    return scipy.spatial.distance.euclidean(x, y)

#%%
# I will probably need to add an input_tensor when combining this with the GAN
#vgg = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(256, 256, 3), pooling=None)
vgg = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3), pooling='max')

vgg.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#%%
imgs = []
for i in range(n):
    img = scipy.misc.imread(path[i], mode='RGB').astype(np.float)
    img = img/127.5 - 1
    imgs.append(img)

imgs = np.stack(imgs)

img_embed = vgg.predict(imgs)
#img_embed = img_embed.flatten()
print(img_embed.shape)

#%%
cossim = {}
amax = {}
for target in range(n):
    cossim.setdefault(target, [])
    for i in range(n):
        cossim[target].append(cos_sim(img_embed[target], img_embed[i]))
    
    top2 = np.array(cossim[target]).argsort()[-2:][::-1]
    top1 = top2[top2 != target]
    amax[target] = top1.item()

#%%
i = 14
k, v = list(amax.keys()), list(amax.values())
plt.imshow(np.concatenate((imgs[k[i]], imgs[v[i]]), axis=1))

#%%
# https://convertio.co/jp2-jpg/
vggz = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(1100, 762, 3), pooling='max')
# Zalando
pathz = glob('./datasets/zalando_test/*')

imgsz = []
for i in pathz:
    img = scipy.misc.imread(i, mode='RGB').astype(np.float)
    img = img/127.5 - 1
    imgsz.append(img)

imgsz = np.stack(imgsz)

img_embedz = vggz.predict(imgsz)
#img_embed = img_embed.flatten()
print(img_embedz.shape)

#%%
# 1, 12, 18, 21
# 26 fails
target = 26
plt.imshow(imgs[target])
cs = []
for i in range(9):
    cs.append(cos_sim(img_embed[target], img_embedz[i]))

#%%
plt.imshow(imgsz[np.argmax(np.array(cs))])