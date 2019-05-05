# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:09:34 2019

@author: Nidhi
"""
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}


import numpy as np
import scipy.misc as smp
import codecs
with codecs.open('D:\\fer2013\\fer2013.csv', 'r', encoding='utf-8',
                 errors='ignore') as fdata:
    df3 = pd.read_csv(fdata)
    
pixel = df3['pixels']
emotion = df3['emotion'] 
count_a = 0
count_d = 0
count_f = 0
count_h = 0
count_s = 0
count_su = 0
count_n = 0
count = 0
for i in range(0, len(pixel), 1000):
    face = [int(j) for j in pixel[i].split(' ')]
    face = np.asarray(face).reshape(48, 48)
    if emotion[i] == 0:
        count_a += 1
        count = count_a
    elif emotion[i] == 1:
        count_d += 1
        count = count_d
    elif emotion[i] == 2:
        count_f += 1
        count = count_f
    elif emotion[i] == 3:
        count_h += 1
        count = count_h
    elif emotion[i] == 4:
        count_s += 1
        count = count_s
    elif emotion[i] == 5:
        count_su += 1
        count = count_su
    elif emotion[i] == 6:
        count_n += 1
        count = count_n
    new_name = emotion_dict[emotion[i]]+'_'+str(count)+'.png'
    smp.imsave(new_name, face)
