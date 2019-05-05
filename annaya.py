# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:34:40 2019

@author: Nidhi
"""
import cv2
import numpy as np
from time import sleep
from scipy.ndimage import zoom
from keras.models import load_model
#from keras.optimizers import SGD
import numpy as np


emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

model = load_model('annaya.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier("D:\Anaconda37\pkgs\opencv-4.1.0-py37hce2de41_0\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(48, 48))# flags=cv2.CV_HAAR_FEATURE_MAX)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        prediction = model.predict(cropped_img)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Emotion', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()