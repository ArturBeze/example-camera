#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 12:13:19 2024

@author: artur
"""

#pip install opencv-python
#pip install opencv-contrib-python

import cv2
import time

cap = cv2.VideoCapture(0)
time.sleep(1)

tracker = cv2.TrackerKCF_create()

success , img = cap.read()
bbox = cv2.selectROI("tracking ", img , False)
tracker.init(img,bbox)

def drawBox(img,bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,50,255),3,1)
    cv2.putText(img, "tracking", (75, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

while True:
    time = cv2.getTickCount()
    success , img = cap.read()

    success, bbox = tracker.update(img)

    if success :
        drawBox(img , bbox)

    else:
        cv2.putText(img, "loos", (75, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-time)
    print(fps)
    cv2.putText(img,str(int(fps)),(75,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

    cv2.imshow("tracking",img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break