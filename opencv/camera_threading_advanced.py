#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:16:47 2024

@author: artur
"""

# Test Coral Edge TPU
# Image recognition with video
# https://github.com/google-coral/examples-camera/tree/master

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""

import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread

class VideoStream:
	def __init__(self, width = 1280, height = 720):
		# Initialize the PiCamera and the camera image stream
		self.cap = cv2.VideoCapture(0)
		self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		# Read first frame from the stream
		(self.ret, self.frame) = self.cap.read()

		# Variable to control when the camera is stopped
		self.stopped = False	

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# Keep looping indefinitely until the thread is stopped
		while True:
			# If the camera is stopped, stop the thread
			if self.stopped:
				# Close camera resources
				self.cap.release()
				return

			# Otherwise, grab the next frame from the stream
			(self.ret, self.frame) = self.cap.read()

	def read(self):
		# Return the most recent frame
		return self.frame

	def stop(self):
		# Indicate that the camera and thread should be stopped
		self.stopped = True



def main():

	# Initialize frame rate calculation
	frame_rate_calc = 1
	freq = cv2.getTickFrequency()

	# used to record the time when we processed last frame 
	prev_frame_time = 0

	# used to record the time at which we processed current frame 
	new_frame_time = 0

	# Initialize video stream
	videostream = VideoStream().start()
	time.sleep(1)

	while videostream.cap.isOpened():
	#while True:

		# Start timer (for calculating frame rate)
		t1 = cv2.getTickCount()

		# Calculating the fps 

		# fps will be number of frame processed in given time frame 
		# since their will be most of time error of 0.001 second 
		# we will be subtracting it to get more accurate result 
		new_frame_time = time.time() 
		fps = 1/(new_frame_time-prev_frame_time) 
		prev_frame_time = new_frame_time 

		# Grab frame from video stream
		cv2_im = videostream.read()

		if not videostream.ret:
			break

		font = cv2.FONT_HERSHEY_SIMPLEX

		#cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
		cv2.putText(cv2_im, f"FPS: {frame_rate_calc:>.2f}", (7, 70), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
		cv2.putText(cv2_im, f"FPS: {fps:>.2f}", (7, 110), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

		# All the results have been drawn on the frame, so it's time to display it.
		cv2.imshow('frame', cv2_im)

		# Calculate framerate
		t2 = cv2.getTickCount()
		time1 = (t2-t1)/freq
		frame_rate_calc= 1/time1

		# Press 'q' to quit
		if cv2.waitKey(1) == ord('q'):
			break

	print("Hello world!")

	# Clean up
	cv2.destroyAllWindows()
	videostream.stop()

if __name__ == "__main__":
	main()
