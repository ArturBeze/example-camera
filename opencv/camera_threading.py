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

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

class VideoStream:
	def __init__(self, width = 1280, height = 720):
		# Initialize the PiCamera and the camera image stream
		self.cap = cv2.VideoCapture(0)
		self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		self.cap.set(cv2.CAP_PROP_FPS, 60)

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
	#default_model_dir = "../all_models"
	default_model_dir = "all_models"
	default_model = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
	default_labels = "coco_labels.txt"

	parser = argparse.ArgumentParser()

	parser.add_argument('--model', help='.tflite model path', default=os.path.join(default_model_dir, default_model))
	parser.add_argument('--labels', help='label file path', default=os.path.join(default_model_dir, default_labels))
	parser.add_argument('--top_k', type=int, help='number of categories with highest score to display', default = 3)
	parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
	parser.add_argument('--threshold', type=float, help='classifier score threshold', default = 0.1)

	args = parser.parse_args()

	print("Loading {} with {} labels.".format(args.model, args.labels))

	interpreter = make_interpreter(args.model)
	interpreter.allocate_tensors()
	labels = read_label_file(args.labels)
	inference_size = input_size(interpreter)



	# Initialize frame rate calculation
	frame_rate_calc = 1
	freq = cv2.getTickFrequency()

	# used to record the time when we processed last frame 
	prev_frame_time = 0

	# used to record the time at which we processed current frame 
	new_frame_time = 0

	# Initialize video stream
	videostream = VideoStream().start()
	fr = videostream.cap.get(cv2.CAP_PROP_FPS)
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
		start_t1 = time.time()
		cv2_im = videostream.read()
		time_elapsed(start_t1, "camera capture")

		if not videostream.ret:
			break



		font = cv2.FONT_HERSHEY_SIMPLEX

		#cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
		cv2.putText(cv2_im, f"FPS: {frame_rate_calc:>.2f} ({fr})", (7, 70), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
		cv2.putText(cv2_im, f"FPS: {fps:>.2f} ({fr})", (7, 110), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

		cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
		cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
		run_inference(interpreter, cv2_im_rgb.tobytes())
		objs = get_objects(interpreter, args.threshold)[:args.top_k]
		cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)



		# All the results have been drawn on the frame, so it's time to display it.
		cv2.imshow('frame', cv2_im)

		# Press 'q' to quit
		if cv2.waitKey(1) == ord('q'):
			break
		
		# Calculate framerate
		t2 = cv2.getTickCount()
		time1 = (t2-t1)/freq
		frame_rate_calc= 1/time1

	# Clean up
	cv2.destroyAllWindows()
	videostream.stop()



def append_objs_to_img(cv2_im, inference_size, objs, labels):
	#pass

	height, width, channels = cv2_im.shape
	scale_x, scale_y = width / inference_size[0], height / inference_size[1]
	for obj in objs:
		bbox = obj.bbox.scale(scale_x, scale_y)
		x0, y0 = int(bbox.xmin), int(bbox.ymin)
		x1, y1 = int(bbox.xmax), int(bbox.ymax)

		percent = int(100 * obj.score)
		label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

		v2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
		cv2_im = cv2.putText(cv2_im, label, (x0, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

	return cv2_im



def time_elapsed(start_time, event):
	time_now = time.time()
	duration = (time_now - start_time) * 1000
	duration=round(duration, 2)
	print (">>> ", duration, " ms (" ,event, ")")



if __name__ == "__main__":
	main()
