# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
import face_recognition 
import argparse
import imutils
import pickle
import time
import cv2
import requests
import json
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
#from processVidInputWithAPI import ProcessVidInput
from processVidInput import ProcessVidInput


def main():

	print("***************************************************")
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--encodings", required=True,
					help="path to serialized db of facial encodings")
	ap.add_argument("-o", "--output", type=str,
					help="path to output video")
	ap.add_argument("-y", "--display", type=int, default=1,
					help="whether or not to display output frame to screen")
	ap.add_argument("-d", "--detection-method", type=str, default="cnn",
					help="face detection model to use: either `hog` or `cnn`")
	args = vars(ap.parse_args())

	# load the known faces and embeddings
	print("[INFO] loading encodings...")
	data = pickle.loads(open(args["encodings"], "rb").read())

	# initialize the video stream and pointer to output video file, then
	# allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	# vs = VideoStream(src=0).start()

	print(sys.version_info)
	vs = cv2.VideoCapture('http://admin:@192.168.2.17:80/media/?action=stream')

	counter=0;
	with ProcessPoolExecutor(max_workers=4) as executor:

		while True:
			frames = []
			for dropFrames in range(16):
				if (dropFrames%4)==0:
					hhyy, frameRead = vs.read(1024)
					frames.append(frameRead)


			futures = [executor.submit(ProcessVidInput().process, counter, frame, data["encodings"], data["names"]) for frame in frames]
			counter=counter+1;
			#key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			#if key == ord("q"):
			#	break

		#for future in as_completed(futures):
		#	cv2 = future.result;
		#	cv2.imshow("Frame", frame);


if __name__ == '__main__':
	main()
