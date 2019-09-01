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
from random import randrange


class ProcessVidInput:

    def process(self, counter, frame, saved_encodings, saved_names):

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #sleepTime =randrange(10);
        #print(f'sleeping {sleepTime}')
        #time.sleep(sleepTime)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        start = time.process_time();
        apiLiveVidFaceBoxes = face_recognition.face_locations(rgb_small_frame, model="hog")
        print("until boxes: ",counter, time.process_time() - start)
        start = time.process_time();
        responseEncodings = [];
        if apiLiveVidFaceBoxes:
            responseEncodings = face_recognition.face_encodings(rgb_small_frame, apiLiveVidFaceBoxes)
            print("until responseEncodings: %s ", time.process_time() - start)

        names = []
        calDistance = 0;
        # loop over the facial embeddings
        # for encoding in apiLiveVidEncodings:
        for encoding in responseEncodings:
            # encoding =  np.array(responseenc, dtype=np.float64)
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(saved_encodings, encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = saved_names[i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # print("until face recognition: %s ", time.process_time() - start)
        start = time.process_time();

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(apiLiveVidFaceBoxes, names):
            # rescale the face coordinates
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            #Perceived length is found as 879 px. My face is about 10 inches. 
            calculated_distance = self.distance_to_camera(10, 870, (right - left))
            print(f'distance calculated {calculated_distance} inches')

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (116, 39, 249), 2)
            y = bottom - 15 if bottom - 15 > 15 else bottom + 15
            cv2.putText(frame, name + " : " + str(calculated_distance), (left, y), cv2.FONT_HERSHEY_DUPLEX,
                        0.75, (116, 39, 249), 2)


        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer
        # if writer is None and args["output"] is not None:
        #	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #	writer = cv2.VideoWriter(args["output"], fourcc, 20,
        #		(frame.shape[1], frame.shape[0]), True)

        # if the writer is not None, write the frame with recognized
        # faces t odisk
        # if writer is not None:
        #	writer.write(frame)

        # check to see if we are supposed to display the output frame to
        # the screen
        # if args["display"] > 0:
        #frameName = "Frame" + str(counter%4)
        #cv2.imshow(frameName, frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        #if key == ord("q"):
        #    break

        #return cv2

    def distance_to_camera(self, known_width, focal_length, perceived_width):
        # compute and return the distance from the maker to the camera
        return int((known_width * focal_length) / perceived_width)