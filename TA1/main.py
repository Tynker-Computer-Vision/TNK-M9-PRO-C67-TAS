import numpy as np
import cv2

confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Read the input video file
video = cv2.VideoCapture("bb2.mp4")

# Define infinite while loop
while True:
    # Read the first frame of the video
    check, image = video.read()

    # Write condition to check the first frame is exist or not
    if (check):
        image = cv2.resize(image, (0, 0), fx=1, fy=1)

        # Display image
        cv2.imshow('Image', image)
        cv2.waitKey(1)
