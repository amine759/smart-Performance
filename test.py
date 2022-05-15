import cv2
from exercices_types import TypeOfExercise
import numpy as np
from utils import *
import mediapipe as mp
from Body_part_angle import BodyPartAngle
from PoseEstimation import poseDetector
from tensorflow.keras.models import load_model
from collections import deque

reconstructed_LRCN = load_model("LRCN_model___Date_Time_2022_05_11__17_20_36___Loss_0.4063866138458252___Accuracy_0.8768472671508789.h5")

cap3 = cv2.VideoCapture('videos/player1/pull_ups1.mp4')
detector = poseDetector()
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
CLASSES_LIST= ["PushUps","PullUps","TennisSwing","Basketball","HighJump","SoccerJuggling"]

def predict_on_video(video_reader, output_file_path):

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    predicted_class_name = ''

    while video_reader.isOpened():

        ok, frame = video_reader.read() 

        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        normalized_frame = resized_frame / 255

        frames_queue.append(normalized_frame)
        
        if len(frames_queue) == SEQUENCE_LENGTH:

            predicted_labels_probabilities = reconstructed_LRCN.predict(np.expand_dims(frames_queue, axis = 0))[0]

            predicted_label = np.argmax(predicted_labels_probabilities)

            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        video_writer.write(frame)
        
    video_reader.release()
    video_writer.release()

output_video_file_path = 'output.mp4'

while cap3.isOpened():
    ret, frame = cap3.read()
    if ret :
        frame = cv2.resize(frame,(940,520), interpolation = cv2.INTER_AREA)
        results = detector.findPose(frame)
        try :
            landmarks = results.pose_landmarks.landmark
            predict_on_video(cap3, output_video_file_path)
        except : 
            pass
        cv2.imshow("frames", frame)

    else :
        print('could not read frame')

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    


