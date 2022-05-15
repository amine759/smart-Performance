import cv2
from exercices_types import TypeOfExercise
import numpy as np
from utils import *
import mediapipe as mp
from Body_part_angle import BodyPartAngle
from PoseEstimation import poseDetector
from tensorflow.keras.models import load_model
from collections import deque

reconstructed_LRCN = load_model("model/LRCN_model.h5")

cap3 = cv2.VideoCapture('videos/player1/pull_ups1.mp4')
detector = poseDetector()
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
CLASSES_LIST= ["PushUps","PullUps","TennisSwing","Basketball","HighJump","SoccerJuggling"]

class exercise_Recognition():
    def __init__(self, sequence_length=SEQUENCE_LENGTH , image_height = IMAGE_HEIGHT, image_width = IMAGE_WIDTH, classes=CLASSES_LIST, model=reconstructed_LRCN):

        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.classes = classes
        self.model = model

    def predict_on_video(self,video_reader, frame):

        frames_queue = deque(maxlen = self.sequence_length)

        predicted_class_name = ''

        while video_reader.isOpened():

            ok, frame = video_reader.read() 

            if not ok:
                break

            resized_frame = cv2.resize(frame, (self.image_width, self.image_height))
            normalized_frame = resized_frame / 255

            frames_queue.append(normalized_frame)
            
            if len(frames_queue) == self.sequence_length:

                predicted_labels_probabilities = self.model.predict(np.expand_dims(frames_queue, axis = 0))[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = self.classes[predicted_label]

        return predicted_class_name

    def write_predictions(self,video_reader,predicted_class_name,output_file_path):
        
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

        while video_reader.isOpened():

            ok, frame = video_reader.read() 

            if not ok:
                break
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            video_writer.write(frame)    
        video_writer.release()

