import cv2
from exercices_types import TypeOfExercise
import numpy as np
from utils import *
import mediapipe as mp
from Body_part_angle import BodyPartAngle
from PoseEstimation import poseDetector
from tensorflow.keras.models import load_model
from collections import deque

LRCN_model = load_model("model/LRCN_model.h5")

detector = poseDetector()
config = Config()

class exercise_Recognition():
    def __init__(self, sequence_length=config.SEQUENCE_LENGTH , image_height = config.IMAGE_HEIGHT, image_width = config.IMAGE_WIDTH, classes=config.CLASSES_LIST, model=LRCN_model):

        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.classes = classes
        self.model = model

    def predict_on_video(self,frames_queue):

        predicted_labels_probabilities = self.model.predict(np.expand_dims(frames_queue, axis = 0))[0]
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = self.classes[predicted_label]

        return predicted_class_name


    def write_predictions(self,video_reader,output_file_path, frame, predicted_class_name):

        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

        cv2.putText(frame, predicted_class_name, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)

        video_writer.write(frame)
        video_writer.release()



