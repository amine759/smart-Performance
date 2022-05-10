import datetime
import time
from threading import Thread
import cv2
from exercices_types import TypeOfExercise
import numpy as np

import imutils
from utils import *
import mediapipe as mp
from Body_part_angle import BodyPartAngle



class poseDetector():

    def __init__(self, detection_confidence = 0.5, track_confidence= 0.5):

        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.Pose = self.mp_pose.Pose(min_detection_confidence=self.detection_confidence, min_tracking_confidence =self.track_confidence)


    def findPose(self, frame, draw=True):
        
        results = self.Pose.process(frame)

        if results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, self.mp_draw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2), self.mp_draw.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2))  

        return results
