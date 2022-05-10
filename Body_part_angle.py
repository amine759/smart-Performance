import re
import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
from utils import *

class BodyPartAngle:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def angle_squat(self,side=None):
        if side == "left":
            hip = detection_body_part(self.landmarks,"LEFT_HIP")
            knee = detection_body_part(self.landmarks,"LEFT_KNEE")
            ankle = detection_body_part(self.landmarks,"LEFT_ANKLE")
        elif side == "right":
            hip = detection_body_part(self.landmarks,"RIGHT_HIP")
            knee = detection_body_part(self.landmarks,"RIGHT_KNEE")
            ankle = detection_body_part(self.landmarks,"RIGHT_ANKLE")
        else : 
            print("zyer am3elm hhh")
            return    
        return calculate_angle(hip,knee, ankle)


    def angle_push_ups(self, side=None):
        if side == "left":
            hip = detection_body_part(self.landmarks,"LEFT_SHOULDER")
            knee = detection_body_part(self.landmarks,"LEFT_ELBOW")
            ankle = detection_body_part(self.landmarks,"LEFT_WRIST")
        elif side == "right":
            hip = detection_body_part(self.landmarks,"RIGHT_SHOULDER")
            knee = detection_body_part(self.landmarks,"RIGHT_ELBOW")
            ankle = detection_body_part(self.landmarks,"RIGHT_WRIST")
        else : 
            print("zyer am3elm hhh")
            return    
        return calculate_angle(hip,knee, ankle)  


    def put_angle(self,frame):
        center = self.angle_squat()
        cv2.putText(frame, str(self.angle_squat()),tuple(np.multiply(center, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2, cv2.LINE_AA)        






