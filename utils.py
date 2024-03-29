import mediapipe as mp
import pandas as pd
import numpy as np
import cv2

mp_pose = mp.solutions.pose

class Config():
    SEQUENCE_LENGTH = 20
    IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
    CLASSES_LIST= ["PushUps","PullUps","TennisSwing","Basketball","HighJump","SoccerJuggling"]

def preprocess_frames(frame):
    resized_frame = cv2.resize(frame,(Config().IMAGE_HEIGHT,Config().IMAGE_WIDTH))
    return resized_frame / 255

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
    return angle


def detection_body_part(landmarks, body_part_name):
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]

def put_angle(frame,values): 
    cv2.putText(frame, str(values),(50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2, cv2.LINE_AA)

# only after each rep 
def assume_right_pose(left_angle,right_angle):
    bias = abs(left_angle - right_angle)
    if bias < 10 :
        return 1.0
    return 1 - 0.01*abs(bias-15)

def right_pose(score_position_rep):
    if score_position_rep < 0.8 :
        return False
    return True