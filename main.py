import cv2
from exercices_types import TypeOfExercise
import numpy as np
from utils import *
import mediapipe as mp
from Body_part_angle import BodyPartAngle
from PoseEstimation import poseDetector


def main():
    cap = cv2.VideoCapture('videos/player1/squat0.mp4')
    cap2 = cv2.VideoCapture('videos/player1/push_up1.mp4')
    cap3 = cv2.VideoCapture('videos/player1/pull_ups1.mp4')
    cap4 = cv2.VideoCapture('videos/player1/sit_up1.mp4')

    detector = poseDetector()
    reps=0
    collect=[]
    der=0
    while cap4.isOpened():
        ret, frame = cap4.read()
        if ret :
            frame = cv2.resize(frame,(940,520), interpolation = cv2.INTER_AREA)
            results = detector.findPose(frame)
            try :
                landmarks = results.pose_landmarks.landmark
                angles,scores,valid = TypeOfExercise(landmarks).estimate_exercise('sit_up',frame)

                per = np.interp(angles, (70, 100), (0, 100))

                if per == 100: 
                    if der == 0 : 
                        reps+=0.5
                        der = 1 
                if per == 0 :
                    if der == 1 : 
                        reps+=0.5
                        count=int(reps)
                        print([count,angles,scores,valid])
                        der = 0
            except : 
                pass
            cv2.imshow("frames", frame)

        else :
            print('could not read frame')

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        

if __name__ == '__main__':
    main()