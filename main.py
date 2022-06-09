import cv2
from exercices_types import TypeOfExercise
import numpy as np
from utils import *
import mediapipe as mp
from Body_part_angle import BodyPartAngle
from PoseEstimation import poseDetector
from model.model import exercise_Recognition
from collections import deque
from datetime import date
from post import * 

today = date.today()
d1 = today.strftime("%d/%m/%Y")


config = Config()

def main():
    cap2 = cv2.VideoCapture('videos/player1/pull_ups1C.mp4')
    cap3 = cv2.VideoCapture('videos/player1/pushups1C.mp4')

    detector = poseDetector()
    reps=0
    der=0
    model = exercise_Recognition()
    frames_queue = deque(maxlen = config.SEQUENCE_LENGTH)
    counts=[]
    score=[]
    while True :
        ret, frame = cap2.read()
        if not ret :
            print('could not read frame')
            break
        frame = cv2.resize(frame,(940,520), interpolation = cv2.INTER_AREA)

        normalized_frame = preprocess_frames(frame) 
        frames_queue.append(normalized_frame)
        results = detector.findPose(frame)
        try :
            landmarks = results.pose_landmarks.landmark

            if len(frames_queue) == config.SEQUENCE_LENGTH :
                predicted_class = model.predict_on_video(frames_queue)
            
            cv2.putText(frame, str(predicted_class), (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
            #model.write_predictions(frame,cap3,predicted_class, output_video_file_path)
        
            angles,scores,valid = TypeOfExercise(landmarks).estimate_exercise(predicted_class,frame)

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
                    score.append(scores)
                    der = 0
        except :
            pass
        cv2.imshow("frames", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break  

    data = {
        "testDate" :d1 ,
        "evaluation" :max(score),
        "repetitions": count,
        "exercise":predicted_class}
    print(data)
    send_data(data)

if __name__ == '__main__':
    main()