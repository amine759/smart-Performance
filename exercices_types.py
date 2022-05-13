from itertools import count
import numpy as np
from Body_part_angle import BodyPartAngle
from utils import *


class TypeOfExercise(BodyPartAngle):
    def __init__(self,landmarks): 
        super().__init__(landmarks)

    def estimate_exercise(self,type,frame):

        if type == 'squat':
            left_angle = self.angle_squat(side='left')
            right_angle = self.angle_squat(side='right')
            score_position_rep = assume_right_pose(left_angle,right_angle)  

        if type == 'push_ups' or 'pull_ups':
            left_angle = self.angle_push_ups(side='left')
            right_angle = self.angle_push_ups(side='right')
            y_coordinate_left = detection_body_part(self.landmarks,"LEFT_SHOULDER")[1]
            y_coordinate_right = detection_body_part(self.landmarks,"RIGHT_SHOULDER")[1]
            score_position_rep = assume_right_pose(y_coordinate_left, y_coordinate_right)
        
        if type == 'sit_up':
            avg_angle = self.angle_sit_up()
            y_coordinate_left = detection_body_part(self.landmarks,"LEFT_SHOULDER")[1]
            y_coordinate_right = detection_body_part(self.landmarks,"RIGHT_SHOULDER")[1]
            score_position_rep = assume_right_pose(y_coordinate_left, y_coordinate_right)

        if type!='sit_up':
            avg_angle = (left_angle + right_angle)//2
        
        valid = right_pose(score_position_rep)
        put_angle(frame,avg_angle)

        return round(avg_angle),score_position_rep,valid