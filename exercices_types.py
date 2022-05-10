from itertools import count
import numpy as np
from Body_part_angle import BodyPartAngle
from utils import *


class TypeOfExercise(BodyPartAngle):
    def __init__(self,landmarks): 
        super().__init__(landmarks)
        
        
    def squat(self, frame): 
        
        left_leg_angle = self.angle_squat(side='left')
        right_leg_angle = self.angle_squat(side='right')
        avg_leg_angle = (left_leg_angle +right_leg_angle)//2 
        score_position_rep = assume_right_pose(left_leg_angle, right_leg_angle)
        valid = right_pose(score_position_rep)

        put_angle(frame,avg_leg_angle)

        return [avg_leg_angle, score_position_rep, valid]

    def estimate_exercice(self, type, frame):
        if type == 'squat':
            data = self.squat(frame)
        elif type == 'push_ups' : 
            counter, avg_arm_angle, score_position_rep,right_pose  = TypeOfExercise(self.landmarks).push_ups(frame)        
        return data
        
    def push_ups(self, counter, status, frame):
        left_arm_angle = self.angle_push_ups(side='left')
        right_arm_angle = self.angle_push_ups(side='right')
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2
        put_angle(frame,avg_arm_angle)

        if status:
            if left_arm_angle < 95:
                counter += 1
                status = False
        else:
            if left_arm_angle > 160:
                status = True
        return counter, avg_arm_angle, score_position_rep,right_pose   