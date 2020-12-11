# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:11:57 2020

@author: User
"""
import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl


def controlsteer():
    angle_target = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'angle_target')
    steer = ctrl.Consequent(np.arange(-1, 1, 0.01), 'steer')
    angle_target.automf(7)
    steer.automf(7)
    rule1 = ctrl.Rule(angle_target['dismal'], steer['dismal'])
    rule2 = ctrl.Rule(angle_target['poor'], steer['poor'])
    rule3 = ctrl.Rule(angle_target['mediocre'], steer['mediocre'])
    rule4 = ctrl.Rule(angle_target['average'], steer['average'])
    rule5 = ctrl.Rule(angle_target['decent'], steer['decent'])
    rule6 = ctrl.Rule(angle_target['good'], steer['good'])
    rule7 = ctrl.Rule(angle_target['excellent'], steer['excellent'])
    angle_target['average'].view()
    steer_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4,rule5,rule6,rule7])
    steer = ctrl.ControlSystemSimulation(steer_ctrl)

    return steer

def controltargetspeed():
    angle_far_target = ctrl.Antecedent(np.arange(-1, 1, 0.1), 'angle_far_target')
    angle_target = ctrl.Antecedent(np.arange(-1, 1, 0.1), 'angle_target')
    target_speed = ctrl.Consequent(np.arange(0,7,0.1),'target_speed')
    angle_far_target.automf(3)
    target_speed.automf(3)
    angle_target.automf(3)

    rule4 = ctrl.Rule(angle_far_target['poor'] & angle_target['poor'], target_speed['good'])
    rule5= ctrl.Rule(angle_far_target['good'] | angle_target['good'], target_speed['poor'])
    rule6= ctrl.Rule(angle_far_target['average'] & angle_target['average'], target_speed['average'])
    rule7= ctrl.Rule(angle_far_target['average'] & angle_target['poor'], target_speed['average'])
    rule8= ctrl.Rule(angle_far_target['poor'] & angle_target['average'], target_speed['average'])
    target_speed_ctrl = ctrl.ControlSystem([rule4,rule5,rule6,rule7,rule8])
    target_speed = ctrl.ControlSystemSimulation(target_speed_ctrl)

    return target_speed


def controlthrottle():

    speed = ctrl.Antecedent(np.arange(0,7,0.1), 'speed')
    desired_speed = ctrl.Antecedent(np.arange(0,7,0.1),'desired_speed')
    throttle = ctrl.Consequent(np.arange(0,0.75,0.05),'throttle')
    
    throttle.automf(3)
    speed.automf(3)
    desired_speed.automf(3)
    
    rule9= ctrl.Rule(speed['poor'] & desired_speed['poor'], throttle['poor'])
    rule10= ctrl.Rule(speed['average'] & desired_speed['poor'], throttle['poor'])
    rule11= ctrl.Rule(speed['good'] & desired_speed['poor'], throttle['poor'])
    rule12= ctrl.Rule(speed['poor'] & desired_speed['average'], throttle['average'])
    rule13= ctrl.Rule(speed['average'] & desired_speed['average'], throttle['average'])
    rule14= ctrl.Rule(speed['good'] & desired_speed['average'], throttle['poor'])
    rule15= ctrl.Rule(speed['poor'] & desired_speed['good'], throttle['good'])
    rule16= ctrl.Rule(speed['average'] & desired_speed['good'], throttle['good'])
    rule17= ctrl.Rule(speed['good'] & desired_speed['good'], throttle['average'])
    
    throttle_ctrl = ctrl.ControlSystem([rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17])
    throttle = ctrl.ControlSystemSimulation(throttle_ctrl)
    
    return throttle
    
    

if __name__ == '__main__': 
    controlthrottle = controlthrottle()
    controltargetspeed = controltargetspeed()
    controlsteer = controlsteer()
    controlthrottle.input['desired_speed'] = 3
    controltargetspeed.input['angle_far_target']= 0.9 
    controltargetspeed.input['angle_target']=0.2
    controlsteer.input['angle_target']=0.3
    controlthrottle.input['speed']=5
    controlthrottle.compute()
    controltargetspeed.compute()
    controlsteer.compute()
    
    print (controlthrottle.output['throttle'])
    
    print (controltargetspeed.output['target_speed'])
    
    print (controlsteer.output['steer'])
