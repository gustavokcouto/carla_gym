import math
from collections import deque

import numpy as np
import carla
import cv2
import pandas as pd

from planner import RoutePlanner
from pid_controller import PIDController
from route_manipulation import downsample_route
from fuzzy_control import controlsteer, controltargetspeed, controlthrottle


def debug(_window, _max, _min, title=''):

    canvas = np.ones((100, 100, 3), dtype=np.uint8)
    w = int(canvas.shape[1] / len(_window))
    h = 99

    for i in range(1, len(_window)):
        y1 = (_max - _window[i-1]) / (_max - _min + 1e-8)
        y2 = (_max - _window[i]) / (_max - _min + 1e-8)

        cv2.line(
                canvas,
                ((i-1) * w, int(y1 * h)),
                ((i) * w, int(y2 * h)),
                (255, 255, 255), 2)

    canvas = np.pad(canvas, ((5, 5), (5, 5), (0, 0)))

    cv2.imshow(title, canvas)
    cv2.waitKey(1)


class AutoPilot():
    def __init__(self, global_plan_gps, global_plan_world_coord):
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40, debug=False)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._command_planner = RoutePlanner(7.5, 25.0, 257)

        self._waypoint_planner.set_route(global_plan_gps, True)

        ds_ids = downsample_route(global_plan_world_coord, 50)
        global_plan_gps = [global_plan_gps[x] for x in ds_ids]

        self._command_planner.set_route(global_plan_gps, True)
        
        window_size = 50
        self.speed_error_window = deque([0 for _ in range(window_size)], maxlen=window_size)
        self.angle_window = deque([0 for _ in range(window_size)], maxlen=window_size)
        self.max_speed_error = 0
        self.min_speed_error = 0
        self.max_angle = 0
        self.min_angle = 0
        self.error_hist = pd.DataFrame(data=[], columns=[
            'target_speed',
            'speed_error',
            'angle',
        ])

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle 

        return angle

    def _get_control(self, target, far_target, position, speed, theta):
        # Steering.
        angle_unnorm = self._get_angle_to(position, theta, target)
        angle = angle_unnorm / 90

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(position, theta, far_target)
        angle_far = angle_far_unnorm / 90

        self.angle_window.append(angle)
        self.max_angle = max(self.max_angle, abs(angle))
        self.min_angle = -abs(self.max_angle)
        debug(self.angle_window, self.max_angle, self.min_angle, 'angle')

        steer_control = controlsteer()
        steer_control.input['angle_target'] = angle
        steer_control.compute()
        steer = steer_control.output['steer']

        speed_target_control = controltargetspeed()
        speed_target_control.input['angle_far_target']= angle_far
        speed_target_control.input['angle_target'] = angle
        speed_target_control.compute()
        target_speed = speed_target_control.output['target_speed']

        self.speed_error_window.append(target_speed - speed)
        self.max_speed_error = max(self.max_speed_error, abs(target_speed - speed))
        self.min_speed_error = -abs(self.max_speed_error)
        debug(self.angle_window, self.max_speed_error, self.min_speed_error, 'speed')
        self.error_hist = self.error_hist.append({'target_speed': target_speed, 'speed_error': target_speed - speed, 'angle': angle}, ignore_index=True)
        self.error_hist.to_csv('error_hist.csv')

        throttle_control = controlthrottle()
        throttle_control.input['desired_speed'] = target_speed
        throttle_control.input['speed'] = speed
        throttle_control.compute()
        throttle = throttle_control.output['throttle']

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        brake = False

        return steer, throttle, brake

    def _get_position(self, observation):
        gps = observation['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def run_step(self, observation):
        position = self._get_position(observation)

        near_node, _ = self._waypoint_planner.run_step(position)
        far_node, _ = self._command_planner.run_step(position)
        print('--------------')
        print(position)
        print(near_node)
        print(far_node)
        print(observation['compass'])
        steer, throttle, brake = self._get_control(near_node, far_node, position, observation['speed'], observation['compass'])

        control = carla.VehicleControl()
        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)

        return control
