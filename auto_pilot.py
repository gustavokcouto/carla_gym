import math

import numpy as np
import carla

from planner import RoutePlanner
from pid_controller import PIDController
from route_manipulation import downsample_route


class AutoPilot():
    def __init__(self, global_plan_gps, global_plan_world_coord):
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._command_planner = RoutePlanner(7.5, 25.0, 257)

        self._waypoint_planner.set_route(global_plan_gps, True)

        ds_ids = downsample_route(global_plan_world_coord, 50)
        global_plan_gps = [global_plan_gps[x] for x in ds_ids]

        self._command_planner.set_route(global_plan_gps, True)


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

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(position, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4 if should_slow else 7.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        brake = False

        return steer, throttle, brake, target_speed

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
        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, position, observation['speed'], observation['compass'])

        control = carla.VehicleControl()
        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)

        return control
