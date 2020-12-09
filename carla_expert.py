import sys
import json

from pathlib import Path

import numpy as np
import tqdm
import carla
import cv2
import pandas as pd

from PIL import Image

from carla_env import CarlaEnv
from auto_pilot import AutoPilot

from route_parser import parse_routes_file
from route_manipulation import interpolate_trajectory
from common import COLOR, CONVERTER

EPISODE_LENGTH = 1000
EPISODES = 10
FRAME_SKIP = 5
SAVE_PATH = Path('./episodes')
DEBUG = True


def collect_episode(env, save_dir):
    save_dir.mkdir()

    (save_dir / 'rgb_left').mkdir()
    (save_dir / 'rgb').mkdir()
    (save_dir / 'rgb_right').mkdir()
    (save_dir / 'map').mkdir()

    env._client.start_recorder(str(save_dir / 'recording.log'))

    route_file = Path('data/route_00.xml')
    trajectory = parse_routes_file(route_file)
    global_plan_gps, global_plan_world_coord = interpolate_trajectory(env._world, trajectory)

    elevate_transform = global_plan_world_coord[0][0]
    elevate_transform.location.z += 0.5

    observations, _, _, _ = env.reset(elevate_transform)

    measurements = list()

    auto_pilot = AutoPilot(global_plan_gps, global_plan_world_coord)

    for step in tqdm.tqdm(range(EPISODE_LENGTH * FRAME_SKIP)):
        control = auto_pilot.run_step(observations)
        
        observations, _, _, _ = env.step(control)
        
        if step % FRAME_SKIP != 0:
            continue

        index = step // FRAME_SKIP
        rgb = observations.pop('rgb')
        rgb_left = observations.pop('rgb_left')
        rgb_right = observations.pop('rgb_right')
        topdown = observations.pop('topdown')

        step_measurements = observations.update({
            'steer': control.steer,
            'throttle': control.throttle,
            'brake': control.brake
        })
        measurements.append(step_measurements)

        if DEBUG:
            cv2.imshow('rgb', cv2.cvtColor(np.hstack((rgb_left, rgb, rgb_right)), cv2.COLOR_BGR2RGB))
            cv2.imshow('topdown', cv2.cvtColor(COLOR[CONVERTER[topdown]], cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        Image.fromarray(rgb_left).save(save_dir / 'rgb_left' / ('%04d.png' % index))
        Image.fromarray(rgb).save(save_dir / 'rgb' / ('%04d.png' % index))
        Image.fromarray(rgb_right).save(save_dir / 'rgb_right' / ('%04d.png' % index))
        Image.fromarray(topdown).save(save_dir / 'map' / ('%04d.png' % index))

    pd.DataFrame(measurements).to_csv(save_dir / 'measurements.csv', index=False)

    env._client.stop_recorder()


def main():
    np.random.seed(1337)

    env = CarlaEnv()
    for _ in range(EPISODES):
        SAVE_PATH.mkdir(exist_ok=True)

        collect_episode(env, SAVE_PATH / ('%03d' % len(list(SAVE_PATH.glob('*')))))


if __name__ == '__main__':
    main()