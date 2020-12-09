import math
import numpy as np
import gym
from gym import spaces
import collections
import queue
import time

import carla


VEHICLE_NAME = 'vehicle.lincoln.mkz2017'

def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


class Camera(object):
    def __init__(self, world, player, w, h, fov, x, y, z, pitch, yaw, type='rgb'):
        bp = world.get_blueprint_library().find('sensor.camera.%s' % type)
        bp.set_attribute('image_size_x', str(w))
        bp.set_attribute('image_size_y', str(h))
        bp.set_attribute('fov', str(fov))

        loc = carla.Location(x=x, y=y, z=z)
        rot = carla.Rotation(pitch=pitch, yaw=yaw)
        transform = carla.Transform(loc, rot)

        self.type = type
        self.queue = queue.Queue()

        self.camera = world.spawn_actor(bp, transform, attach_to=player)
        self.camera.listen(self.queue.put)

    def get(self):
        image = None

        while image is None or self.queue.qsize() > 0:
            image = self.queue.get()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        return array

    def __del__(self):
        self.camera.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()


class CarlaEnv(gym.Env):
    def __init__(self, town='Town01', port=2000):
        super(CarlaEnv, self).__init__()
        self._client = carla.Client('192.168.0.4', port)
        self._client.set_timeout(30.0)

        set_sync_mode(self._client, False)

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()

        self._tick = 0
        self._player = None

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)
        self._cameras = dict()

        self.action_space = spaces.Box(low=-10, high=10,
                                            shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(10,), dtype=np.float32)

    def _spawn_player(self, start_pose):
        vehicle_bp = np.random.choice(self._blueprints.filter(VEHICLE_NAME))
        vehicle_bp.set_attribute('role_name', 'hero')

        self._player = self._world.spawn_actor(vehicle_bp, start_pose)

        self._actor_dict['player'].append(self._player)

    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        self._cameras['rgb'] = Camera(self._world, self._player, 256, 144, 90, 1.2, 0.0, 1.3, 0.0, 0.0)
        self._cameras['rgb_left'] = Camera(self._world, self._player, 256, 144, 90, 1.2, -0.25, 1.3, 0.0, -45.0)
        self._cameras['rgb_right'] = Camera(self._world, self._player, 256, 144, 90, 1.2, 0.25, 1.3, 0.0, 45.0)

    def reset(self):
        set_sync_mode(self._client, True)

        self._time_start = time.time()
        self._cameras.clear()
        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()
        
        self._spawn_player(np.random.choice(self._map.get_spawn_points()))
        self._setup_sensors()

        ticks = 10
        for _ in range(ticks):
            self.step()

        for x in self._actor_dict['camera']:
            x.get()

        self._time_start = time.time()
        self._tick = 0

    def step(self, control=None):
        if control is not None:
            self._player.apply_control(control)

        self._world.tick()
        self._tick += 1

        transform = self._player.get_transform()
        velocity = self._player.get_velocity()

        # Put here for speed (get() busy polls queue).
        obs = {key: val.get() for key, val in self._cameras.items()}
        obs.update({
            'wall': time.time() - self._time_start,
            'tick': self._tick,
            'x': transform.location.x,
            'y': transform.location.y,
            'theta': transform.rotation.yaw,
            'speed': np.linalg.norm([velocity.x, velocity.y, velocity.z]),
            })
        
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info


    def close(self):
        set_sync_mode(self._client, False)
        pass