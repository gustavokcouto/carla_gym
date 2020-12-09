import math
import numpy as np
import gym
from gym import spaces
import collections
import queue
import time

import carla

from PIL import Image, ImageDraw
from route_parser import parse_routes_file
from route_manipulation import interpolate_trajectory


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

        if self.type == 'semantic_segmentation':
            return array[:, :, 0]

        return array

    def __del__(self):
        self.camera.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()


class MapCamera(Camera):
    def __init__(self, world, player, size, fov, z, pixels_per_meter, radius):
        super().__init__(
                world, player,
                size, size, fov,
                0, 0, z, -90, 0,
                'semantic_segmentation')

        self.world = world
        self.player = player
        self.pixels_per_meter = pixels_per_meter
        self.size = size
        self.radius = radius

    def get(self):
        image = Image.fromarray(super().get())
        draw = ImageDraw.Draw(image)

        transform = self.player.get_transform()
        pos = transform.location
        theta = np.radians(90 + transform.rotation.yaw)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        for light in self.world.get_actors().filter('*traffic_light*'):
            delta = light.get_transform().location - pos

            target = R.T.dot([delta.x, delta.y])
            target *= self.pixels_per_meter
            target += self.size // 2

            if min(target) < 0 or max(target) >= self.size:
                continue

            x, y = target
            draw.ellipse(
                    (x-self.radius, y-self.radius, x+self.radius, y+self.radius),
                    13 + light.state.real)

        return np.array(image)


class GNSS(object):
    def __init__(self, world, player):
        bp = world.get_blueprint_library().find('sensor.other.gnss')

        gnss_location = carla.Location(0,0,0)
        gnss_rotation = carla.Rotation(0,0,0)
        gnss_transform = carla.Transform(gnss_location,gnss_rotation)

        self.type = type
        self.queue = queue.Queue()

        self.gnss = world.spawn_actor(bp, gnss_transform, attach_to=player, attachment_type=carla.AttachmentType.Rigid)
        self.gnss.listen(self.queue.put)
        self.gnss_data = np.array([0, 0])
        self.initialized = False

    def get(self):
        if not self.initialized:
            raw_data = self.queue.get()
            self.gnss_data = np.array([raw_data.latitude, raw_data.longitude])
            self.initialized = True

        while self.queue.qsize() > 0:
            raw_data = self.queue.get()
            self.gnss_data = np.array([raw_data.latitude, raw_data.longitude])

        return self.gnss_data

    def __del__(self):
        self.gnss.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()


class IMU(object):
    def __init__(self, world, player):
        imu_bp = world.get_blueprint_library().find('sensor.other.imu')

        imu_location = carla.Location(0,0,0)
        imu_rotation = carla.Rotation(0,0,0)
        imu_transform = carla.Transform(imu_location, imu_rotation)

        self.type = type
        self.queue = queue.Queue()

        self.imu = world.spawn_actor(imu_bp, imu_transform, attach_to=player, attachment_type=carla.AttachmentType.Rigid)
        self.imu.listen(self.queue.put)
        self.imu_data = np.array([0, 0])
        self.initialized = False

    def get(self):
        if not self.initialized:
            raw_data = self.queue.get()
            self.imu_data = np.array([raw_data.accelerometer.x,
                          raw_data.accelerometer.y,
                          raw_data.accelerometer.z,
                          raw_data.gyroscope.x,
                          raw_data.gyroscope.y,
                          raw_data.gyroscope.z,
                          raw_data.compass,
                         ], dtype=np.float64)
            self.initialized = True

        while self.queue.qsize() > 0:
            raw_data = self.queue.get()
            self.imu_data = np.array([raw_data.accelerometer.x,
                          raw_data.accelerometer.y,
                          raw_data.accelerometer.z,
                          raw_data.gyroscope.x,
                          raw_data.gyroscope.y,
                          raw_data.gyroscope.z,
                          raw_data.compass,
                         ], dtype=np.float64)

        return self.imu_data

    def __del__(self):
        self.imu.destroy()

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
        self._cameras['topdown'] = MapCamera(self._world, self._player, 512, 5 * 10.0, 100.0, 5.5, 5)

        self._gnss = GNSS(self._world, self._player)
        self._imu = IMU(self._world, self._player)

    def reset(self, start_pose):
        set_sync_mode(self._client, True)

        self._time_start = time.time()
        self._cameras.clear()
        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()
        
        self._spawn_player(start_pose)
        self._setup_sensors()

        ticks = 10
        for _ in range(ticks):
            self.step()

        for x in self._actor_dict['camera']:
            x.get()

        self._time_start = time.time()
        self._tick = 0
        
        return self.step()

    def tick_scenario(self):
        spectator = self._world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                self._player.get_location() + carla.Location(z=50),
                carla.Rotation(pitch=-90)))

    def step(self, control=None):
        if control is not None:
            self._player.apply_control(control)

        self._world.tick()
        self.tick_scenario()
        self._tick += 1

        transform = self._player.get_transform()
        velocity = self._player.get_velocity()

        # Put here for speed (get() busy polls queue).
        obs = {key: val.get() for key, val in self._cameras.items()}
        gps = self._gnss.get()
        compass = self._imu.get()[-1]
        
        obs.update({
            'wall': time.time() - self._time_start,
            'tick': self._tick,
            'x': transform.location.x,
            'y': transform.location.y,
            'gps': gps,
            'compass': compass,
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
