from pathlib import Path

import carla

from route_parser import parse_routes_file
from route_manipulation import interpolate_trajectory


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


if __name__ == "__main__":
    port = 2000
    client = carla.Client('192.168.0.4', port)
    client.set_timeout(30.0)

    set_sync_mode(client, False)

    town_name = 'Town01'
    world = client.load_world(town_name)
    route_file = Path('data/route_00.xml')
    trajectory = parse_routes_file(route_file)
    route_gps, route = interpolate_trajectory(world, trajectory)
    print(route_gps)