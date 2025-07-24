"""
CARLA 0.9.15 automated driving data collection using BehaviorAgent.
Saves semantic `.png`, `gps.json`, `steer.json`.
"""

import argparse, os, sys, json, random
import numpy as np
import pygame
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

try:
    import carla
except ImportError:
    raise RuntimeError('Cannot import CARLA.')

# globals
frame = 0
output = ""
latest_sem = None
latest_gps = None
latest_steer = 0.0

# sensor callbacks
def sem_callback(image):
    global latest_sem
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, 2]  # class ID channel
    latest_sem = arr.copy()

def gps_callback(data):
    global latest_gps
    latest_gps = (data.latitude, data.longitude)

# save data for one frame
def save_frame():
    global frame, latest_sem, latest_gps, latest_steer
    sem_path = os.path.join(output, f"{frame:06d}_sem.png")
    pygame.image.save(pygame.surfarray.make_surface(latest_sem.swapaxes(0,1)), sem_path)
    with open(os.path.join(output, f"{frame:06d}_gps.json"), 'w') as fp:
        json.dump({'lat': latest_gps[0], 'lon': latest_gps[1]}, fp)
    with open(os.path.join(output, f"{frame:06d}_steer.json"), 'w') as fp:
        json.dump({'steer': latest_steer}, fp)
    frame += 1

def main():
    global output, latest_steer

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/session_01')
    args = parser.parse_args()
    output = args.output
    os.makedirs(output, exist_ok=True)

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("CARLA BehaviorAgent Data Collector")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp = world.get_blueprint_library()
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle_bp = bp.filter('vehicle.tesla.model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # semantic segmentation camera
    cam_bp = bp.find('sensor.camera.semantic_segmentation')
    cam_bp.set_attribute('image_size_x','800')
    cam_bp.set_attribute('image_size_y','600')
    cam_bp.set_attribute('fov','90')
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
    cam.listen(sem_callback)

    # GPS sensor
    gps_bp = bp.find('sensor.other.gnss')
    gps = world.spawn_actor(gps_bp, carla.Transform(), attach_to=vehicle)
    gps.listen(gps_callback)

    world.tick()
    vehicle.set_autopilot(True)

    clock = pygame.time.Clock()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            latest_steer = vehicle.get_control().steer

            world.tick()

            if latest_sem is not None and latest_gps is not None:
                save_frame()
                surf = pygame.transform.scale(
                    pygame.surfarray.make_surface(latest_sem.swapaxes(0,1)),
                    (640, 480)
                )
                screen.blit(surf, (0,0))
                label = pygame.font.SysFont('Arial',18).render(
                    f'Frame {frame} Steer {latest_steer:.2f}', True, (255,255,255))
                screen.blit(label, (10,10))
                pygame.display.flip()
                screen.fill((0,0,0))

            clock.tick(20)

    except KeyboardInterrupt:
        pass

    finally:
        cam.stop()
        gps.stop()
        vehicle.destroy()
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        pygame.quit()
        print("Data collection finished.")

if __name__ == "__main__":
    main()
