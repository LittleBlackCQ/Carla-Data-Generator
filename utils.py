import yaml
import logging 
import carla
import datetime
import os
import numpy as np
import cv2

def config_from_yaml(config_file):
    if config_file == None:
        raise RuntimeError
    with open(config_file, 'r') as f:
        collector_config = yaml.safe_load(f)
    return collector_config

def create_logger(log_dir='log'):
    log_file = 'log_generator_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(log_dir, log_file)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger

def transform_points_to_reference(point, local_transform, reference_transform):
    cords = point[:, :4].copy()
    cords[:, -1] = 1
    local_sensor_to_world = local_transform.get_matrix()
    world_to_reference_sensor = reference_transform.get_inverse_matrix()
    cords = np.dot(np.dot(world_to_reference_sensor, local_sensor_to_world), cords.T).T
    point[:, :3] = cords[:, :3]
    point[:, 1] = - point[:, 1]
    return point

def actor2type(actor):
    CYCLIST = ['vehicle.bh.crossbike',
                'vehicle.diamondback.century',
                'vehicle.harley-davidson.low_rider',
                'vehicle.gazelle.omafiets',
                'vehicle.kawasaki.ninja',
                'vehicle.yamaha.yzf']
    if actor.type_id.startswith('walker'):
        return 'Pedestrian'
    elif actor.bounding_box.extent.x >= 4:
        return 'Truck'
    elif actor.type_id in CYCLIST:
        return 'Cyclist'
    else:
        return 'Car'

def set_sensor_setups(bp, setups, iftick = True, tick="0.05"):
    for key, value in setups.items():
        bp.set_attribute(key, value)
    
    if iftick:
        bp.set_attribute('sensor_tick', tick)

def set_sensor_transform(transform_parameters):
    return carla.Transform(carla.Location(x = transform_parameters.get('x', 0), y = transform_parameters.get('y', 0), z = transform_parameters.get('z', 0)), carla.Rotation(roll = transform_parameters.get('roll', 0), pitch = transform_parameters.get('pitch', 0), yaw = transform_parameters.get('yaw', 0)))

def save_binary_image(image, info, save_path, time_stamp):
    type_name = info["name"]
    type_id = info["id"]

    binary_image = np.zeros(np.shape(image))

    if type(type_id) is list:
        for i in type_id:
            binary_image[image[:, :, 0].astype(np.int32) == i, :] = [255, 255, 255]
    else:
        binary_image[image[:, :, 0].astype(np.int32) == type_id, :] = [255, 255, 255]
    
    save_path = os.path.join(save_path, type_name)
    if not os.path.exists(save_path):
            os.mkdir(save_path)
    cv2.imwrite(os.path.join(save_path, f"{time_stamp}.png"), binary_image)

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def is_obstructing(camera_transform, object_transform, world):
    # Cast a ray from the camera to the vehicle and check for obstructions
    result = world.cast_ray(camera_transform.location, object_transform.location)
    
    return False if len(result) <= 1 else True

def rpy2quaternion(roll, pitch, yaw):
    x=np.sin(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)+np.cos(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
    y=np.sin(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)+np.cos(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
    z=np.cos(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)-np.sin(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
    w=np.cos(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)-np.sin(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
    return [x, y, z, w]

def get_calibration(camera):
    width, height, fov = list(map(float, [camera.attributes['image_size_x'], camera.attributes['image_size_y'], camera.attributes['fov']]))
    calibration = np.identity(3)
    calibration[0, 2] = width / 2.0
    calibration[1, 2] = height / 2.0
    calibration[0, 0] = calibration[1, 1] = width / (2.0 * np.tan(fov * np.pi / 360.0))
    return calibration