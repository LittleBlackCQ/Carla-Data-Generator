import carla
import random
import queue
import numpy as np
import os
import logging
import utils
import fisheye_utils
import cv2

'''
Data collector for collecting point clouds of 
different setups of sensors.

'''

CLIENT_HOST = '127.0.0.1'
CLIENT_PORT = 2000
TRAFFIC_MANAGER_PORT = 8000

class DataCollector:
    def __init__(self, collector_config, logging_path='log'):
        self.logger = utils.create_logger()
        self.collector_config = collector_config
        self.map = collector_config.get('MAP', 'Town04_Opt')
        self.hero_vehicle_name = collector_config.get('HERO_VEHICLE', 'vehicle.tesla.model3')
        self.num_of_env_vehicles = collector_config.get('NUM_OF_ENV_VEHICLES', 0)
        self.data_save_path = collector_config.get('DATA_SAVE_PATH', 'data')
        
        if not os.path.exists(self.data_save_path):
            os.mkdir(self.data_save_path)

        self.sensor_group_list = collector_config['SENSOR_GROUP_LIST']
        
        self.reference_sensor_transform = None

        self.total_timestamp = collector_config.get('TOTAL_TIMESTAMPS', 2000)
        self.save_interval = collector_config.get('SAVE_INTERVAL', None)
        self.start_timestamp = collector_config.get('START_TIMESTAMP', 0)

        self.save_lane = collector_config.get('SAVE_LANE', False)
        self.save_lidar_labels = collector_config.get('SAVE_LIDAR_LABELS', True)

        self.client = None
        self.world = None
        self.sensor_actors = []
        self.sensor_queues = []
        self.hero_vehicle = None
        self.env_vehicles = []
        self.frame = None
        
    def set_synchronization_world(self, synchronous_mode=True, delta_seconds=0.05):
        # Enables synchronous mode for world
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode 
        settings.fixed_delta_seconds = delta_seconds
        self.world.apply_settings(settings)

    def set_synchronization_traffic_manager(self, traffic_manager, global_distance=3, hybrid_physics_mode=True, synchronous_mode=True):
        # Set up the TM in synchronous mode
        traffic_manager.set_synchronous_mode(synchronous_mode)
        traffic_manager.set_global_distance_to_leading_vehicle(global_distance)
        traffic_manager.set_hybrid_physics_mode(hybrid_physics_mode)
        traffic_manager.set_respawn_dormant_vehicles(True)
    
    def set_hero_vehicle(self, set_autopilot=True):
        hero_bp = self.world.get_blueprint_library().find(self.hero_vehicle_name)
        hero_bp.set_attribute('color', '0, 0, 0') # set hero vehicle color to black
        hero_bp.set_attribute('role_name', 'autopilot')

        transform = random.choice(self.world.get_map().get_spawn_points())
        self.hero_vehicle = self.world.spawn_actor(hero_bp, transform)

        if set_autopilot:
            self.hero_vehicle.set_autopilot(True, TRAFFIC_MANAGER_PORT)

        self.logger.info('Set hero vehicle Done!')

    def set_env_vehicles(self):
        vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle*')

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []

        spawn_points = self.world.get_map().get_spawn_points()
        
        self.logger.info(f'Total blueprints: {len(vehicle_blueprints)}')
        self.logger.info(f'Total spawn points: {len(spawn_points)}')
        for i, transform in enumerate(spawn_points):
            if i >= self.num_of_env_vehicles:
                break
            
            vehicle_bp = random.choice(vehicle_blueprints)

            # has no idea about the meaning of driver_id
            if vehicle_bp.has_attribute('driver_id'):
                driver_id = random.choice(vehicle_bp.get_attribute('driver_id').recommended_values)
                vehicle_bp.set_attribute('driver_id', driver_id)

            vehicle_bp.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot all together
            batch.append(SpawnActor(vehicle_bp, transform).then(SetAutopilot(FutureActor, True, TRAFFIC_MANAGER_PORT)))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                self.logger.info(response.error)
            else:
                self.env_vehicles.append(self.world.get_actor(response.actor_id))
        
        self.logger.info('Set env vehicles Done!')

    def set_sensors(self):
        SpawnActor = carla.command.SpawnActor
        batch = []
        for sensor_group in self.sensor_group_list:
            path = os.path.join(self.data_save_path, sensor_group["NAME"])
            if not os.path.exists(path):
                os.mkdir(path)
            ################# Special Setup for Fisheye #################
            def set_sensors_fisheye(sensor_group, batch):
                sensor_transform = sensor_group['TRANSFORM']
                location = carla.Location(x = sensor_transform.get('x', 0), y = sensor_transform.get('y', 0), z = sensor_transform.get('z', 0))

                roll = sensor_transform.get('roll', 0)
                pitch = sensor_transform.get('pitch', 0)
                yaw = sensor_transform.get('yaw', 0)

                sensor_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
                for key, value in sensor_group['SETUP'].items():
                    sensor_bp.set_attribute(key, value)
                sensor_bp.set_attribute('sensor_tick', '0.05')

                transformL = carla.Transform(location, carla.Rotation(yaw = yaw - 90, pitch = pitch, roll = roll))
                batch.append(SpawnActor(sensor_bp, transformL, parent=self.hero_vehicle))

                transformR = carla.Transform(location, carla.Rotation(yaw = yaw + 90, pitch = pitch, roll = roll))
                batch.append(SpawnActor(sensor_bp, transformR, parent=self.hero_vehicle))

                transformT = carla.Transform(location, carla.Rotation(yaw = yaw, pitch = pitch + 90, roll = roll))
                batch.append(SpawnActor(sensor_bp, transformT, parent=self.hero_vehicle))

                transformB = carla.Transform(location, carla.Rotation(yaw = yaw, pitch = pitch - 90, roll = roll))
                batch.append(SpawnActor(sensor_bp, transformB, parent=self.hero_vehicle))

                transformF = carla.Transform(location, carla.Rotation(yaw = yaw, pitch = pitch, roll = roll))
                batch.append(SpawnActor(sensor_bp, transformF, parent=self.hero_vehicle))

            if sensor_group["TYPE"] == "camera_fisheye":
                set_sensors_fisheye(sensor_group, batch)
                continue

            for sensor in sensor_group['SENSOR_GROUP']:
                sensor_bp = self.world.get_blueprint_library().find('sensor.%s'%(sensor['TYPE']))
                for key, value in sensor['SETUP'].items():
                    sensor_bp.set_attribute(key, value)
                sensor_bp.set_attribute('sensor_tick', '0.05')

                sensor_transform = sensor['TRANSFORM']
                transform = carla.Transform(carla.Location(x = sensor_transform.get('x', 0), y = sensor_transform.get('y', 0), z = sensor_transform.get('z', 0)), carla.Rotation(roll = sensor_transform.get('roll', 0), pitch = sensor_transform.get('pitch', 0), yaw = sensor_transform.get('yaw', 0)))
            
                batch.append(SpawnActor(sensor_bp, transform, parent=self.hero_vehicle))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                self.logger.info(response.error)
            else:
                self.sensor_actors.append(self.world.get_actor(response.actor_id))
        self.logger.info('Set sensors Done!')

    def set_sensor_queue(self):
        for sensor in self.sensor_actors:
            q = queue.Queue()
            sensor.listen(q.put)
            self.sensor_queues.append(q)
    
    def set_spectator(self, z=20, pitch=-90):
        spectator = self.world.get_spectator()
        hero_transform = self.hero_vehicle.get_transform()
        spectator.set_transform(carla.Transform(hero_transform.location+carla.Location(z=z), carla.Rotation(yaw=hero_transform.rotation.yaw, pitch=pitch, roll=hero_transform.rotation.roll)))

    def is_bug_vehicle(self, vehicle):
        if vehicle.bounding_box.extent.x == 0.0 or vehicle.bounding_box.extent.y == 0.0:
            return True
        return False 

    def is_cyclist(self, vehicle):
        if vehicle.attributes['number_of_wheels'] == '2':
            return True
        return False
    
    def filter_vehicles_dont_want(self):
        self.logger.info(f'num of environment vehicles before filtering: {len(self.env_vehicles)}')

        DestroyActor = carla.command.DestroyActor
        destroyed_list = []
        batch = []

        for env_vehicle in self.env_vehicles:
            if self.is_bug_vehicle(env_vehicle) or self.is_cyclist(env_vehicle):
                batch.append(DestroyActor(env_vehicle))
                destroyed_list.append(env_vehicle)
        
        for env_vehicle in destroyed_list:
            self.env_vehicles.remove(env_vehicle)

        self.client.apply_batch_sync(batch, True)
        self.logger.info(f'num of environment vehicles after filtering: {len(self.env_vehicles)}')

    def _retrieve_data(self, sensor_queue, timeout = 10.0):
        while True:
            data_origin = sensor_queue.get(timeout=timeout)
            if data_origin.frame == self.frame:
                return data_origin

    def prepare_labels(self, actors, reference_sensor_transform, map=None, data_type=None):
        labels = []
        for actor in actors:
            data_type = utils.actor2type(actor) if data_type == None else data_type
            bb_cords = [0, 0, 0, 1]

            bb_to_actor_matrix = carla.Transform(actor.bounding_box.location).get_matrix()
            actor_to_world_matrix = actor.get_transform().get_matrix()
            world_to_sensor_matrix = reference_sensor_transform.get_inverse_matrix()
            bb_to_sensor_matrix = np.dot(world_to_sensor_matrix, np.dot(actor_to_world_matrix, bb_to_actor_matrix))

            bb_to_sensor_cords = np.transpose(np.dot(bb_to_sensor_matrix, np.transpose(bb_cords)))
            bb_to_sensor_cords = bb_to_sensor_cords[:3]
            bb_to_sensor_cords[1] = - bb_to_sensor_cords[1]

            bb_extents = [actor.bounding_box.extent.x * 2, actor.bounding_box.extent.y * 2, actor.bounding_box.extent.z * 2]

            bb_rotation = - np.radians(actor.get_transform().rotation.yaw - reference_sensor_transform.rotation.yaw) 
            
            bb_label = list(bb_to_sensor_cords) + list(bb_extents) + [bb_rotation] + [data_type]

            if map != None and self.save_lane:
                way_point = map.get_waypoint(actor.get_location())
                bb_label += [way_point.lane_id]

                left_lane = way_point.get_left_lane()
                right_lane = way_point.get_right_lane()

                left_lane_id = left_lane.lane_id if (left_lane != None and str(left_lane.lane_type)=="Driving") else None
                right_lane_id = right_lane.lane_id if (right_lane != None and str(right_lane.lane_type)=="Driving") else None

                bb_label += [left_lane_id, right_lane_id]
                
            labels.append(bb_label)

        return np.array(labels)
    
    def merge_lidar_group(self, data_list, reference_transform, semantic=False):
        point_merged = np.array([])
        for data, transform in data_list:
            if semantic:
                point = np.frombuffer(data, dtype=np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), \
                    ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)])).copy() 
                point = np.array([point['x'], point['y'], point['z'], point['CosAngle'], point['ObjTag']]).T
                point = point.astype(np.float32)

            else:
                point = np.frombuffer(data, dtype=np.float32).reshape(-1, 4).copy()
                
            point = utils.transform_points_to_reference(point, transform, reference_transform)
            if len(point_merged) == 0:
                point_merged = point
            else:
                point_merged = np.concatenate((point_merged, point), axis=0)
        return point_merged
            
    def destroy_actors(self):
        self.hero_vehicle.destroy()
        for actor in self.env_vehicles:
            actor.destroy()
        for actor in self.sensor_actors:
            actor.destroy()

    def start_collecting(self):
        
        self.client = carla.Client(CLIENT_HOST, CLIENT_PORT)
        self.client.set_timeout(50.0)

        self.world = self.client.load_world(self.map)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles) # remove parked vehicles

        traffic_manager = self.client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
        
        try:
            self.logger.info('------------------------ Start Generating ------------------------')

            self.set_hero_vehicle()

            self.set_env_vehicles()
            self.filter_vehicles_dont_want()
        
            self.set_sensors()
            self.set_sensor_queue()

            self.set_synchronization_world()
            self.set_synchronization_traffic_manager(traffic_manager)

            time_stamp = self.start_timestamp
            interval_index = 0

            while time_stamp < self.total_timestamp + self.start_timestamp:
                self.frame = self.world.tick()

                self.set_spectator(z=5, pitch=-30) # set spectator for visualization
                
                ############################ Get Data ##################################

                data_total = [self._retrieve_data(q) for q in self.sensor_queues]
                
                ########################## Check Save Interval ########################

                if self.save_interval != None and (interval_index + 1) != self.save_interval:
                    interval_index += 1
                    continue
                
                interval_index = 0

                ############################# Save Data ###########################
                reference_sensor_transform = None
                for sensor_group in self.sensor_group_list:
                    save_path = os.path.join(self.data_save_path, sensor_group["NAME"])
                    ################### lidar group #############
                    if sensor_group["TYPE"] == 'lidar_group':
                        num_lidars = len(sensor_group["SENSOR_GROUP"])
                        data_group = data_total[:num_lidars]
                        data_total = data_total[num_lidars:]

                        if reference_sensor_transform == None:
                            reference_sensor_transform = data_group[0].transform

                        point_group = [[data.raw_data, data.transform] for data in data_group]
                        point_merged = self.merge_lidar_group(point_group, reference_sensor_transform, semantic=sensor_group.get('SEMANTIC', False))
                        
                        np.save(os.path.join(save_path, "%06d.npy"%(time_stamp)), point_merged)

                    elif sensor_group["TYPE"] == 'camera_fisheye':
                        num_cameras = 5
                        data_group = data_total[:num_cameras]
                        data_total = data_total[num_cameras:]
                        
                        PicSize = sensor_group["PicSize"]
                        FishSize = sensor_group["FishSize"]
                        FOV = sensor_group["FOV"]
                        picture_group = [np.reshape(np.frombuffer(data.raw_data, dtype=np.dtype("uint8")), (PicSize, PicSize, 4))[:, :, :3][:, :, ::-1] for data in data_group]

                        fisheye_picture = fisheye_utils.cube2fisheye(picture_group, PicSize, FishSize, FOV)
                        cv2.imwrite(os.path.join(save_path, "%06d.png"%(time_stamp)), fisheye_picture)
                
                ##################### 3D bboxes labels #######################
                if self.save_lidar_labels:
                    save_path = os.path.join(self.data_save_path, "label3")
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    labels = self.prepare_labels(self.env_vehicles, reference_sensor_transform, map=self.world.get_map())
                    label_hero = self.prepare_labels([self.hero_vehicle], reference_sensor_transform, map=self.world.get_map(), data_type='Hero')
                    labels = np.concatenate((labels, label_hero), axis=0) if len(labels) != 0 else label_hero

                    np.savetxt(os.path.join(save_path, "%06d.txt"%(time_stamp)), labels, fmt='%s')

                self.logger.info(f'Time stamp {time_stamp} saved successfully!')
                time_stamp += 1

        except RuntimeError:
            self.logger.info('Something wrong happened!')
        
        except KeyboardInterrupt:
            self.logger.info('Exit by user!')
        
        finally:
            self.set_synchronization_world(synchronous_mode=False)
            self.logger.info('------------------- Destroying actors -----------------')
            self.destroy_actors()
            self.logger.info('------------------------ Done ------------------------')

        

