import carla
import random
import queue
import numpy as np
import os
import logging
import utils

'''
Data collector for collecting point clouds of 
different setups of lidars.

'''

CLIENT_HOST = '127.0.0.1'
CLIENT_PORT = 2000
TRAFFIC_MANAGER_PORT = 8000

class DataCollector:
    def __init__(self, collector_config, logging_path='log'):
        self.collector_config = collector_config
        self.map = collector_config['MAP']
        self.hero_vehicle = collector_config['HERO_VEHICLE']
        self.num_of_env_vehicles = collector_config['NUM_OF_ENV_VEHICLES']
        self.data_save_path = collector_config['DATA_SAVE_PATH']

        self.lidar_group_list = collector_config['LIDAR_GROUP_LIST']
        self.lidars = []

        for lidar_group in self.lidar_group_list:
            for lidar in lidar_group['LIDARS']:
                self.lidars.append(lidar)

        self.lidar_data_queue = queue.Queue()
        self.lidar_actors = []

        self.reference_sensor_transform = None

        self.total_timestamp = collector_config['TOTAL_TIMESTAMPS']
        self.save_interval = collector_config.get('SAVE_INTERVAL', None)
        self.interval_index = 0

        self.logger = utils.create_logger()
        

    def set_synchronization_and_traffic_manager(self, world, traffic_manager, delta_seconds=0.05, global_distance=2.0, hybrid_physics_mode=True, synchronous_mode=True):
        # Enables synchronous mode for world
        settings = world.get_settings()
        settings.synchronous_mode = synchronous_mode 
        settings.fixed_delta_seconds = delta_seconds
        world.apply_settings(settings)

        # Set up the TM in synchronous mode
        traffic_manager.set_synchronous_mode(synchronous_mode)
        traffic_manager.set_global_distance_to_leading_vehicle(global_distance)
        traffic_manager.set_hybrid_physics_mode(hybrid_physics_mode)
        traffic_manager.set_respawn_dormant_vehicles(True)
    
    def set_hero_vehicle(self, world, set_autopilot=True):
        hero_bp = world.get_blueprint_library().find(self.hero_vehicle)
        hero_bp.set_attribute('color', '0, 0, 0') # set hero vehicle color to black
        hero_bp.set_attribute('role_name', 'hero')

        transform = random.choice(world.get_map().get_spawn_points())
        hero_vehicle = world.spawn_actor(hero_bp, transform)

        if set_autopilot:
            hero_vehicle.set_autopilot(True, TRAFFIC_MANAGER_PORT)

        self.logger.info('Set hero vehicle Done!')
        return hero_vehicle

    def set_env_vehicles(self, world, client, env_vehicle_id_list):
        vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []

        spawn_points = world.get_map().get_spawn_points()
        
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

        for response in client.apply_batch_sync(batch, True):
            if response.error:
                self.logger.info(response.error)
            else:
                env_vehicle_id_list.append(response.actor_id)
        
        self.logger.info('Set env vehicles Done!')

    def set_lidar_sensors(self, world, client, hero_vehicle, lidar_id_list):
        SpawnActor = carla.command.SpawnActor
        batch = []
        for lidar in self.lidars:
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            for key, value in lidar['SETUP'].items():
                lidar_bp.set_attribute(key, value)

            transform = carla.Transform(carla.Location(x = lidar['TRANSFORM']['x'], y = lidar['TRANSFORM']['y'], z = lidar['TRANSFORM']['z']), carla.Rotation(yaw = lidar['TRANSFORM']['yaw']))
        
            batch.append(SpawnActor(lidar_bp, transform, parent=hero_vehicle))

        for response in client.apply_batch_sync(batch, True):
            if response.error:
                self.logger.info(response.error)
            else:
                lidar_id_list.append(response.actor_id)
                self.lidar_actors.append(world.get_actor(response.actor_id))
        self.logger.info('Set lidars Done!')

    def set_lidar_callback(self):
        for lidar_actor in self.lidar_actors:
            lidar_actor.listen(lambda data: self.lidar_callback(data))

    def set_spectator(self, world, hero_vehicle, z=20, pitch=-90):
        spectator = world.get_spectator()
        hero_transform = hero_vehicle.get_transform()
        spectator.set_transform(carla.Transform(hero_transform.location+carla.Location(z=z), carla.Rotation(yaw=hero_transform.rotation.yaw, pitch=pitch, roll=hero_transform.rotation.roll)))

    def lidar_callback(self, data):
        point_cloud = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        transform = data.transform
        self.lidar_data_queue.put((point_cloud, transform))

    def is_bug_vehicle(self, vehicle):
        if vehicle.bounding_box.extent.x == 0.0 or vehicle.bounding_box.extent.y == 0.0:
            return True
        return False 

    def is_cyclist(self, vehicle):
        if vehicle.attributes['number_of_wheels'] == '2':
            return True
        return False
    
    def filter_vehicles_dont_want(self, world, env_vehicle_id_list):
        self.logger.info(f'num of environment vehicles before filtering: {len(env_vehicle_id_list)}')

        for env_vehicle_id in env_vehicle_id_list:
            env_vehicle = world.get_actor(env_vehicle_id)
            if self.is_bug_vehicle(env_vehicle) or self.is_cyclist(env_vehicle):
                env_vehicle.destroy()
                env_vehicle_id_list.remove(env_vehicle_id)
        
        self.logger.info(f'num of environment vehicles after filtering: {len(env_vehicle_id_list)}')

    def prepare_labels(self, actors, data_type='Car'):
        labels = []
        for actor in actors:
            bb_cords = [0, 0, 0, 1]

            bb_to_actor_matrix = carla.Transform(actor.bounding_box.location).get_matrix()
            actor_to_world_matrix = actor.get_transform().get_matrix()
            world_to_sensor_matrix = self.reference_sensor_transform.get_inverse_matrix()
            bb_to_sensor_matrix = np.dot(world_to_sensor_matrix, np.dot(actor_to_world_matrix, bb_to_actor_matrix))

            bb_to_sensor_cords = np.transpose(np.dot(bb_to_sensor_matrix, np.transpose(bb_cords)))
            bb_to_sensor_cords = bb_to_sensor_cords[:3]
            bb_to_sensor_cords[1] = - bb_to_sensor_cords[1]

            bb_extents = [actor.bounding_box.extent.x * 2, actor.bounding_box.extent.y * 2, actor.bounding_box.extent.z * 2]

            bb_rotation = - np.radians(actor.get_transform().rotation.yaw - self.reference_sensor_transform.rotation.yaw) 
            
            bb_label = list(bb_to_sensor_cords) + list(bb_extents) + [bb_rotation] + [data_type]

            labels.append(bb_label)

        return np.array(labels)

    def save_data(self, data_for_one_group, labels, group, time_stamp):
        data_list = data_for_one_group['data']
        transform_list = data_for_one_group['transform']

        def merge_data_list(data_list, transform_list, reference_transform):
            merged = False
            data_merged = None

            for i in range(len(data_list)):
                data = data_list[i]
                transform = transform_list[i]

                intensity = data[:, -1]
                data[:, -1] = 1
                local_sensor_to_world = transform.get_matrix()
                world_to_reference_sensor = reference_transform.get_inverse_matrix()
                data = np.dot(np.dot(world_to_reference_sensor, local_sensor_to_world), data.T).T
                data[:, 1] = - data[:, 1]
                data[:, -1] = intensity

                if not merged:
                    data_merged = data
                    merged = True
                else:
                    data_merged = np.concatenate((data_merged, data), axis=0)                   
                
            return data_merged
        
        data_merged = merge_data_list(data_list, transform_list, self.reference_sensor_transform)

        save_path = os.path.join(self.data_save_path, group)
        points_save_path = os.path.join(save_path, 'points')
        labels_save_path = os.path.join(save_path, 'labels')
        if not os.path.exists(points_save_path):
            os.makedirs(points_save_path)
        if not os.path.exists(labels_save_path):
            os.makedirs(labels_save_path)
        
        np.save(os.path.join(points_save_path, f'%06d.npy'%(time_stamp)), data_merged)
        np.savetxt(os.path.join(labels_save_path, f'%06d.txt'%(time_stamp)), labels, fmt='%s')
        
        self.logger.info(f'{time_stamp} {group} saved successfully!')

    def destroy_actors(self, client, actor_id_list):
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_id_list])

    def start_collecting(self):
        hero_vehicle_id_list = []
        env_vehicle_id_list = []
        lidar_id_list = []

        client = carla.Client(CLIENT_HOST, CLIENT_PORT)
        client.set_timeout(5.0)

        world = client.load_world(self.map)
        world.unload_map_layer(carla.MapLayer.ParkedVehicles) # remove parked vehicles

        traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)

        self.set_synchronization_and_traffic_manager(world, traffic_manager)
        
        try:
            self.logger.info('------------------------ Start Generating ------------------------')

            hero_vehicle = self.set_hero_vehicle(world)
            hero_vehicle_id_list.append(hero_vehicle.id) 

            self.set_env_vehicles(world, client, env_vehicle_id_list)
            self.filter_vehicles_dont_want(world, env_vehicle_id_list)
        
            self.set_lidar_sensors(world, client, hero_vehicle, lidar_id_list)
            self.set_lidar_callback()

            time_stamp = 0
            while time_stamp < self.total_timestamp:
                world.tick()

                self.set_spectator(world, hero_vehicle, z=5, pitch=-30) # set spectator for visualization
                
                if self.save_interval != None and (self.interval_index + 1) % self.save_interval != 0:
                    self.interval_index += 1
                    continue
                
                self.interval_index = 0
                ############################# Prepare Data ###########################

                data_dict = {lidar_group['NAME']: {'data': [], 'transform': []} for lidar_group in self.lidar_group_list}

                for lidar_group in self.lidar_group_list:
                    for i in range(len(lidar_group['LIDARS'])):
                        data, transform = self.lidar_data_queue.get(block=True, timeout=0.05)
                        data_dict[lidar_group['NAME']]['data'].append(data)
                        data_dict[lidar_group['NAME']]['transform'].append(transform)

                        if lidar_group['LIDARS'][i].get('REFERENCE_LIDAR', False):
                            self.reference_sensor_transform = transform

                ############################ Prepare Labels ###########################

                labels = self.prepare_labels(world.get_actors(env_vehicle_id_list))
                label_hero = self.prepare_labels(world.get_actors(hero_vehicle_id_list), data_type='Hero')
                labels = np.concatenate((labels, label_hero), axis=0)

                ############################## Save Data ##############################

                for lidar_group in self.lidar_group_list:
                    self.save_data(data_dict[lidar_group['NAME']], labels, lidar_group['NAME'], time_stamp)

                time_stamp += 1

        except RuntimeError:
            self.logger.info('Something wrong happened!')
        
        except KeyboardInterrupt:
            self.logger.info('Exit by user!')
        
        finally:
            settings = world.get_settings()
            settings.synchronous_mode = False 
            world.apply_settings(settings)
            self.logger.info('------------------- Destroying actors -----------------')
            self.destroy_actors(client, hero_vehicle_id_list)
            self.destroy_actors(client, env_vehicle_id_list)
            self.destroy_actors(client, lidar_id_list)
            self.logger.info('------------------------ Done ------------------------')

        

