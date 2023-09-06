# Carla Data Generator
A code repo for data collection in [CARLA](http://carla.org/).

## Features
- Easily generating simulated data in CARLA.
- You can set up all parameters through the config file.
- LiDAR data synchronization. 
- LiDAR point clouds can be merged in one coordinites.
- LiDAR point cloud with semantic information.
- Fisheye image is implemented by jointing and distorting camera images from 5 directions.
- Now, the code repo can only collect fisheye image and Lidar point cloud for the author's research purposes. Since the author is now an undergraduate student, the free time should be abundant enough to overcome his laziness. Further functions are upcoming. If you are interested in this repo, contact little_black at your convenience.

## Instructions
- First, you should download the carla unreal engine. Guarantee the `carla` module is contained in your pip list. You can see the detailed instruction on how to install carla in your local machine through the official website.
- Launch the carla unreal engine.
- Run `main.py --cfg_file "config_file"` or `main.py -c "config_file"`. The repository provides some config files for the user. You can customize the parameters to fit your needs.
- See `tesla3.yaml` for comments about each parameter in the config file.
