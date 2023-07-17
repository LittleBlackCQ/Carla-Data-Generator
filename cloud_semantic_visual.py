import numpy as np
import open3d as o3d   

dataset = 'lidar'
file_id = f'000004'

LABEL_COLORS = np.array([
        (  0,   0,   0),   # unlabeled     =   0u
        # cityscape
        (128,  64, 128),   # road          =   1u
        (244,  35, 232),   # sidewalk      =   2u
        ( 70,  70,  70),   # building      =   3u
        (102, 102, 156),   # wall          =   4u
        (190, 153, 153),   # fence         =   5u
        (153, 153, 153),   # pole          =   6u
        (250, 170,  30),   # traffic light =   7u
        (220, 220,   0),   # traffic sign  =   8u
        (107, 142,  35),   # vegetation    =   9u
        (152, 251, 152),   # terrain       =  10u
        ( 70, 130, 180),   # sky           =  11u
        (220,  20,  60),   # pedestrian    =  12u
        (255,   0,   0),   # rider         =  13u
        (  0,   0, 142),   # Car           =  14u
        (  0,   0,  70),   # truck         =  15u
        (  0,  60, 100),   # bus           =  16u
        (  0,  80, 100),   # train         =  17u
        (  0,   0, 230),   # motorcycle    =  18u
        (119,  11,  32),   # bicycle       =  19u
        # custom
        (110, 190, 160),   # static        =  20u
        (170, 120,  50),   # dynamic       =  21u
        ( 55,  90,  80),   # other         =  22u
        ( 45,  60, 150),   # water         =  23u
        (157, 234,  50),   # road line     =  24u
        ( 81,   0,  81),   # ground        =  25u
        (150, 100, 100),   # bridge        =  26u
        (230, 150, 140),   # rail track    =  27u
        (180, 165, 180)    # guard rail    =  28u
]) / 255.0

if __name__ == '__main__':

    # load point clouds
    scan_dir = f'data\\{dataset}\\{file_id}.npy'
    scan = np.load(scan_dir)


    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(scan[:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(LABEL_COLORS[scan[:, -1].astype(np.int32)])

    o3d.visualization.draw_geometries([point_cloud])


