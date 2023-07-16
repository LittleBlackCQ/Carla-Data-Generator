import numpy as np
import seaborn as sns
import open3d as mlab   #使用mayavi进行3D点云的可视化

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Hero', 'Cyclist', 'Tram', 'Misc', 'DontCare']
dataset = 'lidar'
file_id = f'000004'

LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain

    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
    (145, 170, 100), # Terrain
]) / 255.0

LABEL_COLORS = LABEL_COLORS.tolist()
if __name__ == '__main__':

  # load point clouds
  scan_dir = f'data\\{dataset}\\{file_id}.npy'
  scan = np.load(scan_dir)

  # load labels
  label_dir = f'data\\label3\\{file_id}.txt'
  with open(label_dir, 'r') as f:
    labels = f.readlines()

  fig = mlab.figure(bgcolor=(0, 0, 0))
  # draw point cloud
  # color = np.divide(LABEL_COLORS[semantic_label], 255.0)
  # color = np.take(LABEL_COLORS, semantic_label, axis=0)
  plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig)

  for line in labels:
    if line == '':
      continue
    line = line.split()
    x,y,z,l,w,h,rot,lab= line
    l, w, h, x, y, z, rot = map(float, [l, w, h, x, y, z, rot])

    distance = np.sqrt(x**2 + y**2 + z**2)
    if distance > 100:
      continue
    
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2 , -h /2, -h / 2, -h / 2]
    #bounding box 8 个点对应的x,y,z坐标
    corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

    # transform the 3d bbox from object coordiante to camera_0 coordinate
    R = np.array([[np.cos(rot), -np.sin(rot),0],
                  [np.sin(rot), np.cos(rot),0],
                  [0, 0, 1]
                  ])
    corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

    # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
    # corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])


    def draw(p1, p2, front=1):
      mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                  color=colors[names.index(lab) * 2 + front], tube_radius=None, line_width=2, figure=fig)


    # draw the upper 4 horizontal lines
    draw(corners_3d[0], corners_3d[1], 0)  # front = 0 for the front lines
    draw(corners_3d[1], corners_3d[2])
    draw(corners_3d[2], corners_3d[3])
    draw(corners_3d[3], corners_3d[0])

    # draw the lower 4 horizontal lines
    draw(corners_3d[4], corners_3d[5], 0)
    draw(corners_3d[5], corners_3d[6])
    draw(corners_3d[6], corners_3d[7])
    draw(corners_3d[7], corners_3d[4])

    # draw the 4 vertical lines
    draw(corners_3d[4], corners_3d[0], 0)
    draw(corners_3d[5], corners_3d[1], 0)
    draw(corners_3d[6], corners_3d[2])
    draw(corners_3d[7], corners_3d[3])

  mlab.view(azimuth=230, distance=50)
  mlab.show()