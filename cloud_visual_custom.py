import numpy as np
import seaborn as sns
import mayavi.mlab as mlab   #使用mayavi进行3D点云的可视化

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Hero', 'Cyclist', 'Tram', 'Misc', 'DontCare']
dataset = 'two_lidars_on_light_06'
file_id = f'000090'

if __name__ == '__main__':

  # load point clouds
  scan_dir = f'data\\{dataset}\\points\\{file_id}.npy'
  scan = np.load(scan_dir)

  # load labels
  label_dir = f'data\\{dataset}\\labels\\{file_id}.txt'
  with open(label_dir, 'r') as f:
    labels = f.readlines()

  fig = mlab.figure(bgcolor=(0, 0, 0))
  # draw point cloud
  plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig)

  for line in labels:
    if line == '':
      continue
    line = line.split()
    x,y,z,l,w,h,rot,lab= line
    l, w, h, x, y, z, rot = map(float, [l, w, h, x, y, z, rot])

    distance = np.sqrt(x**2 + y**2 + z**2)
    if distance > 50:
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