import numpy as np
from utils import quaternion2matrix
import os
import cv2

def point_in_map(dataset, map_type, point):
    # dataset = "map_05"
    # file_id = "000000"
    # map_type = "curb"

    # load image
    image_path = f'map_curbs_crosswalks\\{dataset}\\{map_type}.png'
    img = cv2.imread(image_path)
    
    calibration_dir = f'map_curbs_crosswalks\\{dataset}\\calibration'

    # load labels
    label_dir = os.path.join(calibration_dir, "000000.txt")
    with open(label_dir, 'r') as f:
        labels = f.readlines()

    # load calibration
    intrinsic = np.load(os.path.join(calibration_dir, "intrinsic.npy"))
    extrinsic = np.load(os.path.join(calibration_dir, "extrinsic.npy"))

    def line_to_info(line):
        line = np.array(line.split())
        x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w = line[:10]
        x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w = map(float, [x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w])
        return [x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w]

    def get_points(info, ego_info):
        x, y, z = info
        ego_x, ego_y, ego_z, ego_l, ego_w, ego_h, ego_rot_x, ego_rot_y, ego_rot_z, ego_rot_w = ego_info

        R_ego = quaternion2matrix(np.array([ego_rot_x, ego_rot_y, ego_rot_z, ego_rot_w]))

        corners_3d_world = np.array([[x], [y], [z]])

        # # transform from world coordinate to ego coordinates
        corners_3d_ego = np.dot(np.linalg.inv(R_ego), corners_3d_world - np.array([[ego_x], [ego_y], [ego_z]]))

        # transform from ego coordinates to sensor coordinates
        corners_3d_sensor = np.ones((4, 1))
        corners_3d_sensor[:3, :] = corners_3d_ego
        corners_3d_sensor = np.dot(extrinsic, corners_3d_sensor)

        # project to the image
        corners_y_minus_z_x = np.vstack([corners_3d_sensor[1, :], -corners_3d_sensor[2, :], corners_3d_sensor[0, :]])
        point = np.transpose(np.dot(intrinsic, corners_y_minus_z_x))
        point[:, 0] = point[:, 0] / point[:, 2]
        point[:, 1] = point[:, 1] / point[:, 2]
        return point

    ego_info = line_to_info(labels[-1])
    point = get_points(point, ego_info)

    point = np.array(point[0, :2].astype(int)) if point[0, 2] > 0 else None
    return point
    # def draw_quadrangle(pts, img): 
    #     if isinstance(pts, list):
    #         pts = np.array(pts)
    #     img_poly = cv2.polylines(img, pts, isClosed=True, color=(51, 0, 255), thickness=3)
    #     cv2.imshow("image", img_poly)
    
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # cv2.resizeWindow("image", (1280, 720))
    # cv2.imshow('image', img)

    # draw_quadrangle([points[:2, :]], img)

    
    # # for camera_bbox in camera_bboxes:
    # #     draw_quadrangle([[camera_bbox[0, :], camera_bbox[1, :], camera_bbox[2, :], camera_bbox[3, :]],
    # #                      [camera_bbox[4, :], camera_bbox[5, :], camera_bbox[6, :], camera_bbox[7, :]]], img)
    # #     draw_quadrangle([[camera_bbox[0, :], camera_bbox[4, :]], [camera_bbox[1, :], camera_bbox[5, :]], [camera_bbox[2, :], camera_bbox[6, :]], [camera_bbox[3, :], camera_bbox[7, :]]], img)
    # cv2.imwrite("test.png", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    print(point_in_map("map_05", "curb", [100, 100, 3]))