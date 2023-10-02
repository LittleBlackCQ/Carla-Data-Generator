import numpy as np
from utils import quaternion2matrix
import os
import cv2

if __name__ == '__main__':
    dataset = "cam_front_right"
    file_id = "000008"
    # load image
    image_path = f'data\\{dataset}\\{file_id}.png'
    img = cv2.imread(image_path)

    # load labels
    label_dir = f'data\\label3\\{file_id}.txt'
    with open(label_dir, 'r') as f:
        labels = f.readlines()

    # load calibration
    calibration_dir = f'data\\calibration\\{dataset}'
    intrinsic = np.load(os.path.join(calibration_dir, "intrinsic.npy"))
    extrinsic = np.load(os.path.join(calibration_dir, "extrinsic.npy"))

    def line_to_info(line):
        line = np.array(line.split())
        x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w = line[:10]
        x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w = map(float, [x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w])
        return [x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w]

    def get_bbox(info, ego_info):
        x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w = info
        ego_x, ego_y, ego_z, ego_l, ego_w, ego_h, ego_rot_x, ego_rot_y, ego_rot_z, ego_rot_w = ego_info

        R_ego = quaternion2matrix(np.array([ego_rot_x, ego_rot_y, ego_rot_z, ego_rot_w]))

        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2 , -h /2, -h / 2, -h / 2]

        # bounding box 8 个点对应的x,y,z坐标
        corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # transform the 3d bbox from object coordiante to world coordinate
        R = quaternion2matrix(np.array([rot_x, rot_y, rot_z, rot_w]))

        # corners_3d_ego = np.dot(np.linalg.inv(R_ego), np.dot(R, corners_3d)) + np.array([[x], [y], [z]]) - np.array([[ego_x], [ego_y], [ego_z]])
        corners_3d_world =  np.dot(R, corners_3d) + np.array([[x], [y], [z]])

        # # transform from world coordinate to ego coordinates
        corners_3d_ego = np.dot(np.linalg.inv(R_ego), corners_3d_world - np.array([[ego_x], [ego_y], [ego_z]]))

        # transform from ego coordinates to sensor coordinates
        corners_3d_sensor = np.ones((4, 8))
        corners_3d_sensor[:3, :] = corners_3d_ego
        corners_3d_sensor = np.dot(extrinsic, corners_3d_sensor)

        # project to the image
        corners_y_minus_z_x = np.vstack([corners_3d_sensor[1, :], -corners_3d_sensor[2, :], corners_3d_sensor[0, :]])
        bbox = np.transpose(np.dot(intrinsic, corners_y_minus_z_x))
        bbox[:, 0] = bbox[:, 0] / bbox[:, 2]
        bbox[:, 1] = bbox[:, 1] / bbox[:, 2]
        return bbox
    infos = [line_to_info(line) for line in labels]
    bboxes = [get_bbox(info, infos[-1]) for info in infos]
    camera_bboxes = [bb[:, :2].astype(int) for bb in bboxes if all(bb[:, 2] > 0)]

    def draw_quadrangle(pts, img): 
        if isinstance(pts, list):
            pts = np.array(pts)
        img_poly = cv2.polylines(img, pts, isClosed=True, color=(255, 0, 0), thickness=1)
        cv2.imshow("Polylines", img_poly)

    for camera_bbox in camera_bboxes:
        draw_quadrangle([[camera_bbox[0, :], camera_bbox[1, :], camera_bbox[2, :], camera_bbox[3, :]],
                         [camera_bbox[4, :], camera_bbox[5, :], camera_bbox[6, :], camera_bbox[7, :]]], img)
        draw_quadrangle([[camera_bbox[0, :], camera_bbox[4, :]], [camera_bbox[1, :], camera_bbox[5, :]], [camera_bbox[2, :], camera_bbox[6, :]], [camera_bbox[3, :], camera_bbox[7, :]]], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()