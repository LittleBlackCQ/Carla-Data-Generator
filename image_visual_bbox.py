import numpy as np
from utils import quaternion2matrix
import os
import cv2

if __name__ == '__main__':
    dataset = "cam_front"
    file_id = "000001"
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

    def get_bbox(line):
        line = line.split()
        x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w, lab= line
        x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w = map(float, [x, y, z, l, w, h, rot_x, rot_y, rot_z, rot_w])
        
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2 , -h /2, -h / 2, -h / 2]
        # bounding box 8 个点对应的x,y,z坐标
        corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # transform the 3d bbox from object coordiante to camera_0 coordinate
        R = quaternion2matrix(np.array([rot_x, rot_y, rot_z, rot_w]))

        corners_3d = np.dot(R, corners_3d) + np.array([[x], [y], [z]])

        corners_y_minus_z_x = np.vstack([corners_3d[1, :], -corners_3d[2, :], corners_3d[0, :]])
        bbox = np.transpose(np.dot(intrinsic, corners_y_minus_z_x))
        bbox[:, 0] = bbox[:, 0] / bbox[:, 2]
        bbox[:, 1] = bbox[:, 1] / bbox[:, 2]
        return bbox

    bboxes = [get_bbox(line) for line in labels]
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