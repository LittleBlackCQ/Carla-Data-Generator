import numpy as np

class DataAnalyser:

    ################################# basic utils ###############################
    def points_in_box(self, points, box):
        points = points[:, :3].copy()

        rot = box[6]
        lidar_to_local = np.array([[np.cos(rot), np.sin(rot),0],
                                   [-np.sin(rot), np.cos(rot),0],
                                   [0, 0, 1]])
        
        points = points - box[:3]
        local_points = np.dot(lidar_to_local, points.T).T 

        inbox_indices = (np.abs(local_points[:, 0]) < box[3]/2) & (np.abs(local_points[:, 1]) < box[4]/2) & (np.abs(local_points[:, 2]) < box[5]/2 - 0.04) # to avoid count the point on ground

        return inbox_indices
    
    def points_in_boxes(self, points, boxes):
        inbox_indices = []
        for box in boxes:
            inbox_indices.append(self.points_in_box(points, box))
        
        return np.array(inbox_indices)
    
    def points_num_in_boxes(self, points, boxes):
        inbox_indices = self.points_in_boxes(points, boxes)
        num_inboxes = np.sum(inbox_indices, axis=1)

        return num_inboxes
    
    def points_in_range(self, points, range):
        '''
        range = [xmin, ymin, xmax, ymax]
        '''
        inrange_indices = (points[:, :3] > range[:3]) & (points[:, :3] < range[3:])
        inrange_indices = np.all(inrange_indices, axis=1)
        return inrange_indices
    
    def boxes_in_range(self, boxes, range):
        inrange_indices = []

        for box in boxes:
            inrange_indices.append(self.box_in_range(box, range))
        
        return inrange_indices

    def box_in_range(self, box, range):
        '''
        range = [xmin, ymin, zmin, xmax, ymax, zmax]
        '''
        # corners = self.box_to_corners_global(box)
        center = list(map(float, box[:3]))
        # inrange_indice = (corners[:, :2] > range[:2]) & (corners[:, :2] < range[3:5])
        inrange_indice = (center[:3] > range[:3]) & (center[:3] < range[3:6])

        inrange_indice = np.all(inrange_indice)
        return inrange_indice
    
    def boxes_front(self, boxes):
        rotation = boxes[:, 6] % (2*np.pi)
        front_indices = (rotation < np.pi/2) | (rotation > 3*np.pi/2)

        return front_indices

    def box_to_corners(self, box):
        extent = box[3:6]
        corners = np.array([[1/2, 1/2, 1/2],
                            [1/2, -1/2, 1/2],
                            [-1/2, -1/2, 1/2],
                            [-1/2, 1/2, 1/2],
                            [1/2, 1/2, -1/2],
                            [1/2, -1/2, -1/2],
                            [-1/2, -1/2, -1/2],
                            [-1/2, 1/2, -1/2]])

        corners = corners * extent
        return corners
    
    def box_to_corners_global(self, box):
        corners = self.box_to_corners(box)
        rot = box[6]
        local_rot = np.array([[np.cos(rot), -np.sin(rot),0],
                              [np.sin(rot), np.cos(rot),0],
                              [0, 0, 1]])
        
        corners = np.dot(local_rot, corners.T).T + box[:3]
        return corners
    
    def iou_3d(self, box1, box2):
        '''
        box = [cx, cy, cz, l, w, z, rot]
        '''
        corners1 = self.box_to_corners_global(box1)
        corners2 = self.box_to_corners_global(box2)

        if not polygon_collision(corners1[0:4, 0:2], corners2[0:4, 0:2]):
            return np.round_(0, decimals=5)

        intersection_points = polygon_intersection(corners1[0:4, 0:2], corners2[0:4, 0:2])
        inter_area = polygon_area(intersection_points)

        

        zmax = np.minimum(box1[2], box2[2])
        zmin = np.maximum(box1[2] - box1[5], box2[2] - box2[5])

        inter_vol = inter_area * np.maximum(0, zmax-zmin)

        box1_vol = box1[3] * box1[4] * box1[5]
        box2_vol = box2[3] * box2[4] * box2[5]

        union_vol = (box1_vol + box2_vol - inter_vol)

        iou = inter_vol / union_vol

        if np.isinf(iou) or np.isnan(iou):
            iou = 0

        return np.round_(iou, decimals=5)
    
    def iou_boxes(self, boxes1, boxes2):
        iou_boxes = []
        for box1 in boxes1:
            iou_box = []
            for box2 in boxes2:
                iou_box.append(self.iou_3d(box1, box2))
            iou_boxes.append(iou_box)

        if len(boxes1) == 0:
            iou_boxes = np.zeros((1, len(boxes2)), dtype=np.bool_)
        return np.array(iou_boxes)

    def prepare_points(self, point_path):
        points = np.load(point_path)
        return points
    
    def prepare_boxes(self, label_path):
        boxes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines[:-1]:
            line_list = line.strip().split(' ')
            boxes.append(line_list[:-4] + [line_list[-3]])


        hero_line_list = lines[-1].strip().split(' ')
        self.hero_lane = int(hero_line_list[-3])
        self.hero_left_lane = int(hero_line_list[-2]) if hero_line_list[-2]!='None' else None
        self.hero_right_lane = int(hero_line_list[-1]) if hero_line_list[-1]!='None' else None
        self.hero_cords = np.array(hero_line_list[:3], dtype=np.float32)
        self.hero_box = np.array(hero_line_list[:7], dtype=np.float32)

        boxes = np.array(boxes, dtype=np.float32)

        return boxes

    def prepare_training_boxes(self, label_path):
        boxes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line_list = line.strip().split(' ')
            boxes.append(line_list[:-1])

        boxes = np.array(boxes, dtype=np.float32)
        return boxes

    def prepare_points_boxes(self, point_path=None, label_path=None, raw_data=False):
        points = self.prepare_points(point_path)
        boxes = self.prepare_boxes(label_path)
        return points, boxes


    def remove_points_in_boxes(self, points, boxes):
        indices = self.points_in_boxes(points, boxes)
        indices = np.any(indices, axis=0)
        points = points[indices==False, :]
        return points



