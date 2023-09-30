## Data Structure for BEVFusion
There are some directories in the `data/`.

### cam_*
In the directory like `data/cam_front/`, there are images collected from the camera sensor. The data format is `.png`, start with a 6 number long integer, such as `000001`, which is the frame when the data is collected. The images, labels with the same frame id are collected at the same time.

### label3
In `data/label3/`, there is the information of samples. Each line represents an object. 

In a line, the attributes separated by space are `x y z l w h rot_x rot_y rot_z rot_w velocity label`.

- `x y z` are the 3D location of the object. The `x y z` coordinates are a little bit weird in Carla, which is shown below.

z
^     x
|   /
| /
|------------->y

- `l w h` are the length of the object on the `x y z` axis respectively.
- `rot_x rot_y rot_z rot_w` are the quaternion of the object.
- `velocity` is the speed of the object.
- `label` is a string representing the class of the object. `Car` represents vehicles. `Pedestrian` represents walkers. `Hero` represents the ego vehicle.

### calibration
In the `data/calibration/`, there are intrinsic and extrinsic of all cameras. Now, all the coordinates are transformed to the `cam_front`, which means the `cam_front`'s location is the origin of the coordinates system, and the x-axis is the direction it faces. 

The extrinsic of a camera can help you transform the coordinates in the system of `cam_front` to that camera. The intrinsic of a camera can help you project the 3D bounding boxes to the image. The extrinsic and the intrinsic are stored in `.npy` as numpy ndarrays. The extrinsic is 4 by 4, and the intrinsic is 3 by 3. 

Detailed instruction of these parameters can be found in `image_visual_bbox.py`.

### cam_semantic

The binary map of drivable area is still under development. You can ignore the data.