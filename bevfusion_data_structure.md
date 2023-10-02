## Data Structure for BEVFusion
There are some directories in the `data/`.

### cam_*
In the directory like `data/cam_front/`, there are images collected from the camera sensor. The data format is `.png`, start with a 6 number long integer, such as `000001`, which is the frame when the data is collected. The images, labels with the same frame id are collected at the same time.

For detailed setups of all cameras, please refer to `config/bevfusion.yaml`

### label3
In `data/label3/`, there is the information of samples. Each line represents an object. 

In a line, the attributes separated by space are `x y z l w h rot_x rot_y rot_z rot_w velocity instance_id label`.

- `x y z` are the 3D location of the object. The coordinates are transformed into the world coordinates system. The `x y z` coordinates are a little bit weird in Carla, which is the left hand system.

![carla-coordinates](https://github.com/little-black-sjtu/carla-data-generator/blob/bevfusion/pictures/carla-coordinates.png)

- `l w h` are the length of the object on the `x y z` axis respectively.
- `rot_x rot_y rot_z rot_w` are the quaternion of the object.
- `velocity` is the speed of the object. If the scale of velocity is 0, the object can be considered "stop".
- `instance_id` is the identifier for the object, which is unique during a given episode.
- `label` is a string representing the class of the object. `Car` represents vehicles. `Pedestrian` represents walkers. `Hero` represents the ego vehicle.

### calibration
In the `data/calibration/`, there are intrinsic and extrinsic of all cameras. 

The extrinsic of a camera is the transformation from the ego vehicle's coordinates system to the camera's coordinates system. Therefore, if you want to transform any object into the camera's coordinates system, you should first transform it to the ego vehicle's coordinates system. See [image_visual_bbox.py](https://github.com/little-black-sjtu/carla-data-generator/blob/bevfusion/image_visual_bbox.py#L49) for implementation details.

The intrinsic of a camera can help you project the 3D bounding boxes to the image. 

The extrinsic and the intrinsic are stored in `.npy` as numpy ndarrays. You can simply use `numpy.load` to get the transformation matrix. The extrinsic is 4 by 4, and the intrinsic is 3 by 3. 

Detailed instruction of using these parameters can be found in [image_visual_bbox.py](https://github.com/little-black-sjtu/carla-data-generator/blob/bevfusion/image_visual_bbox.py).

### cam_semantic

The binary map of drivable area is still under development. You can ignore the data.
