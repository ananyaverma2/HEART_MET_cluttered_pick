# Cluttered Picking for METRICS HEART-MET competition

### Overall functionality

![alt text](images/overview.png)

### How to run the code:

To run the code you have to run 2 terminals with the below mentioned commands:

#### Terminal-1
Run object detection code: <br/>
`roslaunch object_detection grasp_pose_estimation.launch`

#### Terminal-2
Publish images from rosbag file: <br/>
`rosbag play <path_to_bag_file/_.bag>`
