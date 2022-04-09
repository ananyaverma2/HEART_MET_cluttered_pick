# HEART_MET_cluttered_pick

### How to run the code:

To run the code you have to run 4 terminals with the below mentioned commands:

#### Terminal-1
Run object detection code:
`roslaunch object_detection object_detection_benchmark.launch`

#### Terminal-2
Publish images from rosbag file:
`rosbag play <path_to_bag_file/_.bag>`

#### Terminal-3
Launch Refbox node:
`roslaunch metrics_refbox metrics_refbox.launch`
This will open a refree box

#### Terminal-4
Launch Ref Client node:
`roslaunch metrics_refbox_client metrics_refbox_client.launch`

Once all the terminals are running, tick configuration, select the target object and press start. 