# HEART_MET_cluttered_pick

### How to run the code:

To run the code you have to run 4 terminals with the below mentioned commands:

#### Terminal-1
Run object detection code: <br/>
`roslaunch object_detection object_detection_benchmark.launch`

#### Terminal-2
Publish images from rosbag file: <br/>
`rosbag play <path_to_bag_file/_.bag>`

#### Terminal-3
Launch Refbox node: <br/>
`roslaunch metrics_refbox metrics_refbox.launch` <br/>
This will open a refree box

#### Terminal-4
Launch Ref Client node: <br/>
`roslaunch metrics_refbox_client metrics_refbox_client.launch`

Once all the terminals are running, tick configuration, select the target object and press start. 

> The modules activity_recognition_ros, metrics_refbox, metrics_refbox_client, metrics_refbox_msgs, rosbag_recorder are taken from [here](https://github.com/HEART-MET)