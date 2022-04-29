#!/usr/bin/env python3

#!/usr/bin/env python3

# Import the necessary libraries
from tokenize import String
from turtle import pos
from urllib import request
import numpy as np
import sys

from sympy import re
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import String # String is the message type
from cv_bridge import CvBridge, CvBridgeError # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from sensor_msgs.msg import Image
from metrics_refbox_msgs.msg import ObjectDetectionResult, Command

#import pytorch
import torch
import torchvision

# Import the necessary libraries
from tokenize import String
from urllib import request
from sympy import capture, re
import rospy # Python library for ROS
from std_msgs.msg import String # String is the message type
import cv2 # OpenCV library
from metrics_refbox_msgs.msg import ObjectDetectionResult, Command
import rospkg
import os
from datetime import datetime
import sys
import numpy as np
import torch
import torchvision
import math
from sensor_msgs.msg import CameraInfo, Image
import copy
from geometry_msgs.msg import PoseStamped

class grasp_pose_estimation():
    def __init__(self) -> None:
        rospy.loginfo("ROS image msg to OpenCV image converter node is ready...")
        self.image_queue = None
        self.clip_size = 2 #manual number
        self.detected_bounding_boxes = []
        self.detected_object_names = []
        self.centroids_of_detected_objects = []
        self.positions_of_detected_objects = []


        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        rospy.loginfo("Run the .bag files...")
        self.P = None
        self.cv_image = None
        self.image_sub = rospy.Subscriber(
            "/arm_cam3d/rgb/image_raw", Image, self._input_image_cb) 

        rospy.Subscriber('/arm_cam3d/rgb/camera_info', CameraInfo, self.callback_camerainfo)  
        #/arm_cam3d/depth/image_rect_raw  
        #/arm_cam3d/aligned_depth_to_rgb/image_raw
        rospy.Subscriber('/arm_cam3d/depth/image_rect_raw', Image, self._depth_image)
        self.object_pose_publisher = rospy.Publisher('/grasp_pose_estimation/predicted_object_pose', PoseStamped, queue_size = 10)    


    def imgmsg_to_cv2(self,img_msg):
        if img_msg.encoding != "bgr8":
            # rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
            dtype = np.dtype("uint8") # Hardcode to 8 bits...
            dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
            image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
            # If the byt order is different between the message and the system.
            if img_msg.is_bigendian == (sys.byteorder == 'little'):
                image_opencv = image_opencv.byteswap().newbyteorder()
            return image_opencv

    def cv2_to_imgmsg(self,cv_image):
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg

    def imgmsg_to_cv2_depth(self,img_msg):
        if img_msg.encoding != "bgr8":
            # rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
            dtype = np.dtype("uint8") # Hardcode to 8 bits...
            dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
            image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 1), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
            # If the byt order is different between the message and the system.
            if img_msg.is_bigendian == (sys.byteorder == 'little'):
                image_opencv = image_opencv.byteswap().newbyteorder()
            return image_opencv

    
    def centroid(self,bounding_boxes, img):

        for bbox in bounding_boxes:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            length = x2-x1
            breadth = y2-y1
            centroid = [int((breadth/2)+y1), int((length/2)+x1)]
            self.centroids_of_detected_objects.append(centroid)
            cv2.line(img, (x1-10, y1-10), (x1+10, y1+10), (0,255,255), 3)
            cv2.line(img, (centroid[1]-10, centroid[0]-10), (centroid[1]+10, centroid[0]+10), (0,0,255), 3)
            cv2.line(img, (centroid[1]-10, centroid[0]+10), (centroid[1]+10, centroid[0]-10), (0,0,255), 3)
        
        return self.centroids_of_detected_objects

    def object_inference(self):

        rospy.loginfo("Object Inferencing Started...")

        opencv_img = self.image_queue[0]

        print("initial size ", opencv_img.shape)

        # opencv image dimension in Height x Width x Channel
        clip = torch.from_numpy(opencv_img)

        #convert to torch image dimension Channel x Height x Width
        clip = clip.permute(2, 0, 1)

        # print(clip.shape)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        clip = ((clip / 255.) * 2) - 1.

        # For inference
        model.eval()
        x = [clip]
        predictions = model(x)

        #print prediction boxes on input image
        output_bb_ary = predictions[0]['boxes'].detach().numpy()
        output_labels_ary = predictions[0]['labels'].detach().numpy()
        output_scores_ary = predictions[0]['scores'].detach().numpy()

        detected_object_list = []
        detected_object_score = []
        detected_bb_list = []

        # Extract required objects from prediction output
        print("---------------------------")
        print("Name of the objects, Score\n")
        for idx, value in enumerate(output_labels_ary):
            object_name = self.COCO_INSTANCE_CATEGORY_NAMES[value]
            score = output_scores_ary[idx]

            if score > 0.5:
                detected_object_list.append(object_name)
                detected_object_score.append(score)
                detected_bb_list.append(output_bb_ary[idx])

                print("{}, {}".format(object_name, score))


        # Only publish the target object requested by the referee
        for object_idx in range(len(detected_bb_list)):
            # Referee output message publishing
            object_detection_msg = ObjectDetectionResult()
            object_detection_msg.message_type = ObjectDetectionResult.RESULT
            object_detection_msg.result_type = ObjectDetectionResult.BOUNDING_BOX_2D
            object_detection_msg.object_found = True
            object_detection_msg.box2d.min_x = int(detected_bb_list[object_idx][0])
            object_detection_msg.box2d.min_y = int(detected_bb_list[object_idx][1])
            object_detection_msg.box2d.max_x = int(detected_bb_list[object_idx][2])
            object_detection_msg.box2d.max_y = int(detected_bb_list[object_idx][3])

            #convert OpenCV image to ROS image message
            ros_image = self.cv2_to_imgmsg(self.image_queue[0])
            object_detection_msg.image = ros_image

            # calculate centroids
            self.detected_bounding_boxes.append([object_detection_msg.box2d.min_x, object_detection_msg.box2d.min_y, 
                                            object_detection_msg.box2d.max_x, object_detection_msg.box2d.max_y])
            self.detected_object_names.append(detected_object_list[object_idx])

        # ready for next image
        # self.stop_sub_flag = False

        return self.detected_bounding_boxes, self.detected_object_names, detected_bb_list, opencv_img


    def callback_camerainfo(self, msg):
        camera_info_P = np.array(msg.P)
        self.P = np.array(camera_info_P).reshape([3, 4])
        self.binning_x = max(1, msg.binning_x)
        self.binning_y = max(1, msg.binning_y)
        self.raw_roi = copy.copy(msg.roi)


    def projectPixelTo3dRay(self, centroids_of_detected_objects, P, cv_image):
        """
        :param coor_2D:        rectified pixel coordinates
        :type coor_2D:         (u, v)
        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """

        for centroids in centroids_of_detected_objects:

            self.P[0,0] /= self.binning_x
            self.P[1,1] /= self.binning_y
            self.P[0,2] = (self.P[0,2] - self.raw_roi.x_offset) / self.binning_x
            self.P[1,2] = (self.P[1,2] - self.raw_roi.y_offset) / self.binning_y
        
            # centroids are considered as [H*W] on the contrary to opencv where it takes it as [W*H]
            x = (centroids[0] - self.P[0][2]) / self.P[0][0]
            y = (centroids[1] - self.P[1][2]) / self.P[1][1]
            norm = math.sqrt(x*x + y*y + 1)
            print("coordinates without norm : ", x,y, "1")
            x /= norm
            y /= norm
            z = 1.0 / norm

            depth = (cv_image[centroids[0]][centroids[1]]/1000)[0]
            coord_3D = [x, y, z*depth]

            print("#########################################################################")
            print("the depth is : ", depth)
            print("the original coordinates are : ", x, y, z)
            print("the final coordinates are : ", coord_3D)
            print("#########################################################################")
            self.positions_of_detected_objects.append(coord_3D)

        return self.positions_of_detected_objects

    def _depth_image(self, msg):
        self.depth_image = self.imgmsg_to_cv2_depth(msg)
        self.frame_id = msg.header.frame_id


    def _input_image_cb(self, msg):
        """
        :msg: sensor_msgs.Images
        :returns: None
        """

        # convert ros image to opencv image
        print("image receive")
        self.cv_image = self.imgmsg_to_cv2(msg)
        if self.image_queue is None:
            self.image_queue = []
        
        self.image_queue.append(self.cv_image)

        if len(self.image_queue) > self.clip_size:
            #Clip size reached
            rospy.loginfo("Image received..")
            
            # deregister subscriber
            self.image_sub.unregister()

            # call object inference method
            bounding_boxes, object_names, detected_bb_list, opencv_img = self.object_inference() 
            centroids_of_detected_objects = self.centroid(bounding_boxes, opencv_img)
            positions_of_detected_objects = self.projectPixelTo3dRay(centroids_of_detected_objects, self.P, self.depth_image)

            object_pose= PoseStamped()
            
            for i in positions_of_detected_objects:
                #publish object pose
                rospy.loginfo("Publishing object pose: %s" % i)
                object_pose.header.frame_id = self.frame_id
                object_pose.header.stamp = rospy.Time.now()
                object_pose.pose.position.x = i[0]
                object_pose.pose.position.y = i[1]
                object_pose.pose.position.z = i[2]
                object_pose.pose.orientation.w = 1.0
                self.object_pose_publisher.publish(object_pose)
               
            print("=============================================================")
            print("The extracted bounding boxes are : ", bounding_boxes)
            print("The extracted objects are : ", object_names)
            print("The extracted centroid of the objects are : ", centroids_of_detected_objects)
            print("The positions of detected objects are : ", positions_of_detected_objects)
            print("=============================================================")

            for i in detected_bb_list:
                opencv_img = cv2.rectangle(opencv_img, (i[0], i[1]), (i[2], i[3]), (255,255,255), 2)

            cv2.imshow('Output Img', opencv_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                

if __name__ == "__main__":
    rospy.init_node("convert_pose_2Dto3D")
    object_img = grasp_pose_estimation()
    rospy.spin()

