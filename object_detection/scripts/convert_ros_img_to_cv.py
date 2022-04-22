#!/usr/bin/env python3

# Import the necessary libraries
from tokenize import String
from urllib import request

from sympy import capture, re
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import String # String is the message type
import cv2 # OpenCV library
from sensor_msgs.msg import Image
from metrics_refbox_msgs.msg import ObjectDetectionResult, Command
import rospkg
import os
from datetime import datetime
import sys
import numpy as np

#import pytorch
import torch
import torchvision

class object_detection():
    def __init__(self) -> None:
        rospy.loginfo("Object Detection node is ready...")
        self.image_queue = None
        self.clip_size = 5 #manual number
        self.stop_sub_flag = False
        self.cnt = 0
        self.final_bounding_boxes = []

        # COCO dataset labels
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
        
        # #publisher
        # self.output_bb_pub = rospy.Publisher("/metrics_refbox_client/object_detection_result", ObjectDetectionResult, queue_size=10)
        
        # #subscriber
        # self.requested_object = None
        # self.referee_command_sub = rospy.Subscriber("/metrics_refbox/command", Command, self._referee_command_cb)

        # # waiting for referee box to be ready
        # rospy.loginfo("Waiting for referee box ...")

        rospy.loginfo("Run the .bag files...")
        self.image_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", Image, self._input_image_cb)

    def imgmsg_to_cv2(self, img_msg):
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

    def cv2_to_imgmsg(self, cv_image):
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg
        
    def _input_image_cb(self, msg):
        """
        :msg: sensor_msgs.Image
        :returns: None
        """

        if not self.stop_sub_flag:
            
            # convert ros image to opencv image
            cv_image = self.imgmsg_to_cv2(msg)
            if self.image_queue is None:
                self.image_queue = []
            
            self.image_queue.append(cv_image)
            # print("Counter: ", len(self.image_queue))

            if len(self.image_queue) > self.clip_size:
                #Clip size reached
                # print("Clip size reached...")
                rospy.loginfo("Image received..")
                
                self.stop_sub_flag = True

                # pop the first element
                self.image_queue.pop(0)

                # save all images on local drive

                # create folder for incoming images
                # get an instance of RosPack with the default search paths
                rospack = rospkg.RosPack()

                # get the file path for object_detection package
                pkg_path = rospack.get_path('object_detection')
                captured_images_path = pkg_path + "/captured_images/"
                
                if not os.path.exists(captured_images_path):
                    # 'makedirs' creates a directory with it's path, if applicable.
                    os.makedirs(captured_images_path)

                # get date and time                    
                # datetime object containing current date and time
                now = datetime.now()

                # dd/mm/YY H:M:S
                dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

                for i in self.image_queue:
                    # save image path
                    img_path = captured_images_path + 'captured_images_' + dt_string + '_' + str(self.cnt) + '.jpg'
                    
                    # save image to local drive
                    cv2.imwrite(img_path, i)
                    self.cnt+=1

                rospy.loginfo("Input images saved on local drive")

                # call object inference method
                # print("Image queue size: ", len(self.image_queue))

                # waiting for referee box to be ready
                # rospy.loginfo("Waiting for referee box to be ready...")
                # while self.requested_object is None:
                #     pass
                
                # deregister subscriber
                self.image_sub.unregister()

                # call object inference method
                bounding_boxes = self.object_inference()  
                print(bounding_boxes)

                    
    
    def object_inference(self):

        rospy.loginfo("Object Inferencing Started...")
        
        opencv_img = self.image_queue[0]

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

        # print("---------------------------")
        # print("Fast RCNN output: \n",predictions)
        # print("---------------------------")

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

            if score > 0.0:
                detected_object_list.append(object_name)
                detected_object_score.append(score)
                detected_bb_list.append(output_bb_ary[idx])

                print("{}, {}".format(object_name, score))

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

            self.final_bounding_boxes.append([object_detection_msg.box2d.min_x, object_detection_msg.box2d.min_y, object_detection_msg.box2d.max_x, object_detection_msg.box2d.max_y])

            # print("for id {} the bounding box is [ {}, {}, {}, {}]".format(object_idx, object_detection_msg.box2d.min_x, object_detection_msg.box2d.min_y, object_detection_msg.box2d.max_x, object_detection_msg.box2d.max_y))

            # #convert OpenCV image to ROS image message
            # ros_image = self.cv2_to_imgmsg(self.image_queue[0])
            # object_detection_msg.image = ros_image

            # #publish message
            # self.output_bb_pub.publish(object_detection_msg)

            # #draw bounding box on target detected object
            # opencv_img = cv2.rectangle(opencv_img, (int(detected_bb_list[object_idx][0]), 
            #                                         int(detected_bb_list[object_idx][1])), 
            #                                         (int(detected_bb_list[object_idx][2]), 
            #                                         int(detected_bb_list[object_idx][3])), 
            #                                         (255,255,255), 2)

            # display image
            # cv2.imshow('Output Img', opencv_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


        # ready for next image
        self.stop_sub_flag = False
        self.image_queue = []

        return self.final_bounding_boxes

    # def _referee_command_cb(self, msg):
        
    #     # Referee comaand message (example)
    #     '''
    #     task: 1
    #     command: 1
    #     task_config: "{\"Target object\": \"Cup\"}"
    #     uid: "0888bd42-a3dc-4495-9247-69a804a64bee"
    #     '''

    #     # START command from referee
    #     if msg.task == 1 and msg.command == 1:

    #         print("\nStart command received")

    #         # start subscriber for image topic
    #         self.image_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", 
    #                                             Image, 
    #                                             self._input_image_cb)

    #         #extract target object from task_config            
    #         self.requested_object = msg.task_config.split(":")[1].split("\"")[1]
    #         print("\n")
    #         print("Requested object: ", self.requested_object)
    #         print("\n")
        
    #     # STOP command from referee
    #     if msg.command == 2:
    #         self.stop_sub_flag = True
    #         self.image_sub.unregister()
    #         rospy.loginfo("Subscriber stopped")
                

if __name__ == "__main__":
    rospy.init_node("object_detection_node")
    object_detection_obj = object_detection()
    
    rospy.spin()
