#!/usr/bin/env python3

# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge, CvBridgeError # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from sensor_msgs.msg import Image

#import pytorch
import torch
import torchvision

opencv_path = "/home/kvnptl/work/heart_met_competition/heart_met_ws/src/image_node/temp_images/temp_images_0.jpg"

opencv_img = cv2.imread(opencv_path, -1)


# opencv dim Height x Width x Channel
clip = torch.from_numpy(opencv_img)

#convert to dim Channel x Height x Width
clip = clip.permute(2, 0, 1)

print(clip.shape)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

clip = ((clip / 255.) * 2) - 1.

# For inference
model.eval()
x = [clip]
predictions = model(x)


output_ary = predictions[0]['boxes'].detach().numpy()


for i in output_ary:
    opencv_img = cv2.rectangle(opencv_img, (i[0], i[1]), (i[2], i[3]), (255,255,255), 2)

cv2.imshow('Output Img', opencv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(predictions)