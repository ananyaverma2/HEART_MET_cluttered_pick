#!/usr/bin/env python3

import math
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image
import sys


'''
values of the different camera matrix 
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [613.71923828125, 0.0, 314.70098876953125, 0.0, 613.986083984375, 246.9615020751953, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [613.71923828125, 0.0, 314.70098876953125, 0.0, 0.0, 613.986083984375, 246.9615020751953, 0.0, 0.0, 0.0, 1.0, 0.0]
'''
'''
value of P from the rosbag file
P = [[538.12050153   0.         320.19070834   0.        ]
 [  0.         538.81613509 230.98657922   0.        ]
 [  0.           0.           1.           0.        ]]
'''


class pose_estimation():
    def __init__(self) -> None:
        rospy.loginfo("ROS image msg to OpenCV image converter node is ready...")
        self.stop_sub_flag = False
        self.coor_2D = [30,40]
        self.P = None
        rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/camera_info', CameraInfo, self.callback_camerainfo)
        self.cv2_img = None
        rospy.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image_raw", Image, self.callback_pointcloud) 

    def callback_camerainfo(self, msg):
        camera_info_P = np.array(msg.P)
        self.P = np.array(camera_info_P).reshape([3, 4])

    def imgmsg_to_cv2(self,img_msg):
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
    
    def cx(self, P):
        """ Returns x center """
        return self.P[0][2]
    def cy(self, P):
        """ Returns y center """
        return self.P[1][2]
    def fx(self, P):
        """ Returns x focal length """
        return self.P[0][0]
    def fy(self, P):
        """ Returns y focal length """
        return self.P[1][1]

    def projectPixelTo3dRay(self, coor_2D, P, cv2_img):
        """
        :param coor_2D:        rectified pixel coordinates
        :type coor_2D:         (u, v)
        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """
        
        P = np.array([self.P])
        
        x = (coor_2D[0] - self.cx(self.P)) / self.fx(self.P)
        y = (coor_2D[1] - self.cy(self.P)) / self.fy(self.P)
        norm = math.sqrt(x*x + y*y + 1)
        x /= norm
        y /= norm

        print("the 2d points are : ", x,y)

        rospy.loginfo("CV2IMG: %s" % cv2_img)
        rospy.loginfo("CV2IMG_points: %s" % cv2_img[x][y])

        return [x,y]

    def callback_pointcloud(self, msg):
        """
        :msg: sensor_msgs.Image
        :returns: None

        """
        self.cv2_img = self.imgmsg_to_cv2(msg)
        points = self.projectPixelTo3dRay(self.coor_2D, self.P, self.cv2_img)


if __name__ == "__main__":
    rospy.init_node("convert_pose_2Dto3D")
    object_img = pose_estimation()
    rospy.spin()