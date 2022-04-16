#!/usr/bin/env python3

import sensor_msgs.msg
import math
import numpy as np
from sensor_msgs import point_cloud2
import rospy
from scipy.spatial import cKDTree
from sensor_msgs.msg import CameraInfo, PointCloud2


def cx(P):
    """ Returns x center """
    return P[0,2]
def cy(P):
    """ Returns y center """
    return P[1,2]
def fx(P):
    """ Returns x focal length """
    return P[0,0]
def fy(P):
    """ Returns y focal length """
    return P[1,1]

def projectPixelTo3dRay(coor_2D, P, gen):
        """
        :param coor_2D:        rectified pixel coordinates
        :type coor_2D:         (u, v)
        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """
        x = (coor_2D[0] - cx(P)) / fx(P)
        y = (coor_2D[1] - cy(P)) / fy(P)
        norm = math.sqrt(x*x + y*y + 1)
        x /= norm
        y /= norm

        points = gen[:,0:2]
        tree = cKDTree(points)
        idx = tree.query((x, y))[1]
        resultant_z = gen[idx, 2]
        point_3D = [x, y, resultant_z]
        return point_3D

def callback_pointcloud(data):
    gen = point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=True) 
    for p in gen:
        print(" x : %f  y: %f  z: %f" %(p[0],p[1],p[2]))
    return gen

def callback_camerainfo(data):
    camera_info_P = np.array(data.P)
    P = np.array(P).reshape([3, 4])
    return P

if __name__ == '__main__':
    rospy.init_node('convert_pose_2Dto3D', anonymous=True)
    gen = rospy.Subscriber('/arm_cam3d/depth_registered/points', PointCloud2, callback_pointcloud)
    P = rospy.Subscriber('/arm_cam3d/rgb/camera_info', CameraInfo, callback_camerainfo)
    rospy.sleep(10)
    #rospy.spin()

    # P = [613.71923828125, 0.0, 314.70098876953125, 0.0, 0.0, 613.986083984375, 246.9615020751953, 0.0, 0.0, 0.0, 1.0, 0.0]
    # P = np.array(P).reshape([3, 4])
    coor_2D = [30,40]
    coor_3D = projectPixelTo3dRay(coor_2D, P, gen)
    print(coor_3D)

'''
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [613.71923828125, 0.0, 314.70098876953125, 0.0, 613.986083984375, 246.9615020751953, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [613.71923828125, 0.0, 314.70098876953125, 0.0, 0.0, 613.986083984375, 246.9615020751953, 0.0, 0.0, 0.0, 1.0, 0.0]
'''
