import sensor_msgs.msg
import math
import numpy

def mkmat(rows, cols, L):
    mat = numpy.matrix(L, dtype='float64')
    mat.resize((rows,cols))
    return mat

def cx():
    """ Returns x center """
    return P[0,2]
def cy():
    """ Returns y center """
    return P[1,2]
def fx():
    """ Returns x focal length """
    return P[0,0]
def fy():
    """ Returns y focal length """
    return P[1,1]

def projectPixelTo3dRay(uv):
        """
        :param uv:        rectified pixel coordinates
        :type uv:         (u, v)
        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """
        x = (uv[0] - cx()) / fx()
        y = (uv[1] - cy()) / fy()
        norm = math.sqrt(x*x + y*y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm
        return (x, y, z)
