import numpy as np
from scipy.spatial.transform import Rotation


class Select_prtcl:
    @staticmethod
    def sphere(posd, cen, rad, box_size):
        return posd[np.linalg.norm(np.minimum(np.fabs(posd-cen), np.fabs(posd-cen)-box_size), axis=1) < rad]

    @staticmethod
    def cube(posd, cen, side, box_size):
        return posd[( np.minimum(np.fabs(posd-cen), np.fabs(posd-cen)-box_size) < side/2 ).all(axis=1)]

class Transform:
    @staticmethod
    def shift_origin_wrap(posd, pos_vec, wrap_beyond_dist, box_size):
        posd_shifted = posd - pos_vec
        posd_shifted[posd_shifted < -1*wrap_beyond_dist] += box_size
        posd_shifted[posd_shifted > wrap_beyond_dist] -= box_size
        return posd_shifted

    @staticmethod
    def rotate(posd, rot_vec):
        axis1 = rot_vec
        axis1 /= np.linalg.norm(axis1)
        
        i = 2
        while True:
            for_cross = np.zeros(3)
            for_cross[i] = 1
            axis2 = np.cross(axis1,for_cross)
            print(axis2)
            if np.any(axis2 > 1e-4):
                break
            else:
                print(i)
                i-=1
        axis2 /= np.linalg.norm(axis2)
            
        axis3 = np.cross(axis1,axis2)
        axis3 /= np.linalg.norm(axis3)

        rot_mat = np.matrix([axis1,axis2,axis3])
        print(rot_mat)
        assert np.isclose(np.linalg.det(rot_mat), 1), np.linalg.det(rot_mat)
        rot = Rotation.from_matrix(rot_mat)
        rot.apply(posd)

def Region(shape='sphere',**kwargs):
    if shape=='sphere':
        return SphereRegion(**kwargs)
    if shape=='cube':
        return CubeRegion(**kwargs)

class SphereRegion:
    def __init__(self, cen, rad, box_size):
        self.cen = cen
        self.rad = rad
        self.box_size = box_size
    
    def selectPrtcl(self, posd, shift_origin=False):
        posd_select = posd[np.linalg.norm(np.minimum(np.fabs(posd-self.cen), np.fabs(posd-self.cen)-self.box_size), axis=1) <= self.rad]
        if not shift_origin:
            return posd_select
        else:
            return Transform.shift_origin_wrap(posd_select, self.cen, self.rad*1.5, self.box_size)
            



class CubeRegion:
    def __init__(self, cen, side, box_size):
        self.cen = cen
        self.side = side
        self.box_size = box_size
    
    def selectPrtcl(self, posd, shift_origin=False):
        posd_select = posd[( np.minimum(np.fabs(posd-cen), np.fabs(posd-cen)-box_size) < side/2 ).all(axis=1)]
        if not shift_origin:
            return posd_select
        else:
            return Transform.shift_origin_wrap(posd_select, self.cen, self.side, self.box_size)




