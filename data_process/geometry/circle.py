import numpy as np
from geometry.curve import Curve
import pdb

class Circle(Curve):
    def __init__(self, point_indices, point_data, is_outer):
        assert len(point_indices) == 2, "Circle must be defined by 1 points"
        assert point_data is not None
        super(Circle, self).__init__(point_indices, point_data)
        self.type = 'circle'
        self.center = self.point_geom[0, :]
        self.radius = self.point_geom[1, 0]
        self.center_idx = point_indices[0]
        self.radius_idx = point_indices[1]
        self.is_outer = is_outer

        pt1 = np.array([self.center[0], self.center[1]+self.radius])
        pt2 = np.array([self.center[0], self.center[1]-self.radius])
        pt3 = np.array([self.center[0]+self.radius, self.center[1]])
        pt4 = np.array([self.center[0]-self.radius, self.center[1]])
        self.pt1 = pt1
        self.pt2 = pt2 
        self.pt3 = pt3
        self.pt4 = pt4
        self.bbox = self.verts_to_bbox(np.vstack([pt1, pt2, pt3, pt4]))
        self.bottom_left = np.array([self.bbox[0], self.bbox[2]])


        

   