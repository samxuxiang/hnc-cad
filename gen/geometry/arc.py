import numpy as np
import math
from geometry.curve import Curve


class Arc(Curve):
    def __init__(self, point_indices, point_data, is_outer):
        assert len(point_indices) == 4, "Arc must be defined by 3 points"
        assert point_data is not None
        super(Arc, self).__init__(point_indices, point_data)
        self.type = 'arc'
        self.is_outer = is_outer
        self.start = self.point_geom[0, :]
        self.mid = self.point_geom[1, :]
        self.center = self.point_geom[2, :]
        self.end = self.point_geom[3, :]

        self.r1 = math.sqrt( (self.start[0] - self.center[0])**2 + (self.start[1] - self.center[1])**2 )
        self.r2 = math.sqrt( (self.end[0] - self.center[0])**2 + (self.end[1] - self.center[1])**2 )
        self.radius = (self.r1+self.r2)/2

        self.start_idx = point_indices[0]
        self.mid_idx = point_indices[1]
        self.center_idx = point_indices[2]
        self.end_idx = point_indices[3]

        self.bbox = self.verts_to_bbox(np.vstack([self.start, self.end, self.mid]))
        self.bottom_left = np.array([self.bbox[0], self.bbox[2]])
        
    

  
