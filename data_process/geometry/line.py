import numpy as np
from geometry.curve import Curve

class Line(Curve):
    def __init__(self, point_indices, point_data, is_outer):
        assert len(point_indices) == 2, "Line must be defined by two points"
        assert point_data is not None
        super(Line, self).__init__(point_indices, point_data)
        pt0 = self.point_geom[0, :]
        pt1 = self.point_geom[1, :]
        self.type = 'line'
        self.start = pt0
        self.end = pt1
        self.start_idx = point_indices[0]
        self.end_idx = point_indices[1]
        self.is_outer = is_outer

        self.bbox = self.verts_to_bbox(np.vstack([pt0, pt1]))
        self.bottom_left = np.array([self.bbox[0], self.bbox[2]])

        


    