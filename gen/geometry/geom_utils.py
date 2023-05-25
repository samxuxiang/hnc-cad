import math
import numpy as np

def angle_from_vector_to_x(vec):
    assert vec.size == 2
    # We need to find a unit vector
    angle = 0.0

    l = np.linalg.norm(vec)
    uvec = vec/l

    # 2 | 1
    #-------
    # 3 | 4
    if uvec[0] >=0:
        if uvec[1] >= 0:
            # Qadrant 1
            angle = math.asin(uvec[1])
        else:
            # Qadrant 4
            angle = 2.0*math.pi - math.asin(-uvec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(uvec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-uvec[1])
    return angle


def convert_angle_to_1to360_range(angle_rad):
    """
    Converts the given angle in radians into 1-360 degrees range
    """
    angle = math.degrees(angle_rad)
    # Lifted from: https://stackoverflow.com/questions/12234574/calculating-if-an-angle-is-between-two-angles
    angle=(int(angle) % 360) + (angle-math.trunc(angle)) # converts angle to range -360 + 360
    if angle > 0.0:
        return angle
    else:
        return angle + 360.0


def angle_is_between(angle_rad, a_rad, b_rad):
    """
    Checks if angle is in between the range of a and b
    (All angles must be given in radians)
    """
    angle = convert_angle_to_1to360_range(angle_rad)
    a = convert_angle_to_1to360_range(a_rad)
    b = convert_angle_to_1to360_range(b_rad)
    if a < b:
        return a <= angle and angle <= b
    return a <= angle or angle <= b


def quantize_verts(verts, n_bits=8):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2 ** n_bits - 1
    verts_quantize = (verts - min_range) * range_quantize / (max_range - min_range)
    return verts_quantize.astype("int32")


def dequantize_verts(verts, n_bits=8, add_noise=False):
    """Convert quantized vertices to floats."""
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2 ** n_bits - 1
    verts = verts.astype("float32")
    verts = verts * (max_range - min_range) / range_quantize + min_range
    if add_noise:
        verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
    return verts


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center, vert_center


def scale_vertices(vertices):
    """Scale the vertices so that the long diagonal of the bounding box is one."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    extents = vert_max - vert_min
    scale = np.sqrt(np.sum(extents ** 2))
    return vertices / scale, scale


