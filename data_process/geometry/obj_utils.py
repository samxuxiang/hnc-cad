import json
import numpy as np
from pathlib import Path
import pdb


def read_wire_obj(obj_path):
    """Read vertices and lines from .obj file defining a wire body."""
    vertex_list = []
    loops = []

    # Read vertice and curves
    with open(obj_path) as obj_file:

        for line in obj_file:
            tokens = line.split()
            if not tokens:
                continue

            line_type = tokens[0]

            if line_type == "v":
                vertex_list.append([float(x) for x in tokens[1:]])

            if line_type == "g":
                pdb.set_trace()


            
       
            # Read meta data
            meta_data = line.strip('# ').strip(' \n').split(' ')
            meta_name = meta_data[0]
            if meta_name == 'Extrude':
                extrude_values= [float(x) for x in meta_data[1:]]
            elif meta_name == 'T_origin':
                t_orig = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'T_xaxis':
                t_x = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'T_yaxis':
                t_y = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'T_zaxis':
                t_z = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'ExtrudeOperation:':
                set_op = meta_data[1]


        vertices = np.array(vertex_list)



        meta_info = {'extrude_value': extrude_values,
                     'set_op': set_op,
                     't_orig': t_orig,
                     't_x': t_x,
                     't_y': t_y,
                     't_z': t_z}

        total_in_outs.append(in_outs)
        
    return np.array(flat_vertices_list, dtype=np.float32), flat_hyperedge, total_in_outs, meta_info


def write_wire_obj(vertices, faces, file_path, transpose=True, scale=1.0):
    """Write vertices and hyperedges to obj."""
    vertex_dimension = vertices.shape[1]
    assert vertex_dimension in (2, 3)
    if transpose and vertex_dimension == 3:
        # Permute 3D vertices where z comes first followed by x and y
        vertices = vertices[:, [1, 2, 0]]
    vertices *= scale
    if faces is not None:
        if len(faces) > 0:
            if min(min(faces)) == 0:
                f_add = 1
            else:
                f_add = 0
    with open(file_path, "w") as f:
        for v in vertices:
            if vertex_dimension == 2:
                f.write("v {} {} {}\n".format(v[0], v[1], 0.0))
            else:
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            line = "l"
            for i in face:
                # Pradeep: always adding 1 to the face index makes sense to me. Not sure why
                # PolyGen does this conditionally (see L95 above)
                # Something to note.
                line += " {}".format(i + 1)
            line += "\n"
            f.write(line)

