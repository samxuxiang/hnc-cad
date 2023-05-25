import os 
import numpy as np
from pathlib import Path 
from geometry.obj_parser import OBJParser
import meshio

EXTRUDE_R = 1
SKETCH_R = 1
BBOX_RANGE = 1
CUBOID_RANGE = 1


def quantize(data, n_bits=8, min_range=-1.0, max_range=1.0):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize) # clip values
    return data_quantize.astype('int32')


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


def find_files_path(folder, extension):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extension)])


def get_loop_bbox(loop):
    bbox = []
    for curve in loop:
        bbox.append(curve.bbox)
    bbox_np = np.vstack(bbox)
    bbox_tight = [np.min(bbox_np[:,0]), np.max(bbox_np[:,1]), np.min(bbox_np[:,2]), np.max(bbox_np[:,3])]
    return np.asarray(bbox_tight) 


def get_face_bbox(face):
    bbox = []
    for loop in face:
        bbox.append(get_loop_bbox(loop))
    bbox_np = np.vstack(bbox)
    bbox_tight = [np.min(bbox_np[:,0]), np.max(bbox_np[:,1]), np.min(bbox_np[:,2]), np.max(bbox_np[:,3])]
    return np.asarray(bbox_tight) 


def sort_faces(sketch):
    bbox_list = []
    for face in sketch:
        bbox_list.append(get_face_bbox(face))
    bbox_list = np.vstack(bbox_list)
    min_values = bbox_list[:, [0,2]]
    # Sort by X min, then by Y min (increasing)
    ind = np.lexsort(np.transpose(min_values)[::-1])
    sorted_sketch = [sketch[x] for x in ind]
    return sorted_sketch


def sort_loops(face):
    assert face[0][0].is_outer
    if len(face) == 1:
        return face # only one outer loop, no need to sort 

    # Multiple loops in a face
    bbox_list = []
    inner_faces = face[1:]
    for loop_idx, loop in enumerate(inner_faces):
        assert not loop[0].is_outer # all loops are inner
        bbox_list.append(get_loop_bbox(loop))
    bbox_list = np.vstack(bbox_list)
    min_values = bbox_list[:, [0,2]]
    # Sort by X min, then by Y min (increasing)
    ind = np.lexsort(np.transpose(min_values)[::-1])
    
    # Outer + sorted inner 
    sorted_face = [face[0]] + [inner_faces[x] for x in ind]
    return sorted_face


def curve_connection(loop):
    adjM = np.zeros((len(loop), 2)) # 500 should be large enough
    for idx, curve in enumerate(loop):
        assert curve.type != 'circle'
        adjM[idx, 0] = curve.start_idx
        adjM[idx, 1] = curve.end_idx
    return adjM


def find_adj(index, adjM):
    loc_start = adjM[index][0]
    loc_end = adjM[index][1]
    matching_adj = np.where(adjM[:, 0]==loc_start)[0].tolist() + np.where(adjM[:, 0]==loc_end)[0].tolist() +\
                   np.where(adjM[:, 1]==loc_start)[0].tolist() + np.where(adjM[:, 1]==loc_end)[0].tolist()
    matching_adj = list(set(matching_adj) - set([index])) # unique, do not count itself 
    assert len(matching_adj) >= 1
    return matching_adj


def flip_curve(curve):
    tmp = curve.start_idx
    tmp2 = curve.start
    curve.start_idx = curve.end_idx
    curve.start = curve.end
    curve.end_idx = tmp 
    curve.end = tmp2 


def sort_start_end(sorted_loop):
    prev_curve = sorted_loop[0]
    for next_curve in sorted_loop[1:]:
        if prev_curve.end_idx != next_curve.start_idx:

            shared_idx = list(set([prev_curve.start_idx, prev_curve.end_idx]).intersection(
                set([next_curve.start_idx, next_curve.end_idx])
            )) 
            
            # back to itself
            if len(shared_idx) == 2:
                flip_curve(next_curve) 

            else:
                assert len(shared_idx) == 1
                shared_idx = shared_idx[0]
                if prev_curve.end_idx != shared_idx:
                    flip_curve(prev_curve)
                if next_curve.start_idx != shared_idx:
                    flip_curve(next_curve)
        prev_curve = next_curve 

    return
    

def sort_curves(loop):
    """ sort loop start / end vertex """
    if len(loop) == 1:
        assert loop[0].type == 'circle'
        return loop # no need to sort circle

    curve_bbox = []
    for curve in loop:
        curve_bbox.append(curve.bbox)
    curve_bbox = np.vstack(curve_bbox)
    min_values = curve_bbox[:, [0,2]]
    # Sort by X min, then by Y min (increasing)
    ind = np.lexsort(np.transpose(min_values)[::-1])

    # Start from bottom left 
    start_curve_idx = ind[0]
    sorted_idx = [start_curve_idx]
    curve_adjM = curve_connection(loop)

    iter = 0
    while True:
        # Determine next curve
        matching_adj = find_adj(sorted_idx[-1], curve_adjM)
        matching_adj = list(set(matching_adj) - set(sorted_idx)) # remove exisiting ones

        if len(matching_adj) == 0:
            break # connect back to itself 

        if iter > 10000: # should be enough?
            raise Exception('fail to sort loop')

        # Second curve has two options, choose increasing x direction 
        if len(matching_adj) > 1: 
            bottom_left0 = loop[matching_adj[0]].bottom_left
            bottom_left1 = loop[matching_adj[1]].bottom_left
            if bottom_left1[0] > bottom_left0[0]:
                sorted_idx.append(matching_adj[1]) 
            else:
                sorted_idx.append(matching_adj[0]) 
        else:
            # Follow curve connection
            sorted_idx.append(matching_adj[0])

        iter += 1

    assert len(list(set(sorted_idx))) == len(sorted_idx) # no repeat
    sorted_loop = [loop[x] for x in sorted_idx]

    # Make sure curve end is next one's start 
    sort_start_end(sorted_loop)
    assert sorted_loop[0].start_idx == sorted_loop[-1].end_idx
    return sorted_loop

    
def parse_curve(line, cmds, params, center, scale, bit):
    if line.type == 'line':
        start = quantize((line.start-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        end = quantize((line.end-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        parameter = np.concatenate([start, end, [-1]*4])
        params.append(parameter)
        cmds.append(1)
    elif line.type == 'arc':
        start = quantize((line.start-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        mid = quantize((line.mid-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        end = quantize((line.end-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        parameter = np.concatenate([start, mid, end, [-1]*2])
        params.append(parameter)
        cmds.append(2)
    elif line.type == 'circle':
        pt1 = quantize((line.pt1-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt2 = quantize((line.pt2-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt3 = quantize((line.pt3-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt4 = quantize((line.pt4-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        parameter = np.concatenate([pt1, pt2, pt3, pt4])
        params.append(parameter)
        cmds.append(3)
    return


def parse_curve2(line, vertex, center, scale, bit):
    if line.type == 'line':
        start = quantize((line.start-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        end = quantize((line.end-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        vertex.append(start)
        vertex.append(end)
    elif line.type == 'arc':
        start = quantize((line.start-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        mid = quantize((line.mid-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        end = quantize((line.end-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        vertex.append(start)
        vertex.append(mid)
        vertex.append(end)
    elif line.type == 'circle':
        pt1 = quantize((line.pt1-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt2 = quantize((line.pt2-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt3 = quantize((line.pt3-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt4 = quantize((line.pt4-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        vertex.append(pt1)
        vertex.append(pt2)
        vertex.append(pt3)
        vertex.append(pt4)
    return


def convert_profile(sketch, bit, cad_uid, sketch_uid):
    """ convert to code format """
    # Sort faces in a sketch based on bbox (X->Y) 
    sorted_sketch = sort_faces(sketch)

    # Normalization for the entire sketch
    vertex_total = []
    for face in sorted_sketch:
        sorted_face = sort_loops(face)
        for loop in sorted_face:
            sorted_loop = sort_curves(loop)
            vertices = get_vertex(sorted_loop)
            vertex_total.append(vertices)
    vertex_total = np.vstack(vertex_total)
    sketch_center_v, sketch_center = center_vertices(vertex_total)
    _, sketch_scale = normalize_vertices_scale(sketch_center_v)

    sketch_bbox = []
    # Multiple faces in a sketch
    for face in sorted_sketch:
        # Sort inner loops in face based on min bbox coords(X->Y)
        sorted_face = sort_loops(face)

        # Multiple loops in a face
        for loop in sorted_face:
            # Sort curves in a loop, 
            sorted_loop = sort_curves(loop)
            
            # Loop parameters and bbox (sketch-coordinate)
            vertex = []
            for line in sorted_loop:
                parse_curve2(line, vertex, sketch_center, sketch_scale, bit)
            vertex = np.vstack(vertex)
            
            x_min, x_max = np.min(vertex[:,0]), np.max(vertex[:,0])
            y_min, y_max = np.min(vertex[:,1]), np.max(vertex[:,1])
            
            bbox = np.array([x_min, x_max, y_min, y_max])
            sketch_bbox.append(bbox)
    
    sketch_bbox = np.vstack(sketch_bbox)
    data = {
        'profile': sketch_bbox,
        'name': cad_uid,
        'uid': cad_uid.split('/')[-1] + '_' + str(sketch_uid) 
    }
    return data


def get_vertex(loop):
    """
    Fetch all vertex in curvegen style
    """
    vertices = []
    for curve in loop:
        if curve.type == 'line':
            vertices.append(curve.start)
            vertices.append(curve.end)
        if curve.type == 'arc':
            vertices.append(curve.start)
            vertices.append(curve.mid)
            vertices.append(curve.end)
        if curve.type == 'circle':
            vertices.append(curve.pt1)
            vertices.append(curve.pt2)
            vertices.append(curve.pt3)
            vertices.append(curve.pt4)
    vertices = np.vstack(vertices)
    return vertices
                
             
def center_vertices(vertices):
  """Translate the vertices so that bounding box is centered at zero."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  vert_center = 0.5 * (vert_min + vert_max)
  return vertices - vert_center, vert_center


def normalize_vertices_scale(vertices):
  """Scale the vertices so that the long diagonal of the bounding box is one."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = 0.5*np.sqrt(np.sum(extents**2))  # -1 ~ 1
  return vertices / scale, scale


def convert_loop(sketch, bit, cad_uid, sketch_uid):
    """ convert to code format """
    # Sort faces in a sketch based on bbox (X->Y) 
    sorted_sketch = sort_faces(sketch)

    # Multiple faces in a sketch
    loop_datas = []
    for face in sorted_sketch:
        # Sort inner loops in face based on min bbox coords(X->Y)
        sorted_face = sort_loops(face)

        # Multiple loops in a face
        for loop in sorted_face:
            # Sort curves in a loop, 
            sorted_loop = sort_curves(loop)

            # Normalize per loop
            loop_center_v, loop_center = center_vertices(get_vertex(sorted_loop))
            _, loop_scale = normalize_vertices_scale(loop_center_v)

            # Loop parameters (bbox-coordinate)
            curves = []
            cmds = []
            for line in sorted_loop:
                parse_curve(line, cmds, curves, loop_center, loop_scale, bit)
            curves = np.vstack(curves)
            cmds = np.array(cmds)

            loop_data = {
                'param': curves,
                'cmd': cmds,
                'name': cad_uid,
                'uid': cad_uid.split('/')[-1] + '_' + str(sketch_uid)  + '_' + str(len(loop_datas)),
            }
            loop_datas.append(loop_data)
   
    return loop_datas


def process_solid(data):
    """
    Compute the 3D bbox of extruded solids (normalized 3D space)
    """
    project_folder, bit = data
    extrude_files = find_files(project_folder, 'extrude.stl')
    obj_files = find_files(project_folder, '.obj')
    if len(obj_files) == 0 or len(extrude_files)==0 or len(obj_files)!=len(extrude_files):
        return []

    cuboids = []
    for ext_file, obj_file in zip(extrude_files, obj_files):
        # Load cuboid mesh
        mesh = meshio.read(str(ext_file))
        vertex = mesh.points
        x_min, x_max = np.min(vertex[:,0]), np.max(vertex[:,0])
        y_min, y_max = np.min(vertex[:,1]), np.max(vertex[:,1])
        z_min, z_max = np.min(vertex[:,2]), np.max(vertex[:,2])
    
        x_min = quantize(x_min, n_bits=bit, min_range=-CUBOID_RANGE, max_range=+CUBOID_RANGE)
        x_max = quantize(x_max, n_bits=bit, min_range=-CUBOID_RANGE, max_range=+CUBOID_RANGE)
        y_min = quantize(y_min, n_bits=bit, min_range=-CUBOID_RANGE, max_range=+CUBOID_RANGE)
        y_max = quantize(y_max, n_bits=bit, min_range=-CUBOID_RANGE, max_range=+CUBOID_RANGE)
        z_min = quantize(z_min, n_bits=bit, min_range=-CUBOID_RANGE, max_range=+CUBOID_RANGE)
        z_max = quantize(z_max, n_bits=bit, min_range=-CUBOID_RANGE, max_range=+CUBOID_RANGE)

        # Read in the obj file 
        parser = OBJParser(Path(obj_file)) 
        _, _, meta_info = parser.parse_file(1.0)  
    
        # Set operation 
        set_op = meta_info['set_op']
        if set_op == 'JoinFeatureOperation' or set_op == 'NewBodyFeatureOperation':
            extrude_op = 0 #'add'
        elif set_op == 'CutFeatureOperation':
            extrude_op = 1 #'cut'
        elif set_op == 'IntersectFeatureOperation':
            extrude_op = 2 #'intersect'

        cuboids.append([extrude_op, x_min, x_max, y_min, y_max, z_min, z_max])
    cuboids = np.vstack(cuboids)    

    data = {'name': str(obj_file.parent)[-13:],
            'solid':  cuboids}

    return [data]


def process_profile(data):
    """
    Compute the 2D bbox of loops in a sketch profile (normalized 2D space)
    """
    project_folder, bit = data
    obj_files = find_files(project_folder, '.obj')
    if len(obj_files) == 0:
        return []

    datas = []
    for uid, obj_file in enumerate(obj_files):
        # Read in the obj file 
        parser = OBJParser(Path(obj_file)) 
        _, sketch, _ = parser.parse_file(1.0)  
        # Convert
        try:
            parsed_data = convert_profile(sketch, bit, str(obj_file.parent)[-13:], uid)
            datas.append(parsed_data)
        except Exception:
            pass
    return datas


def process_loop(data):
    """
    Compute the curve parameters in a loop (normalized 2D space)
    """
    project_folder, bit = data
    obj_files = find_files(project_folder, '.obj')
    if len(obj_files) == 0:
        return []
    
    datas = []
    for uid, obj_file in enumerate(obj_files):
        # Read in the obj file 
        parser = OBJParser(Path(obj_file)) 
        _, sketch, _ = parser.parse_file(1.0)  
        # Convert
        try:
            datas += convert_loop(sketch, bit, str(obj_file.parent)[-13:], uid)
        except Exception:
           pass
    return datas


def normalize_vertices_scale2(vertices):
  """Scale the vertices so that the long diagonal of the bounding box is one."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = max(extents) / (2*SKETCH_R)
  return vertices / scale, scale


def parse_curve3(line, params, cmds, bit, center, scale):
    if line.type == 'line':
        start = quantize((line.start-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        parameter = np.concatenate([start, [-1]*6])
        params.append(parameter)
        cmds.append(4)

    elif line.type == 'arc':
        start = quantize((line.start-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        mid = quantize((line.mid-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        parameter = np.concatenate([start, mid, [-1]*4])
        params.append(parameter)
        cmds.append(5)  

    elif line.type == 'circle':
        pt1 = quantize((line.pt1-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt2 = quantize((line.pt2-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt3 = quantize((line.pt3-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        pt4 = quantize((line.pt4-center)/scale, n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        parameter = np.concatenate([pt1, pt2, pt3, pt4])
        params.append(parameter)
        cmds.append(6) 
    return


def convert_cad(sketch, bit):
    """ convert to code format """
    # Sort faces in sketch based on min bbox coords (X->Y) 
    sorted_sketch = sort_faces(sketch)

    # Normalization for the entire sketch
    vertex_total = []
    for face in sorted_sketch:
        sorted_face = sort_loops(face)
        for loop in sorted_face:
            sorted_loop = sort_curves(loop)
            vertices = get_vertex(sorted_loop)
            vertex_total.append(vertices)
    vertex_total = np.vstack(vertex_total)
    sketch_center_v, center = center_vertices(vertex_total)
    _, scale = normalize_vertices_scale2(sketch_center_v)

    curve = []
    command = []

    # Multiple faces in a sketch
    for face in sorted_sketch:
        # Sort inner loops in face based on min bbox coords(X->Y)
        sorted_face = sort_loops(face)

        # Multiple loops in a face
        for loop in sorted_face:
            # Sort curves in a loop, 
            sorted_loop = sort_curves(loop)
            
            # XY coordinate
            for line in sorted_loop:
                parse_curve3(line, curve, command, bit, center, scale)

            curve.append(np.array([-1]*8)) # loop end
            command.append(3)

        curve.append(np.array([-1]*8)) # face end
        command.append(2)

    curve.append(np.array([-1]*8)) # sketch end
    command.append(1)

    params = np.vstack(curve).astype(int)
    cmds = np.array(command).astype(int)
    
    return cmds, params, center, scale


ROT = np.array([[-1, -1,  0,  0,  0,  1, -1,  1,  0],
                [-1,  0,  0,  0, -1,  1,  0,  1,  1],
                [-1,  0,  0,  0,  0,  1,  0,  1,  0],
                [-1,  0,  0,  0,  1,  1,  0,  1, -1],
                [-1,  1,  0,  0,  0,  1,  1,  1,  0],
                [ 0, -1,  0, -1,  0,  1, -1,  0, -1],
                [ 0, -1,  0,  0,  0,  1, -1,  0,  0],
                [ 0, -1,  0,  1,  0,  1, -1,  0,  1],
                [ 0,  1,  0, -1,  0,  1,  1,  0,  1],
                [ 0,  1,  0,  0,  0,  1,  1,  0,  0],
                [ 0,  1,  0,  1,  0,  1,  1,  0, -1],
                [ 1, -1,  0,  0,  0,  1, -1, -1,  0],
                [ 1,  0, -1,  0, -1,  0, -1,  0, -1],
                [ 1,  0, -1,  0,  1,  0,  1,  0,  1],
                [ 1,  0,  0,  0, -1, -1,  0,  1, -1],
                [ 1,  0,  0,  0, -1,  0,  0,  0, -1],
                [ 1,  0,  0,  0, -1,  1, -1, -1, -1],
                [ 1,  0,  0,  0, -1,  1,  0, -1, -1],
                [ 1,  0,  0,  0,  0,  1,  0, -1,  0],
                [ 1,  0,  0,  0,  1, -1,  0,  1,  1],
                [ 1,  0,  0,  0,  1,  0,  0,  0,  1],
                [ 1,  0,  0,  0,  1,  1,  0, -1,  1],
                [ 1,  0,  1,  0, -1,  0,  1,  0, -1],
                [ 1,  0,  1,  0,  1,  0, -1,  0,  1],
                [ 1,  1,  0,  0,  0,  1,  1, -1,  0]])


def process_model(data):
    """Convert a sequence of obj files to vector format."""
    project_folder, bit = data
    obj_files = find_files(project_folder, '.obj')
    if len(obj_files) == 0:
        return []
    
    se_param = []
    se_ext = []
    se_cmd = []

    for obj_file in obj_files:
        # Read in the obj file 
        parser = OBJParser(Path(obj_file)) 
        _, sketch, meta_info = parser.parse_file(1.0)  
        
        # Convert it to a vector code         
        try:
            cmds, params, center, scale = convert_cad(sketch, bit)
        except Exception as ex:
            print(f'Can not convert code for {str(obj_file.parent)}')
            return []
    
        # Set operation 
        set_op = meta_info['set_op']
        if set_op == 'JoinFeatureOperation' or set_op == 'NewBodyFeatureOperation':
            extrude_op = 1 #'add'
        elif set_op == 'CutFeatureOperation':
            extrude_op = 2 #'cut'
        elif set_op == 'IntersectFeatureOperation':
            extrude_op = 3 #'intersect'
        ext_type = np.array([extrude_op])

        # Sketch rescaling values 
        center = quantize(np.array(center), n_bits=bit, min_range=-1, max_range=+1) # scale 0~1
        scale = quantize(np.array([scale]), n_bits=bit, min_range=0, max_range=+1) # center -1 ~ 1
        
        # Extrude values
        ext_v = quantize(np.array(meta_info['extrude_value']), n_bits=bit, min_range=-EXTRUDE_R, max_range=+EXTRUDE_R)

        # Transformation origin
        ext_T =  quantize(np.array(meta_info['t_orig']), n_bits=bit, min_range=-SKETCH_R, max_range=+SKETCH_R)
        
        # Transformation rotation
        ext_TX = np.clip(np.rint(np.array(meta_info['t_x'])).astype(int), -1, 1)     # -1 / 0 / 1 
        ext_TY = np.clip(np.rint(np.array(meta_info['t_y'])).astype(int), -1, 1) 
        ext_TZ = np.clip(np.rint(np.array(meta_info['t_z'])).astype(int), -1, 1) 
        ext_R = np.concatenate([ext_TX, ext_TY, ext_TZ]) 
        assert (ROT == ext_R).all(axis=1).sum() == 1
        ext_R_idx = np.where((ROT == ext_R).all(axis=1)==True)[0][0]  # 25 different selection
  
        # Full parameters
        extrude_param = np.concatenate([center, scale, ext_v, ext_T, np.array([ext_R_idx]), ext_type]) 
        se_ext.append(extrude_param)
        se_param.append(params)
        se_cmd.append(cmds)
    
    data = {'name': str(obj_file.parent)[-13:],
            'cad_cmd': se_cmd,
            'cad_param': se_param,
            'cad_ext': se_ext,
            }

    return [data]

