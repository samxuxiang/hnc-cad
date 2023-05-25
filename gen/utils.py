import os 
from pathlib import Path
import os
import numpy as np 
from pathlib import Path
import math 
import pdb 
from collections import OrderedDict
from config import * 


def dequantize(verts, n_bits=6, min_range=-1, max_range=1):
  """Convert quantized vertices to floats."""
  range_quantize = 2**n_bits - 1
  verts = verts.astype('float32')
  verts = verts * (max_range - min_range) / range_quantize + min_range
  return verts


def rads_to_degs(rads):
    return 180*rads/math.pi


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


def find_arc_geometry(a, b, c):
        A = b[0] - a[0] 
        B = b[1] - a[1]
        C = c[0] - a[0]
        D = c[1] - a[1]
    
        E = A*(a[0] + b[0]) + B*(a[1] + b[1])
        F = C*(a[0] + c[0]) + D*(a[1] + c[1])
    
        G = 2.0*(A*(c[1] - b[1])-B*(c[0] - b[0])) 

        if G == 0:
            raise Exception("zero G")

        p_0 = (D*E - B*F) / G
        p_1 = (A*F - C*E) / G

        center = np.array([p_0, p_1])
        radius = np.linalg.norm(center - a)

        angles = []
        for xx in [a,b,c]:
            angle = angle_from_vector_to_x(xx - center)
            angles.append(angle)

        ab = b-a
        ac = c-a
        cp = np.cross(ab, ac)
        if cp >= 0:
            start_angle_rads = angles[0]
            end_angle_rads = angles[2]
        else:
            start_angle_rads = angles[2]
            end_angle_rads = angles[0]

        return center, radius, start_angle_rads, end_angle_rads


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
 
class CADparser:
    """ Parse into OBJ files """
    def __init__(self, bit):
        self.vertex_dict = OrderedDict()
        self.bit = bit
    
    def perform(self, sketches, extrudes): 
        se_datas = []   
        END = np.where(extrudes==0)[0][0]
        extrudes = extrudes[:END] # valid extrude 

        END = np.where(extrudes==1)[0]
        extrudes = np.split(extrudes, END+1)[:-1] # split extrudes

        END = np.where(sketches[:,0]==0)[0][0] 
        sketches = sketches[:END] # valid sketch

        END = np.where(sketches[:,0]==1)[0]
        sketches = np.split(sketches, END+1)[:-1] # split sketch 
    
        # (2) Sequentially parse each pair of SE into obj 
        for (sketch, extrude) in zip(sketches, extrudes):
            center = dequantize(extrude[0:2]-EXT_PAD, n_bits=self.bit, 
                                             min_range=-1, max_range=1)
            scale = dequantize(extrude[2]-EXT_PAD, n_bits=self.bit, 
                                             min_range=0, max_range=1)
            extrude_value = dequantize(extrude[3:5]-EXT_PAD, n_bits=self.bit, 
                                             min_range=-EXTRUDE_R, max_range=EXTRUDE_R)
            extrude_T = dequantize(extrude[5:8]-EXT_PAD, n_bits=self.bit, 
                                         min_range=-EXTRUDE_R, max_range=EXTRUDE_R)
            extrude_R = ROT[extrude[8]-EXT_PAD]
            extrude_op = extrude[9]-EXT_PAD 
            extrude_param = {'value': extrude_value, 
                            'R': extrude_R,
                            'T': extrude_T,
                            'op': extrude_op,
                            'S': scale}

            sketch = sketch[:-1]
            extrude = extrude[:-1]
            
            faces = np.split(sketch, np.where(sketch[:,0]==2)[0]+1)
            faces = faces[:-1]
            
            # Each face
            se_str = ""
            for face_idx, face in enumerate(faces):
                face_str = "face\n"
                face = face[:-1]
                loops = np.split(face, np.where(face[:,0]==3)[0]+1)
                assert len(loops[-1]) == 0
                loops = loops[:-1]
               
                # Each loop
                for loop_idx, loop in enumerate(loops):
                    loop = loop[:-1]
                    curves = np.split(loop, np.where(loop[:,0]==4)[0]+1)
                    assert len(curves[-1]) == 0
                    curves = curves[:-1]
                    
                    loop_curves = []
                    for curve in curves:
                        curve = curve[:-1] - SKETCH_PAD # remove padding
                        loop_curves.append(curve)

                    # Draw a single loop curves
                    next_loop_curves = loop_curves[1:]
                    next_loop_curves += loop_curves[:1]
                   
                    cur_str = []
                    for cur, next_cur in zip(loop_curves, next_loop_curves):
                        self.obj_curve(cur, next_cur, cur_str, center, scale)
                   
                    loop_string = ""
                    for cur in cur_str:
                        loop_string += f"{cur}\n"
                    
                    if loop_idx == 0:
                        face_str += f"out\n{loop_string}\n"
                    else:
                        face_str += f"in\n{loop_string}\n"

                se_str += face_str 

            vertex_str = self.convert_vertices() 
            
            # (3) Convert extrusion parameters
            se_data = {'vertex': vertex_str, 'curve': se_str, 'extrude': extrude_param}
            se_datas.append(se_data)
            self.vertex_dict.clear()

        return se_datas


    def obj_curve(self, curve, next_curve, cur_str,_center_, _scale_):
        if len(curve)==4: # Circle
            p1 = dequantize(curve[0], n_bits=self.bit, min_range=-SKETCH_R, max_range=SKETCH_R)*_scale_+_center_
            p2 = dequantize(curve[1], n_bits=self.bit, min_range=-SKETCH_R, max_range=SKETCH_R)*_scale_+_center_
            p3 = dequantize(curve[2], n_bits=self.bit, min_range=-SKETCH_R, max_range=SKETCH_R)*_scale_+_center_
            p4 = dequantize(curve[3], n_bits=self.bit, min_range=-SKETCH_R, max_range=SKETCH_R)*_scale_+_center_
            center = np.asarray([0.5*(p1[0]+p2[0]),  0.5*(p3[1]+p4[1])])
            radius = (np.linalg.norm(p1-p2) + np.linalg.norm(p3-p4))/4.0
            center_idx = self.save_vertex(center[0], center[1], 'p')
            radius_idx = self.save_vertex(radius, 0.0, 'r')
            cur_str.append(f"c {center_idx} {radius_idx}")

        elif len(curve) == 2: # Arc
            start_v = dequantize(curve[0], n_bits=self.bit, min_range=-SKETCH_R, max_range=SKETCH_R)*_scale_+_center_
            mid_v = dequantize(curve[1], n_bits=self.bit, min_range=-SKETCH_R, max_range=SKETCH_R)*_scale_+_center_
            end_v = dequantize(next_curve[0], n_bits=self.bit, min_range=-SKETCH_R, max_range=SKETCH_R)*_scale_+_center_
            center, _, _, _ = find_arc_geometry(start_v, mid_v, end_v)
            center_idx = self.save_vertex(center[0], center[1], 'p')
            start_idx = self.save_vertex(start_v[0], start_v[1], 'p')
            mid_idx = self.save_vertex(mid_v[0], mid_v[1], 'p')
            end_idx = self.save_vertex(end_v[0], end_v[1], 'p')
            cur_str.append(f"a {start_idx} {mid_idx} {center_idx} {end_idx}")

        elif len(curve) == 1: # Line
            start_v = dequantize(curve[0], n_bits=self.bit, min_range=-SKETCH_R, max_range=SKETCH_R)*_scale_+_center_
            end_v = dequantize(next_curve[0], n_bits=self.bit, min_range=-SKETCH_R, max_range=SKETCH_R)*_scale_+_center_
            start_idx = self.save_vertex(start_v[0], start_v[1], 'p')
            end_idx = self.save_vertex(end_v[0], end_v[1], 'p')
            cur_str.append(f"l {start_idx} {end_idx}")

        else:
            assert False
                                            

    def save_vertex(self, h_x, h_y, text):
        unique_key = f"{text}:x{h_x}y{h_y}"
        index = 0
        for key in self.vertex_dict.keys():
            # Vertex location already exist in dict
            if unique_key == key: 
                return index 
            index += 1
        # Vertex location does not exist in dict
        self.vertex_dict[unique_key] = [h_x, h_y]
        return index


    def convert_vertices(self):
        """ Convert all the vertices to .obj format """
        vertex_strings = ""
        for pt in self.vertex_dict.values():
            # e.g. v 0.123 0.234 0.345 1.0
            vertex_string = f"v {pt[0]} {pt[1]}\n"
            vertex_strings += vertex_string
        return vertex_strings


def parse3d(point3d):
    x = point3d['x']
    y = point3d['y']
    z = point3d['z']
    return str(x)+' '+str(y)+' '+str(z)


def parse3d_sample(point3d):
    x = point3d[0]
    y = point3d[1]
    z = point3d[2]
    return str(x)+' '+str(y)+' '+str(z)


def write_obj(file, curve_strings, curve_count, vertex_strings, vertex_count, extrude_info, refP_info):
    """Write an .obj file with the curves and verts"""
        
    with open(file, "w") as fh:
        # Write Meta info
        fh.write("# WaveFront *.obj file\n")
        fh.write(f"# Vertices: {vertex_count}\n")
        fh.write(f"# Curves: {curve_count}\n")
        fh.write("# ExtrudeOperation: "+extrude_info['set_op']+"\n")
        fh.write("\n")

        # Write vertex and curve
        fh.write(vertex_strings)
        fh.write("\n")
        fh.write(curve_strings)
        fh.write("\n")

        #Write extrude value 
        extrude_string = 'Extrude '
        for value in extrude_info['extrude_values']:
            extrude_string += str(value)+' '
        fh.write(extrude_string)
        fh.write("\n")

        #Write refe plane value 
        p_orig = parse3d(refP_info['pt']['origin'])
        x_axis = parse3d(refP_info['pt']['x_axis'])
        y_axis = parse3d(refP_info['pt']['y_axis'])
        z_axis = parse3d(refP_info['pt']['z_axis'])
        fh.write('T_origin '+p_orig)
        fh.write("\n")
        fh.write('T_xaxis '+x_axis)
        fh.write("\n")
        fh.write('T_yaxis '+y_axis)
        fh.write("\n")
        fh.write('T_zaxis '+z_axis)
        

def write_obj_sample(save_folder, data):
    for idx, write_data in enumerate(data):
        obj_name = Path(save_folder).stem + '_'+ str(idx).zfill(3) + "_param.obj"
        obj_file = Path(save_folder) / obj_name
        extrude_param = write_data['extrude']
        vertex_strings = write_data['vertex']
        curve_strings = write_data['curve']
        
        """Write an .obj file with the curves and verts"""
        if extrude_param['op'] == 1: #'add'
            set_op = 'NewBodyFeatureOperation'
        elif extrude_param['op'] == 2: #'cut'
            set_op = 'CutFeatureOperation'
        elif extrude_param['op'] == 3: #'cut'
            set_op = 'IntersectFeatureOperation'
        
        with open(obj_file, "w") as fh:
            # Write Meta info
            fh.write("# WaveFront *.obj file\n")
            fh.write("# ExtrudeOperation: "+set_op+"\n")
            fh.write("\n")

            # Write vertex and curve
            fh.write(vertex_strings)
            fh.write("\n")
            fh.write(curve_strings)
            fh.write("\n")

            #Write extrude value 
            extrude_string = 'Extrude '
            for value in extrude_param['value']:
                extrude_string += str(value)+' '
            fh.write(extrude_string)
            fh.write("\n")

            #Write refe plane value 
            p_orig = parse3d_sample(extrude_param['T'])
            x_axis = parse3d_sample(extrude_param['R'][0:3])
            y_axis = parse3d_sample(extrude_param['R'][3:6])
            z_axis = parse3d_sample(extrude_param['R'][6:9])
    
            fh.write('T_origin '+p_orig)
            fh.write("\n")
            fh.write('T_xaxis '+x_axis)
            fh.write("\n")
            fh.write('T_yaxis '+y_axis)
            fh.write("\n")
            fh.write('T_zaxis '+z_axis)
            fh.write("\n")