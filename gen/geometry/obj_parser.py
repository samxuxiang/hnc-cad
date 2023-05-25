import os
import sys
import numpy as np

from geometry.arc import Arc
from geometry.circle import Circle
from geometry.line import Line

from geometry import geom_utils
import pdb


class OBJParser:
    """
    A class to read an OBJ file containing the sketch data
    and hand it back in a form which is easy to work with.
    """
    def __init__(self, pathname=None):
        self.pathname = pathname


    def convert_vertices(self, vertices):
        """Convert all the vertices to .obj format"""
        vertex_strings = ""
        for pt in vertices:
            # e.g. v 0.123 0.234 0.345 1.0
            vertex_string = f"v {pt[0]} {pt[1]}\n"
            vertex_strings += vertex_string
        return vertex_strings


    def convert_curves(self, faces):
        curve_strings = ""
        total_curve = 0

        # Faces (multiple closed regions)
        for group_idx, loops in enumerate(faces):
            curve_strings += f"\nface\n"
            # Multiple loops (inner and outer)
            for loop in loops: 
                if loop[0].is_outer:  
                    curve_strings += f"out\n"
                else:
                    curve_strings += f"in\n"
                # All curves in one loop
                for curve in loop:
                    total_curve += 1
                    if curve.type == 'line':
                        curve_strings += f"l {curve.start_idx} {curve.end_idx}\n"
                    elif curve.type == 'circle':
                        curve_strings += f"c {curve.center_idx} {curve.radius_idx}\n"
                    elif curve.type == 'arc':
                        curve_strings += f"a {curve.start_idx} {curve.mid_idx} {curve.center_idx} {curve.end_idx}\n"

        return curve_strings, total_curve


    def parse3d(self, point3d):
        x = point3d[0]
        y = point3d[1]
        z = point3d[2]
        return str(x)+' '+str(y)+' '+str(z)


    def write_obj2(self, file, vertices, faces, meta_info, scale=None):
        """ Write to .obj file """
        vertex_strings = self.convert_vertices(vertices)
        curve_strings, total_curve = self.convert_curves(faces)
        
        with open(file, "w") as fh:
            # Write Meta info
            fh.write("# WaveFront *.obj file\n")
            fh.write(f"# Vertices: {len(vertices)}\n")
            fh.write(f"# Curves: {total_curve}\n")
            fh.write("\n")

            # Write vertex and curve
            fh.write(vertex_strings)
            fh.write("\n")
            fh.write(curve_strings)
            fh.write("\n")

            #Write extrude value 
            fh.write("ExtrudeOperation: " + meta_info['set_op']+"\n")
            extrude_string = 'Extrude '
            for value in meta_info['extrude_value']:
                extrude_string += str(value)+' '
            fh.write(extrude_string)
            fh.write("\n")
        
            #Write refe plane transformation 
            p_orig = self.parse3d(meta_info['t_orig'])
            x_axis = self.parse3d(meta_info['t_x'])
            y_axis = self.parse3d(meta_info['t_y'])
            z_axis = self.parse3d(meta_info['t_z'])
            fh.write('T_origin '+p_orig)
            fh.write("\n")
            fh.write('T_xaxis '+x_axis)
            fh.write("\n")
            fh.write('T_yaxis '+y_axis)
            fh.write("\n")
            fh.write('T_zaxis '+z_axis)
            fh.write("\n")

            # Normalized object 
            if scale is not None:
                fh.write('Scale '+str(scale))


    def write_obj(self, file, curve_strings, total_curve, vertex_strings, total_v, meta_info, scale=None):
        """ Write to .obj file """
        #vertex_strings = self.convert_vertices(vertices)
        #curve_strings, total_curve = self.convert_curves(faces)
        
        with open(file, "w") as fh:
            # Write Meta info
            fh.write("# WaveFront *.obj file\n")
            fh.write(f"# Vertices: {total_v}\n")
            fh.write(f"# Curves: {total_curve}\n")
            fh.write("\n")

            # Write vertex and curve
            fh.write(vertex_strings)
            fh.write("\n")
            fh.write(curve_strings)
            fh.write("\n")

            #Write extrude value 
            fh.write("ExtrudeOperation: " + meta_info['set_op']+"\n")
            extrude_string = 'Extrude '
            for value in meta_info['extrude_value']:
                extrude_string += str(value)+' '
            fh.write(extrude_string)
            fh.write("\n")
        
            #Write refe plane transformation 
            p_orig = self.parse3d(meta_info['t_orig'])
            x_axis = self.parse3d(meta_info['t_x'])
            y_axis = self.parse3d(meta_info['t_y'])
            z_axis = self.parse3d(meta_info['t_z'])
            fh.write('T_origin '+p_orig)
            fh.write("\n")
            fh.write('T_xaxis '+x_axis)
            fh.write("\n")
            fh.write('T_yaxis '+y_axis)
            fh.write("\n")
            fh.write('T_zaxis '+z_axis)
            fh.write("\n")

            # Normalized object 
            if scale is not None:
                fh.write('Scale '+str(scale))


    def parse_file(self, scale=1.0):
        """ 
        Parse obj file
        Return
            vertex 2D location numpy
            curve list (geometry class)
            extrude parameters
        """ 
       
        assert self.pathname is not None, "File is None"
        assert self.pathname.exists(), "No such file"

        # Parse file 
        vertex_list = []
        
        # Read vertice
        with open(self.pathname) as obj_file:
            for line in obj_file:
                tokens = line.split()
                if not tokens:
                    continue
                line_type = tokens[0]
                # Vertex
                if line_type == "v":
                    vertex_list.append([float(x) for x in tokens[1:]])
        vertices = np.array(vertex_list, dtype=np.float64) * scale

        # Read curves
        faces = []
        loops = []
        loop = []
        
        # Read in all lines
        lines = []
        with open(self.pathname) as obj_file:
            for line in obj_file:
                lines.append(line)

        # Parse all lines
        faces = []
        for str_idx, line in enumerate(lines):
            tokens = line.split()
            if not tokens:
                continue
            line_type = tokens[0]

            # Start of a new face 
            if line_type == "face":
                faces.append(self.read_face(lines, str_idx+1, vertices))

            # Read meta data
            meta_data = line.strip('# ').strip(' \n').split(' ')
            meta_name = meta_data[0]
        
            if meta_name == 'Extrude':
                extrude_values = [float(x) for x in meta_data[1:]]
                extrude_values = [x*scale for x in extrude_values]
            elif meta_name == 'T_origin':
                t_orig = [float(x) for x in meta_data[1:]] 
                t_orig = [x*scale for x in t_orig] 
            elif meta_name == 'T_xaxis':
                t_x = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'T_yaxis':
                t_y = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'T_zaxis':
                t_z = [float(x) for x in meta_data[1:]] 
            elif meta_name == 'ExtrudeOperation:':
                set_op = meta_data[1]

        meta_info = {'extrude_value': extrude_values,
                     'set_op': set_op,
                     't_orig': t_orig,
                     't_x': t_x,
                     't_y': t_y,
                     't_z': t_z,
                    }

        return vertices, faces, meta_info    
         


    def read_face(self, lines, str_idx, vertices):
        loops = []
        loop = []
        for line in lines[str_idx:]:
            tokens = line.split()
            if not tokens:
                continue
            line_type = tokens[0]

            if line_type == 'face':
                break

            # Start of a new loop 
            if line_type == "out" or line_type == "in":
                if len(loop) > 0:
                    loops.append(loop)
                loop = []
                is_outer = (line_type == 'out')

            # Line
            if line_type == 'l':
                c_tok = tokens[1:]
                curve = Line([int(c_tok[0]), int(c_tok[1])], vertices, is_outer=is_outer)
                loop.append(curve)

            # Arc 
            if line_type == 'a':
                c_tok = tokens[1:]
                curve = Arc([int(c_tok[0]), int(c_tok[1]), int(c_tok[2]), int(c_tok[3])], vertices, is_outer=is_outer)
                loop.append(curve)

            # Circle 
            if line_type == 'c':
                c_tok = tokens[1:]
                curve = Circle([int(c_tok[0]), int(c_tok[1])], vertices, is_outer=is_outer)
                loop.append(curve)

        loops.append(loop)
        return loops
