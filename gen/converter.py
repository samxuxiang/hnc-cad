import numpy as np
import os
from collections import OrderedDict
import copy

# OCC
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_XYZ, gp_Ax3, gp_Trsf, gp_Pln
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, 
                                    BRepBuilderAPI_MakeWire,
                                    BRepBuilderAPI_MakeEdge)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire
from OCC.Core.gp import (gp_Vec, gp_Ax2, gp_Dir, gp_Circ)
from OCC.Extend.DataExchange import write_stl_file
from utils import write_obj


def round_float(point):
    point['x'] = round(point['x'], 9)
    point['y'] = round(point['y'], 9)
    point['z'] = round(point['z'], 9)
    return


def same_plane(plane1, plane2):
    same = True 
    trans1 = plane1['pt']
    trans2 = plane2['pt']
    for key in trans1.keys():
        v1 = trans1[key]
        v2 = trans2[key]
        if v1['x'] != v2['x'] or v1['y'] != v2['y'] or v1['z'] != v2['z']:
            same = False 
    return same 


def create_xyz(xyz):
    return gp_XYZ(
        xyz["x"],
        xyz["y"],
        xyz["z"]
    )


def get_ax3(transform_dict):
    origin = create_xyz(transform_dict["origin"])
    x_axis = create_xyz(transform_dict["x_axis"])
    y_axis = create_xyz(transform_dict["y_axis"])
    z_axis = create_xyz(transform_dict["z_axis"])
    # Create new coord (orig, Norm, x-axis)
    axis3 = gp_Ax3(gp_Pnt(origin), gp_Dir(z_axis), gp_Dir(x_axis)) 
    return axis3


def get_transform(transform_dict):
    axis3 = get_ax3(transform_dict)
    transform_to_local = gp_Trsf()
    transform_to_local.SetTransformation(axis3) 
    return transform_to_local.Inverted()


def create_sketch_plane(transform_dict):
    axis3 = get_ax3(transform_dict)
    return gp_Pln(axis3)


def create_point(point_dict, transform):
    pt2d = gp_Pnt(
        point_dict["x"],
        point_dict["y"],
        point_dict["z"]
    )
    return pt2d.Transformed(transform)


def create_vector(vec_dict, transform):
    vec2d = gp_Vec(
        vec_dict["x"],
        vec_dict["y"],
        vec_dict["z"]
    )
    return vec2d.Transformed(transform)


def create_unit_vec(vec_dict, transform):
    vec2d = gp_Dir(
        vec_dict["x"],
        vec_dict["y"],
        vec_dict["z"]
    )
    return vec2d.Transformed(transform)


class DeepCADReconverter:
    """ DeepCAD Data Reconverter """
    
    def __init__(self, data, fileName):
        self.data = data
        self.vertex_dict = OrderedDict()
        self.PRECISION = 1e-5
        self.eps = 1e-7
        self.fileName = fileName
        self.x_axis = gp_Dir(1.0, 0.0, 0.0)


    def my_op(self, big, small, op_name): 
        if op_name == 'cut':
            op = BRepAlgoAPI_Cut(big, small) 
        elif op_name == 'fuse':
            op = BRepAlgoAPI_Fuse(big, small)
        elif op_name == 'common':
            op = BRepAlgoAPI_Common(big, small)
        op.SetFuzzyValue(self.PRECISION) 
        op.Build() 
        return op.Shape()
    
    
    def parse_json(self, save_folder):
        """ parse the json data to vector format """
        cur_solid = None
        extrude_idx = 0

        # Fetch all sketch-extrude information
        for timeline_ent in self.data["sequence"]:

            # Parse a single sketch-extrude  
            if timeline_ent["type"] == "ExtrudeFeature":

                extrude_uuid = timeline_ent["entity"]
                extrusion = self.data["entities"][extrude_uuid]

                # Extrusion with empty loop (skip) 
                if len(extrusion["profiles"]) == 0:
                    self.vertex_dict.clear()
                    continue

                # Parse extrusion values and type 
                extrude_params = self.get_ExtrudeParam(extrusion)

                # Parse vectors and brep
                ext_solid, curve_strings, curve_count, p_trans = self.parse_solid(extrusion, extrude_params)

                # Check empty solid (skip)
                props = GProp_GProps()
                brepgprop_VolumeProperties(ext_solid, props)
                solid_volume = props.Mass()
                if solid_volume == 0:
                    self.vertex_dict.clear()
                    continue
                
                prev_solid = copy.deepcopy(cur_solid)
                # Perform set operation and save current cad up to this step
                set_op = extrusion["operation"]
                if set_op == "NewBodyFeatureOperation" or set_op == "JoinFeatureOperation":
                    if cur_solid is None:
                        cur_solid = ext_solid
                    else:
                        cur_solid = self.my_op(cur_solid, ext_solid, 'fuse')
                elif set_op == "CutFeatureOperation":
                    cur_solid = self.my_op(cur_solid, ext_solid, 'cut')
                elif set_op == "IntersectFeatureOperation":
                    cur_solid = self.my_op(cur_solid, ext_solid, 'common')
                else:
                    raise Exception("Unknown operation type")
               
                # Nothing happened
                if cur_solid is None:
                    self.vertex_dict.clear()
                    continue

                # Check solid changes (skip)
                if prev_solid is not None:
                    props = GProp_GProps()
                    brepgprop_VolumeProperties(prev_solid, props)
                    solid_volume1 = props.Mass()
                    brepgprop_VolumeProperties(cur_solid, props)
                    solid_volume2 = props.Mass()
                    if (solid_volume1 - solid_volume2) == 0:
                        self.vertex_dict.clear()
                        continue

                # Solid not valid (skip)
                analyzer = BRepCheck_Analyzer(cur_solid)
                if not analyzer.IsValid():
                    self.vertex_dict.clear()
                    continue
                    #raise Exception("brep check failed")
                '''
                # Save step 
                step_name = save_folder.stem + '_'+ str(extrude_idx).zfill(3) + "_extrude.step"
                output_path =  os.path.join(str(save_folder), step_name)
                write_step_file(ext_solid, output_path)

                step_name = save_folder.stem + '_'+ str(extrude_idx).zfill(3) + "_post.step"
                output_path =  os.path.join(str(save_folder), step_name)
                write_step_file(cur_solid, output_path)
                '''
                # Save stl 
                stl_name = save_folder.stem + '_'+ str(extrude_idx).zfill(3) + "_extrude.stl"
                output_path =  os.path.join(str(save_folder), stl_name)
                write_stl_file(ext_solid, output_path, linear_deflection=0.001, angular_deflection=0.5)

                stl_name = save_folder.stem + '_'+ str(extrude_idx).zfill(3) + "_post.stl"
                output_path =  os.path.join(str(save_folder), stl_name)
                write_stl_file(cur_solid, output_path, linear_deflection=0.001, angular_deflection=0.5)
                
                # Save vector data .obj file 
                vertex_strings = self.convert_vertices()    
                obj_name = save_folder.stem + '_'+ str(extrude_idx).zfill(3) + "_param.obj"
                obj_file = save_folder / obj_name
                write_obj(obj_file, curve_strings, curve_count, vertex_strings, len(self.vertex_dict), extrude_params, p_trans)
                
                # Add one to extrusion index 
                extrude_idx += 1

                # Clear dictionary 
                self.vertex_dict.clear()
       
        if extrude_idx == 0:
            raise Exception('no data created, unknown error')
        
        return 
    

    def get_ExtrudeParam(self, entity):
        """
        Given an extrude entity, extract the extrusion parameters
        Returns 
            extrude_type
            extrude_values  [small, large]
        """
        extrude_dict = {}

        # Extrude set operation
        extrude_dict['set_op'] = entity["operation"]

        # Extrusion offset
        extrude_offset = 0.0
        if entity["start_extent"]["type"] == 'OffsetStartDefinition':
            extrude_offset =  entity["start_extent"]['offset']['value']

        # Extrude values and type [extrude v1, extrude v2, taper value]
        if entity["extent_type"] == 'OneSideFeatureExtentType':
            extrude_direct = entity["extent_one"]["distance"]["role"]
            if extrude_direct == "AlongDistance":
                extrude_direct = 1
            else:
                extrude_direct = -1
            extrude_value = extrude_direct*entity["extent_one"]["distance"]["value"] + extrude_offset
            extrude_value_sorted = sorted([0+extrude_offset, extrude_value])
            extrude_type = 'oneside'
            taper_value = entity["extent_one"]['taper_angle']["value"]
            assert taper_value == 0.0  # do not consider taper for now
        
        elif entity["extent_type"] == 'TwoSidesFeatureExtentType':
            extrude_direct1 = entity["extent_one"]["distance"]["role"]
            if extrude_direct1 == "AlongDistance":
                extrude_direct1 = 1
            else:
                extrude_direct1 = -1
            extrude_direct2 = entity["extent_two"]["distance"]["role"]
            if extrude_direct2 == "AlongDistance":
                extrude_direct2 = 1
            else:
                extrude_direct2 = -1
            v1 = extrude_direct1*entity["extent_one"]["distance"]["value"] + extrude_offset
            v2 = extrude_direct2*entity["extent_two"]["distance"]["value"] + extrude_offset
            extrude_value_sorted = sorted([v1, v2])
            extrude_type = 'twoside'
            v1_tap = entity["extent_one"]["taper_angle"]["value"]
            v2_tap = entity["extent_two"]["taper_angle"]["value"]
            assert v1_tap == 0
            assert v2_tap == 0
            
        elif entity["extent_type"] == 'SymmetricFeatureExtentType':
            extrude_direct = entity["extent_one"]["distance"]["role"]
            if extrude_direct == "AlongDistance":
                extrude_direct = 1
            else:
                extrude_direct = -1
            # half length maybe?
            v1 = extrude_direct*entity["extent_one"]["distance"]["value"] + extrude_offset
            v2 = -extrude_direct*entity["extent_one"]["distance"]["value"] + extrude_offset
            extrude_value_sorted = sorted([v1, v2])
            extrude_type = 'symmetric'

        else:
            raise Exception("Not a valid extrude type")

        extrude_dict['extrude_type'] = extrude_type
        extrude_dict['extrude_values'] = extrude_value_sorted
        return extrude_dict


    def parse_solid(self, extrusion, extrude_params):
        """ 
        Create extrusion for a single sketch-extrude step 
        """
        curve_strings = ""
        curve_count = 0
        sketch_id = []
        p_trans = []
        
        # Same extrusion parameter can be applied to 
        # multiple different closed loops from same sketch
        all_faces = []
        for profile_ids in extrusion["profiles"]: 
            profile_uuid = profile_ids["profile"]
            sketch_uuid = profile_ids["sketch"]
            sketch = self.data["entities"][sketch_uuid]
            profile = sketch["profiles"][profile_uuid]
            
            # Initialize curve string and plane transformation
            p_trans.append({'pt': sketch["transform"]})
            sketch_id.append(sketch_uuid)

            # Create closed loop 
            ref_face, face, curve_string, c_count = self.parse_sketch(sketch, profile)
            curve_strings += curve_string
            curve_count += c_count
            all_faces.append(face)

        # Merge all faces in the same plane 
        plane_face = all_faces[0]
        for face in all_faces[1:]:
            plane_face = self.my_op(plane_face, face, 'fuse')
        
        # Extrude face to 3d shape
        solid = self.extrude_face(ref_face, plane_face, extrude_params)
        
        # Extrude must refer to the same sketch plane 
        if not sketch_id.count(sketch_id[0]) == len(sketch_id):
            tran0 = p_trans[0]
            for tran in p_trans[1:]:
                if not same_plane(tran0, tran):
                    raise Exception("extrude refers to different sketch plane")

        return solid, curve_strings, curve_count, p_trans[0]


    def build_body(self, face, normal, value):
        extrusion_vec = gp_Vec(normal).Multiplied(value)
        make_prism = BRepPrimAPI_MakePrism(face, extrusion_vec)
        make_prism.Build()
        prism = make_prism.Prism()
        return prism.Shape()


    def extrudeBasedOnType(self, face, normal, distance):
        # Extrude based on the two bound values 
        if not (distance[0] < distance[1]):
            raise Exception("incorrect distance")
        large_value = distance[1]
        small_value = distance[0]
        
        if large_value == 0:
            return self.build_body(face, -normal, -small_value) 
        elif small_value == 0:
            return self.build_body(face, normal, large_value)
        elif np.sign(large_value) == np.sign(small_value):
            if large_value < 0:
                body1 = self.build_body(face, -normal, -small_value)
                body2 = self.build_body(face, -normal, -large_value)
                return self.my_op(body1, body2, 'cut')
            else:
                assert large_value > 0
                body1 = self.build_body(face, normal, small_value)
                body2 = self.build_body(face, normal, large_value)
                return self.my_op(body2, body1, 'cut')
        else:
            assert np.sign(large_value) != np.sign(small_value)
            body1 = self.build_body(face, normal, large_value)
            body2 = self.build_body(face, -normal, -small_value)
            return self.my_op(body1, body2, 'fuse')


    def extrude_face(self, ref_face, face, extrude_params):
        distance = extrude_params['extrude_values']
        surf = BRepAdaptor_Surface(ref_face).Plane()
        normal = surf.Axis().Direction()
        extruded_shape = self.extrudeBasedOnType(face, normal, distance)
        return extruded_shape


    def parse_sketch(self, sketch, profile):
        """ 
        Sketch in one closed loop (one out, multiple ins) 
        """
        # Transformation from local to global xyz coord
        transform = get_transform(sketch["transform"])

        # Create face region (automatically infer from all wires)
        curve_count = 0
        string_list = []
        face_list = []
        plane = create_sketch_plane(sketch["transform"])
        
        for idx, pl in enumerate(profile['loops']):
            # Create loop
            loop, curve_string, num_curve = self.parse_loop(pl["profile_curves"], transform)
            
            # Create face for one loop
            face_builder = BRepBuilderAPI_MakeFace(plane, loop)
            if not face_builder.IsDone():
                raise Exception("face builder not done")
            face = face_builder.Face()

            # Fix face 
            fixer = ShapeFix_Face(face)
            fixer.SetPrecision(self.PRECISION)
            fixer.FixOrientation()  

            analyzer = BRepCheck_Analyzer(fixer.Face())
            if not analyzer.IsValid():
                raise Exception("face check failed")

            face_list.append(fixer.Face())
            curve_count += num_curve
            string_list.append(curve_string)
        
        # Return closed sketch face 
        if len(face_list) == 1:
            final_str = "face\n"
            for l_idx, l_str in enumerate(string_list):
                final_str += f"out\n{l_str}\n"
            return face_list[0], face_list[0], final_str, curve_count

        # Find the outer wire (rest becomes inner wires)
        props = GProp_GProps()
        outer_idx = 0
        redo = True
        while redo:
            for f_idx, face in enumerate(face_list):
                # Skip outer face itself
                if f_idx == outer_idx:
                    continue
                # Cut inner face from outer 
                cut_face = self.my_op(face_list[outer_idx], face, 'cut')
                # Compute area, check if inner is larger than outer 
                brepgprop_SurfaceProperties(cut_face, props)
                area = props.Mass()
                if area == 0.0:
                    outer_idx = f_idx 
                    break
            redo = False
       
        # Create final closed loop face 
        inner_idx = list(set(list(range(0, len(face_list)))) - set([outer_idx]))
        inner_faces = [face_list[i] for i in inner_idx]
        final_face = face_list[outer_idx]
        for face in inner_faces:
            final_face = self.my_op(final_face, face, 'cut')

        out_str = ""
        in_str = ""
        # Append inner outer information to string 
        for c_idx, c_str in enumerate(string_list):
            if c_idx == outer_idx:
                out_str += 'out\n' + c_str+'\n'
            else:
                in_str += 'in\n' + c_str+'\n'
        final_str = "face\n" + out_str + in_str 

        return face_list[0], final_face, final_str, curve_count

       
    def parse_loop(self, profile_loop, transform):
        """ Create face in one closed loop  """
        topo_wire = BRepBuilderAPI_MakeWire()
        curve_strings = ''
        curve_count = 0

        # Loop through all the curves in one loop
        for profile_curve in profile_loop:
            # Create curve edge
            curve_edge, curve_string = self.parse_curve(profile_curve, transform)
            if curve_edge == -1: # omitted 
                continue 
            topo_wire.Add(curve_edge)
            if not topo_wire.IsDone():
                raise Exception("wire builder not done")

            curve_string += '\n'
            curve_count += 1
            curve_strings += curve_string 

        fixer = ShapeFix_Wire()
        fixer.Load(topo_wire.Wire())
        fixer.SetPrecision(self.PRECISION)
        fixer.FixClosed()  
        fixer.Perform()

        return fixer.Wire(), curve_strings, curve_count

    
    def parse_curve(self, curve, transform):
        if curve["type"] == "Line3D":
            round_float(curve['start_point'])
            round_float(curve['end_point'])
            return self.create_line(curve, transform)
        elif curve["type"] == "Circle3D":
            return self.create_circle(curve, transform)
        elif curve["type"] == "Arc3D":
            round_float(curve['start_point'])
            round_float(curve['end_point'])
            return self.create_arc(curve, transform)
        else:
            raise Exception("unknown curve type") 


    def create_line(self, line, transform):
        if line["start_point"]["z"] > 0 or line["end_point"]["z"] > 0:
            raise Exception("non-zero z")

        start = create_point(line["start_point"], transform)
        end = create_point(line["end_point"], transform)
        if start.Distance(end)==0:
            raise Exception("start/end point same location")
        
        topo_edge = BRepBuilderAPI_MakeEdge(start, end)
       
        # Save pre-transform 
        start_idx = self.save_vertex(line["start_point"]["x"]+0.0, line["start_point"]["y"]+0.0, 'p')
        end_idx = self.save_vertex(line["end_point"]["x"]+0.0, line["end_point"]["y"]+0.0, 'p')
        curve_string = f"l {start_idx} {end_idx}"
        return topo_edge.Edge(), curve_string


    def create_arc(self, arc, transform):
        if arc["start_point"]["z"] > 0 or arc["end_point"]["z"] > 0 or arc["center_point"]["z"] > 0:
            raise Exception("non-zero z")
        
        start = create_point(arc["start_point"], transform)
        center = create_point(arc["center_point"], transform)
        end = create_point(arc["end_point"], transform)

        mid_raw = self.get_mid_point(arc)
        mid = create_point({'x': mid_raw[0], 'y': mid_raw[1], 'z': 0.0}, transform)
        arc_occ = GC_MakeArcOfCircle(start, mid, end).Value()
        topo_edge = BRepBuilderAPI_MakeEdge(arc_occ)

        # Save pre-transform 
        start_idx = self.save_vertex(arc["start_point"]["x"]+0.0, arc["start_point"]["y"]+0.0, 'p')
        end_idx = self.save_vertex(arc["end_point"]["x"]+0.0, arc["end_point"]["y"]+0.0, 'p')
        center_idx = self.save_vertex(arc["center_point"]["x"]+0.0, arc["center_point"]["y"]+0.0, 'p')
        mid_idx = self.save_vertex(mid_raw[0]+0.0, mid_raw[1]+0.0, 'p')
        curve_string = f"a {start_idx} {mid_idx} {center_idx} {end_idx}"

        return topo_edge.Edge(), curve_string


    def get_mid_point(self, curve):
        ref_vec = np.array([curve['reference_vector']['x'],
                            curve['reference_vector']['y']])
        center = np.array([curve['center_point']['x'],
                           curve['center_point']['y']])
        radius = curve['radius']
        start_angle = curve['start_angle']
        end_angle = curve['end_angle']
        mid_angle = (start_angle + end_angle) / 2
        rot_mat = np.array([[np.cos(mid_angle), -np.sin(mid_angle)],
                            [np.sin(mid_angle), np.cos(mid_angle)]])
        mid_vec = rot_mat @ ref_vec
        return center + mid_vec * radius
    
    
    def create_circle(self, circle, transform, ref_vector3d=None):
        if circle["center_point"]["z"] > 0:
            raise Exception("non-zero z")

        center = create_point(circle["center_point"], transform)
        radius = circle["radius"]
        normal = create_unit_vec({"x": 0.0, "y": 0.0, "z": 1.0}, transform)
        if ref_vector3d is None:
            ref_vector3d = self.x_axis.Transformed(transform)
        axis = gp_Ax2(center, normal, ref_vector3d)

        gp_circle = gp_Circ(axis, abs(float(radius)))
        topo_edge = BRepBuilderAPI_MakeEdge(gp_circle)
        
        center_idx = self.save_vertex(circle["center_point"]["x"]+0.0, circle["center_point"]["y"]+0.0, 'p')
        radius_idx = self.save_vertex(abs(float(radius))+0.0, 0, 'r')
        curve_string = f"c {center_idx} {radius_idx}"
        return topo_edge.Edge(), curve_string


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





class OBJReconverter:
    """ OBJ Data Reconverter """
    
    def __init__(self):
        self.vertex_dict = OrderedDict()
        self.PRECISION = 1e-5
        self.eps = 1e-7
        self.x_axis = gp_Dir(1.0, 0.0, 0.0)


    def convert_curve(self, curve):
        """
        convert to json dict format 
        """
        json_curve = {}

        if curve.type == 'circle':
            json_curve['type'] = 'Circle3D'
            json_curve["center_point"] = {'x': curve.center[0], 'y': curve.center[1], 'z': 0}
            json_curve['radius'] = curve.radius
        
        if curve.type == 'line':
            json_curve['type'] = 'Line3D'
            json_curve["start_point"] = {'x': curve.start[0], 'y': curve.start[1], 'z': 0}
            json_curve["end_point"] = {'x': curve.end[0], 'y': curve.end[1], 'z': 0}
        
        if curve.type == 'arc':
            json_curve['type'] = 'Arc3D'
            json_curve["start_point"] = {'x': curve.start[0], 'y': curve.start[1], 'z': 0}
            json_curve["end_point"] = {'x': curve.end[0], 'y': curve.end[1], 'z': 0}
            json_curve["mid_point"] = {'x': curve.mid[0], 'y': curve.mid[1], 'z': 0}
            json_curve["center_point"] = {'x': curve.center[0], 'y': curve.center[1], 'z': 0}

        json_curve['is_outer'] = curve.is_outer
        return json_curve


    def convert_vertices(self):
        """ Convert all the vertices to .obj format """
        vertex_strings = ""
        for pt in self.vertex_dict.values():
            # e.g. v 0.123 0.234 0.345 1.0
            vertex_string = f"v {pt[0]} {pt[1]}\n"
            vertex_strings += vertex_string
        return vertex_strings


    def parse_obj(self, faces, meta_info):
        
        """
        reconstruct brep from obj file 
        """
        # At least one needs to match 
        for face in faces:
            for loop in face:
                if len(loop) > 1:
                    for idx, curve in enumerate(loop[:-1]):
                        next_curve = np.vstack([loop[idx+1].start, loop[idx+1].end])
                        diff1 = np.sum(np.abs(curve.start - next_curve), 1)
                        diff2 = np.sum(np.abs(curve.end - next_curve), 1)

                        if min(diff2) ==0 or min(diff1) == 0:
                            continue # edge connected 

                        assert min(diff1) < 1e-3 or min(diff2) < 1e-3 # difference should be small

                        if min(diff1) > min(diff2):
                            min_idx = np.argmin(diff2)
                            if min_idx == 0:
                                loop[idx+1].start_idx = curve.end_idx 
                                loop[idx+1].start = curve.end 
                            else:
                                loop[idx+1].end_idx = curve.end_idx 
                                loop[idx+1].end = curve.end
                        else:
                            min_idx = np.argmin(diff1)
                            if min_idx == 0:
                                loop[idx+1].start_idx = curve.start_idx 
                                loop[idx+1].start = curve.start 
                            else:
                                loop[idx+1].end_idx = curve.start_idx 
                                loop[idx+1].end = curve.start

                    # Solve start / end connection
                    shared_idx = list(set([loop[-2].start_idx, loop[-2].end_idx]).intersection(
                                      set([loop[-1].start_idx, loop[-1].end_idx])
                                 ))

                    assert len(shared_idx) >= 1

                    if len(shared_idx) == 2:
                        assert len(loop) == 2 # do nothing 
                    else:
                        if shared_idx[0] == loop[-1].start_idx:
                            do_start = False 
                        else:
                            do_start = True 
                        start_curve = np.vstack([loop[0].start, loop[0].end])

                        if do_start:
                            diff = np.sum(np.abs(loop[-1].start - start_curve), 1)   
                        else: 
                            diff = np.sum(np.abs(loop[-1].end - start_curve), 1)   
                        assert min(diff) < 1e-3 

                        min_idx = np.argmin(diff)
                        if min_idx == 0:
                            if do_start:
                                loop[-1].start_idx = loop[0].start_idx 
                                loop[-1].start = loop[0].start 
                            else:
                                loop[-1].end_idx = loop[0].start_idx 
                                loop[-1].end = loop[0].start 
                        else:
                            if do_start:
                                loop[-1].start_idx = loop[0].end_idx 
                                loop[-1].start = loop[0].end 
                            else:
                                loop[-1].end_idx = loop[0].end_idx 
                                loop[-1].end = loop[0].end 

        # Parse groups to json loop/curve profile
        extrusion = {}
        extrusion['profiles'] = []
        for face in faces:
            profile = {}
            profile['loops'] = []
            for loop in face:
                pl = {}
                pl['profile_curves'] = []
                for curve in loop:
                    # convert to json format
                    pl['profile_curves'].append(self.convert_curve(curve))
                profile['loops'].append(pl)
            extrusion['profiles'].append(profile)
        
        # Parse transform 
        sketch = {}
        transform = {}
        transform['origin'] = {'x': meta_info['t_orig'][0], 'y': meta_info['t_orig'][1], 'z': meta_info['t_orig'][2]}
        transform['x_axis'] = {'x': meta_info['t_x'][0], 'y': meta_info['t_x'][1], 'z': meta_info['t_x'][2]}
        transform['y_axis'] = {'x': meta_info['t_y'][0], 'y': meta_info['t_y'][1], 'z': meta_info['t_y'][2]}
        transform['z_axis'] = {'x': meta_info['t_z'][0], 'y': meta_info['t_z'][1], 'z': meta_info['t_z'][2]}
        sketch['transform'] = transform 

        # Parse extrude 
        extrude_params = {}
        extrude_params['extrude_type'] = meta_info['set_op']
        extrude_params['extrude_values'] = meta_info['extrude_value']

        # Create sketch 
        all_faces = []
        curve_strings = ""
        curve_count = 0
        for profile in extrusion["profiles"]: 
            ref_face, face, curve_string, c_count = self.parse_sketch(sketch, profile)
            curve_strings += curve_string
            curve_count += c_count
            all_faces.append(face)
        
        # Merge all faces in the same plane 
        plane_face = all_faces[0]
        for face in all_faces[1:]:
            plane_face = self.my_op(plane_face, face, 'fuse')
        solid = self.extrude_face(ref_face, plane_face, extrude_params)
        return solid, curve_strings, curve_count


    def my_op(self, big, small, op_name): 
        if op_name == 'cut':
            op = BRepAlgoAPI_Cut(big, small) 
        elif op_name == 'fuse':
            op = BRepAlgoAPI_Fuse(big, small)
        elif op_name == 'common':
            op = BRepAlgoAPI_Common(big, small)
        op.SetFuzzyValue(self.PRECISION) 
        op.Build() 
        return op.Shape()
            

    def build_body(self, face, normal, value):
        extrusion_vec = gp_Vec(normal).Multiplied(value)
        make_prism = BRepPrimAPI_MakePrism(face, extrusion_vec)
        make_prism.Build()
        prism = make_prism.Prism()
        return prism.Shape()


    def extrudeBasedOnType(self, face, normal, distance):
        # Extrude based on the two bound values 
        if not (distance[0] < distance[1]):
            raise Exception("incorrect distance")
        large_value = distance[1]
        small_value = distance[0]
        
        if large_value == 0:
            return self.build_body(face, -normal, -small_value) 
        elif small_value == 0:
            return self.build_body(face, normal, large_value)
        elif np.sign(large_value) == np.sign(small_value):
            if large_value < 0:
                body1 = self.build_body(face, -normal, -small_value)
                body2 = self.build_body(face, -normal, -large_value)
                return self.my_op(body1, body2, 'cut')
            else:
                assert large_value > 0
                body1 = self.build_body(face, normal, small_value)
                body2 = self.build_body(face, normal, large_value)
                return self.my_op(body2, body1, 'cut')
        else:
            assert np.sign(large_value) != np.sign(small_value)
            body1 = self.build_body(face, normal, large_value)
            body2 = self.build_body(face, -normal, -small_value)
            return self.my_op(body1, body2, 'fuse')


    def extrude_face(self, ref_face, face, extrude_params):
        distance = extrude_params['extrude_values']
        surf = BRepAdaptor_Surface(ref_face).Plane()
        normal = surf.Axis().Direction()
        extruded_shape = self.extrudeBasedOnType(face, normal, distance)
        return extruded_shape


    def parse_sketch(self, sketch, profile):
        """ 
        Sketch in one closed loop (one out, multiple ins) 
        """
        # Transformation from local to global xyz coord
        transform = get_transform(sketch["transform"])

        # Create face region (automatically infer from all wires)
        outer_facelist = []
        inner_facelist = []
        curve_count = 0
        outer_string = []
        inner_string = []
        plane = create_sketch_plane(sketch["transform"])
        
        for idx, pl in enumerate(profile['loops']):
            # Create loop
            loop, curve_string, num_curve = self.parse_loop(pl["profile_curves"], transform)
            # Create face
            face_builder = BRepBuilderAPI_MakeFace(plane, loop)
            if not face_builder.IsDone():
                raise Exception("face builder not done")
            face = face_builder.Face()
            # Fix face 
            fixer = ShapeFix_Face(face)
            fixer.SetPrecision(self.PRECISION)
            fixer.FixOrientation()  
    
            analyzer = BRepCheck_Analyzer(fixer.Face())
            if not analyzer.IsValid():
                raise Exception("face check failed")

            curve_count += num_curve

            if pl["profile_curves"][0]['is_outer']:
                outer_facelist.append(fixer.Face())
                outer_string.append(curve_string)
            else:
                inner_facelist.append(fixer.Face())
                inner_string.append(curve_string)
        
        # Create final closed loop face 
        assert len(outer_facelist) > 0
        final_face = outer_facelist[0]
        for face in outer_facelist[1:]:
            final_face = self.my_op(final_face, face, 'fuse')
        for face in inner_facelist:
            final_face = self.my_op(final_face, face, 'cut')


        # Append inner outer information to string 
        assert len(outer_string) == 1
        out_str = ""
        in_str = ""
        for c_str in outer_string:
            out_str += 'out\n' + c_str+'\n'
        for c_str in inner_string:
            in_str += 'in\n' + c_str+'\n'
        final_str = "face\n" + out_str + in_str 
    
        return outer_facelist[0], final_face, final_str, curve_count


    def parse_loop(self, profile_loop, transform):
        """ Create face in one closed loop  """
        topo_wire = BRepBuilderAPI_MakeWire()
        curve_strings = ''
        curve_count = 0

        # Loop through all the curves in one loop
        for profile_curve in profile_loop:
            curve_edge, curve_string = self.parse_curve(profile_curve, transform)
            topo_wire.Add(curve_edge)
            if not topo_wire.IsDone():
                raise Exception("wire builder not done")

            curve_string += '\n'
            curve_count += 1
            curve_strings += curve_string 

        fixer = ShapeFix_Wire()
        fixer.Load(topo_wire.Wire())
        fixer.SetPrecision(self.PRECISION)
        fixer.FixClosed()  
        fixer.Perform()
        return fixer.Wire(), curve_strings, curve_count


    def parse_curve(self, curve, transform):
        if curve["type"] == "Line3D":
            return self.create_line(curve, transform)
        elif curve["type"] == "Circle3D":
            return self.create_circle(curve, transform)
        elif curve["type"] == "Arc3D":
            return self.create_arc(curve, transform)
        else:
            raise Exception("unknown curve type") 


    def create_line(self, line, transform):
        start = create_point(line["start_point"], transform)
        end = create_point(line["end_point"], transform)
        if start.Distance(end)==0:
            raise Exception("start/end point same location")
        topo_edge = BRepBuilderAPI_MakeEdge(start, end)

        # Save pre-transform 
        star_idx = self.save_vertex(line["start_point"]["x"]+0.0, line["start_point"]["y"]+0.0, 'p')
        end_idx = self.save_vertex(line["end_point"]["x"]+0.0, line["end_point"]["y"]+0.0, 'p')
        curve_string = f"l {star_idx} {end_idx}"
        return topo_edge.Edge(), curve_string


    def create_arc(self, arc, transform):
        start = create_point(arc["start_point"], transform)
        mid = create_point(arc["mid_point"], transform)
        end = create_point(arc["end_point"], transform)
        arc_occ = GC_MakeArcOfCircle(start, mid, end).Value()
        topo_edge = BRepBuilderAPI_MakeEdge(arc_occ)

        # Save pre-transform 
        start_idx = self.save_vertex(arc["start_point"]["x"]+0.0, arc["start_point"]["y"]+0.0, 'p')
        end_idx = self.save_vertex(arc["end_point"]["x"]+0.0, arc["end_point"]["y"]+0.0, 'p')
        center_idx = self.save_vertex(arc["center_point"]["x"]+0.0, arc["center_point"]["y"]+0.0, 'p')
        mid_idx = self.save_vertex(arc["mid_point"]["x"]+0.0, arc["mid_point"]["y"]+0.0, 'p')
        curve_string = f"a {start_idx} {mid_idx} {center_idx} {end_idx}"
        return topo_edge.Edge(), curve_string
    
    
    def create_circle(self, circle, transform):
        center = create_point(circle["center_point"], transform)
        radius = circle["radius"]
        normal = create_unit_vec({"x": 0.0, "y": 0.0, "z": 1.0}, transform)
        ref_vector3d = self.x_axis.Transformed(transform)
        axis = gp_Ax2(center, normal, ref_vector3d)
        gp_circle = gp_Circ(axis, abs(float(radius)))
        topo_edge = BRepBuilderAPI_MakeEdge(gp_circle)

        center_idx = self.save_vertex(circle["center_point"]["x"]+0.0, circle["center_point"]["y"]+0.0, 'p')
        radius_idx = self.save_vertex(abs(float(radius))+0.0, 0, 'r')
        curve_string = f"c {center_idx} {radius_idx}"
        return topo_edge.Edge(), curve_string


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



class SketchPoint:
    """
    Class representing a point in a sketch with an index
    Used to weld curve points together
    """    
    def __init__(self, x=0.0, y=0.0, json_data=None):
        if json_data:
            self.x = json_data["x"]
            self.y = json_data["y"]
        else:
            self.x = x
            self.y = y
        self.index = None
