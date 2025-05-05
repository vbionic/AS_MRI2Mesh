#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import logging
import numpy as np
import xml.etree.ElementTree as ET
from trimesh.primitives import Sphere, Cylinder
from as_bin.utils.affine_transformation import H4x4_decomposition, H4x4_composition
from trimesh.transformations import angle_between_vectors, rotation_matrix
import pathlib

#----------------------------------------------------------------------------
_MLP_h1_MeshLabProject = 'MeshLabProject'
_MLP_h2_MeshGroup      = 'MeshGroup'
_MLP_h3_MLMesh         = 'MLMesh'
_MLP_h4_MLMatrix44     = 'MLMatrix44'
_MLP_h4_RenderingOption= 'RenderingOption'
#----------------------------------------------------------------------------
def xml_root_indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            xml_root_indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

#----------------------------------------------------------------------------
def mlp_parse(pth):
    meshes = []
    mlp_tree = ET.parse(pth)
    mlp_h1_MeshLabProject = mlp_tree.getroot()
    for h2_element in mlp_h1_MeshLabProject:
        if h2_element.tag == _MLP_h2_MeshGroup:
            mlp_h2_MeshGroup = h2_element
            for h3_element in mlp_h2_MeshGroup:
                if h3_element.tag == _MLP_h3_MLMesh:
                    mlp_h3_MLMesh = h3_element
                    #logging.info(f"{mlp_h3_MLMesh.attrib['label']} at {mlp_h3_MLMesh.attrib['filename']}") 
                    mlp_h4_MLMatrix44 = np.eye(4)
                    for h4_element in mlp_h3_MLMesh:
                        if h4_element.tag == _MLP_h4_MLMatrix44:
                            mlp_h4_MLMatrix44_str = h4_element.text
                            mlp_h4_MLMatrix44_tab = mlp_h4_MLMatrix44_str.split()
                            mlp_h4_MLMatrix44 = [float(v) for v in mlp_h4_MLMatrix44_tab]
                            mlp_h4_MLMatrix44 = np.array(mlp_h4_MLMatrix44) 
                            mlp_h4_MLMatrix44 = np.reshape(mlp_h4_MLMatrix44, (4,4))
                    affin_transform_matrices = H4x4_decomposition(mlp_h4_MLMatrix44)
                    mesh_dict = {"filename": mlp_h3_MLMesh.attrib["filename"],
                                "label": mlp_h3_MLMesh.attrib["label"],
                                "H": mlp_h4_MLMatrix44,
                                "affin_transform_matrices": affin_transform_matrices
                                }
                    meshes.append(mesh_dict)
                else:
                    continue
        else:#if h2_element.tag == "RasterGroup":
            continue
    return meshes

#----------------------------------------------------------------------------
def mlp_create(meshes_dict, work_dir):#mshs_pth, points, default_primitive_poin_stl_pth):
    mlp_h1_MeshLabProject = ET.Element(_MLP_h1_MeshLabProject)
    #mlp_h1_MeshLabProject.text = "\n "
    mlp_h2_MeshGroup      = ET.SubElement(mlp_h1_MeshLabProject, _MLP_h2_MeshGroup)
    #mlp_h2_MeshGroup.text = "\n  "
    for meshe_dict in meshes_dict:
        mlp_h3_MLMesh      = ET.SubElement(mlp_h2_MeshGroup, _MLP_h3_MLMesh)
        #mlp_h3_MLMesh.text = "\n   "
        meshe_dict_keys = list(meshe_dict.keys())
        if "filename" in meshe_dict_keys:
            abs_pth = os.path.abspath(meshe_dict["filename"])
            #relative = os.path.relpath(meshe_dict["filename"])
            wd_rel_pth = os.path.relpath(abs_pth, start = work_dir)
            wd_rel_pth = wd_rel_pth.replace('\\','/')
            mlp_h3_MLMesh.attrib["filename"] = './' + wd_rel_pth
        else:
            mlp_h3_MLMesh.attrib["filename"] = "unkn"
        if "label" in meshe_dict_keys:
            mlp_h3_MLMesh.attrib["label"] = meshe_dict["label"]
        else:
            mlp_h3_MLMesh.attrib["label"] = mlp_h3_MLMesh.attrib["filename"].split(".")[0]
        if "visible" in meshe_dict_keys:
            mlp_h3_MLMesh.attrib["visible"] = meshe_dict["visible"]
        if 'affin_transform_matrices' in meshe_dict_keys:
            atms = meshe_dict["affin_transform_matrices"]
            mdks = atms.keys()
            T4x4 = atms["T4x4"] if "T4x4" in mdks else None
            R4x4 = atms["R4x4"] if "R4x4" in mdks else None
            K4x4 = atms["K4x4"] if "K4x4" in mdks else None
            T3   = atms["T3"  ] if "T3"   in mdks else None
            R3x3 = atms["R3x3"] if "R3x3" in mdks else None
            K3x3 = atms["K3x3"] if "K3x3" in mdks else None
            H = H4x4_composition(T4x4=T4x4, T3=T3, R4x4=R4x4, R3x3=R3x3, K4x4=K4x4, K3x3=K3x3)
            if np.count_nonzero(H - np.eye(4)) != 0:
                Hstr = ""
                for ii in range(4):
                    Hstr += f"\n"
                    for jj in range(4):
                        if H[ii,jj]==0.0:
                            Hstr += f"0 "
                        elif H[ii,jj]==1.0:
                            Hstr += f"1 " 
                        else:
                            Hstr += f"{H[ii,jj]} " 
                Hstr += f"\n"
                mlp_h4_MLMatrix44      = ET.SubElement(mlp_h3_MLMesh, _MLP_h4_MLMatrix44)
                mlp_h4_MLMatrix44.text = Hstr

        mlp_h4_RenderingOption = ET.SubElement(mlp_h3_MLMesh, _MLP_h4_RenderingOption)
        if "RenderingOptions_text" in meshe_dict_keys:
            mlp_h4_RenderingOption.text = meshe_dict["RenderingOptions_text"]
        for k in meshe_dict_keys:
            if k.find("Color") != -1:
                mlp_h4_RenderingOption.attrib[k] = meshe_dict[k]
        #mlp_h3_MLMesh.tag 
        #mlp_h3_MLMesh.text
    tree = ET.ElementTree(mlp_h1_MeshLabProject)
    return tree

#----------------------------------------------------------------------------
def normalize_file_lineendings_toLF(pth):
    # replacement strings
    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'

    with open(pth, 'rb') as open_file:
        content = open_file.read()
        
    # Windows => Unix
    content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

    with open(pth, 'wb') as open_file:
        open_file.write(content)

#----------------------------------------------------------------------------
def export_meshlab_mlp(pth, mshs_pth=None, primitives_dicts = None, 
                       default_primitive_poin_stl_pth = None, default_primitive_segment_stl_pth = None):
    meshes_dict = []
    if not mshs_pth is None:
        for m_pth_label, m_pth in mshs_pth.items():
            meshe_dict = {
                "filename": m_pth,
                "label": m_pth_label,
                "visible": "1" if (m_pth_label.find("bone")!=-1) else "0",
                #"RenderingOptions_text":"010001000000000000000010000001010100000010100000010010111011110000001001",
                "RenderingOptions_text":"010001000000000000000010000001000100000010100000010010111011110000001001",
                "wireColor": "64 64 64 255",
                "solidColor": "192 192 192 128"
            }
            meshes_dict.append(meshe_dict)

    if default_primitive_poin_stl_pth is None:
        point_stl_fn = f"_ball1.stl"
        sph1 = Sphere(radius = 1.0, center = [0,0,0])
        default_primitive_poin_stl_pth = os.path.abspath(point_stl_fn)
        sph1.export(default_primitive_poin_stl_pth)
                
    if default_primitive_segment_stl_pth is None:
        stl_fn = f"_cylinder1.stl"
        cyl1 = Cylinder(radius = 1.0, height = 1, sections = 4)
        default_primitive_segment_stl_pth = os.path.abspath(stl_fn)
        cyl1.export(default_primitive_segment_stl_pth)
                
    if (not primitives_dicts is None):
        for series_dict in primitives_dicts:
            color   = "100 100 100 255"            if (not "color"   in series_dict.keys()) else series_dict["color"]
            scale   = 1                            if (not "scale"   in series_dict.keys()) else series_dict["scale"]
            dz      = None                         if (not "dz"      in series_dict.keys()) else series_dict["dz"]
            dx      = 1                            if (not "dy"      in series_dict.keys()) else series_dict["dy"]
            dy      = 1                            if (not "dx"      in series_dict.keys()) else series_dict["dx"]
            visible = "1"                          if (not "visible" in series_dict.keys()) else series_dict["visible"]
            is_wireframe = None                   if (not "is_wireframe" in series_dict.keys()) else series_dict["is_wireframe"]
            is_solid     = None                   if (not "is_solid" in series_dict.keys()) else series_dict["is_solid"]
            for label, value in series_dict["named_data_serie"].items():
                if type(value) is float:
                    continue
                is_point           = len(value) == 3
                is_sphere          = len(value) == 4
                is_segment         = len(value) == 6
                is_segment_scalled = len(value) == 9
                is_full            = len(value) == 12
                if not is_point and not is_sphere and not is_segment and not is_segment_scalled and not is_full:
                    continue
                T3   = value[0:3]
                if is_segment or is_segment_scalled or is_full:
                    if ("filename" in series_dict.keys()):
                        filename = series_dict["filename"]
                    else:
                        filename = default_primitive_segment_stl_pth
                    curr_dir = [0,0,1]
                    req_dir  = value[3:6]
                    rot_dir  = np.cross(curr_dir, req_dir) 
                    a = angle_between_vectors(curr_dir, req_dir)
                    if not (np.isnan(a) or a==0.0):
                        R4x4 = rotation_matrix(a, rot_dir)
                    else:
                        R4x4 = np.eye(4)
                    if is_segment:
                        dz = np.linalg.norm(req_dir)
                    elif is_segment_scalled or is_full:
                        dx, dy, dz = value[6:9]

                    if is_full:
                        
                        org_dir = [1,0,0]
                        R3x3 = R4x4[:3,:3]
                        curr_dir = np.dot(R3x3, org_dir)
                        req_dir  = value[9:12]
                        rot_dir  = np.cross(curr_dir, req_dir) 
                        a = angle_between_vectors(curr_dir, req_dir)
                        if not (np.isnan(a) or a==0.0):
                            R4x4_new = rotation_matrix(a, rot_dir)
                        else:
                            R4x4_new = np.eye(4)
                        R4x4 = R4x4_new @ R4x4
                elif is_point or is_sphere:
                    if ("filename" in series_dict.keys()):
                        filename = series_dict["filename"]
                    else:
                        filename= default_primitive_poin_stl_pth
                    R4x4 = np.eye(4)
                    if dz is None:
                        dz = 1
                    if is_sphere:
                        scale *= value[3]
                wireframe_cfg_str  = "010001000000000000000010000001000100000010100000010010111011110000001001"
                solid_cfg_str      = "100001000000000000000000000001010100000010100000000100111011110000001001"
                RenderingOptions_text = solid_cfg_str
                if not is_solid is None:
                    RenderingOptions_text = solid_cfg_str if is_solid else wireframe_cfg_str
                elif not is_wireframe is None:
                    RenderingOptions_text = solid_cfg_str if not is_wireframe else wireframe_cfg_str

                K3x3= np.eye(3)
                K3x3[2,2] = dz
                K3x3[1,1] = dx
                K3x3[0,0] = dy
                K3x3 *= scale

                meshe_dict = {
                    "filename": filename,
                    "label": label,
                    "RenderingOptions_text":RenderingOptions_text,
                    "solidColor": color,
                    "affin_transform_matrices": {"T3": T3, "K3x3": K3x3, "R4x4": R4x4},
                    "visible": visible
                }
                meshes_dict.append(meshe_dict)
    work_dir = pathlib.Path(pth).parent
    mlp_obj = mlp_create(meshes_dict, work_dir)
    xml_root_indent(mlp_obj.getroot())
    mlp_obj.write(pth, encoding="utf-8", xml_declaration=True)
    normalize_file_lineendings_toLF(pth)

#----------------------------------------------------------------------------
def try_find_valid_pth_from_mlp_pth(pth_in, src_mlp_pth = None):
    if os.path.isfile(pth_in):
        return pth_in
    else:
        if not src_mlp_pth is None:
            src_mlp_dir = pathlib.Path(src_mlp_pth).parent
            pth_r1 = os.path.join(src_mlp_dir, pth_in)
            pth_s = os.path.relpath(pth_r1, start = pathlib.Path.cwd())
            if os.path.isfile(pth_s):
                logging.warning(f"  Found a file at path {pth_s}. Use it")
                return pth_s
        
        pth_in = os.path.normpath(pth_in)
        pth_l = pth_in.split(os.sep)
        for sid in range(len(pth_l)):
            pth_s = os.path.join(*pth_l[sid:])
            if os.path.isfile(pth_s):
                logging.warning(f"  Found a file at path {pth_s}. Use it")
                return pth_s
        logging.error(f"  Did not found a valid file for {pth_in}. Give up")
        sys.exit(1)