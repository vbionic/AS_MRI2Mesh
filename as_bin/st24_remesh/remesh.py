"""
Import a mesh file and convert it to tetras.
"""

import numpy as np
import copy
import math
from argparse import ArgumentParser
import os, sys
import pathlib
import glob
import time
from time import gmtime, strftime
from datetime import datetime
import logging
from pandas.core.common import flatten
import json
import subprocess
import multiprocessing
import meshio
import pyvista as pv
import pyacvd
import scipy
from scipy.signal import decimate, resample_poly

import trimesh
from trimesh.voxel import creation

#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_arg import arg2boolAct
from v_utils.v_json import jsonUpdate, jsonDumpSafe
from v_utils.v_arg import convert_dict_to_cmd_line_args, convert_cmd_line_args_to_dict, convert_cfg_files_to_dicts
from v_utils.v_arg import print_cfg_list, print_cfg_dict

from as_bin.utils.dir_utils import createDirIfNeeded
from as_bin.utils.mesh_utils import mesh_repair, adjust_z_at_top
#-----------------------------------------------------------------------------------------
    
def write_MESH_file(points:np.array, tetras:np.array, tetras_types:np.array, out_pth, points_are_indexed_from_zero = True, filter_tetras_types = None):
    with open(out_pth, 'w', newline='\n') as f:
        points_start_str  = f"MeshVersionFormatted 2\n"
        points_start_str += f"Dimension 3\n"
        points_start_str += f"Vertices\n"
        points_start_str += f"{len(points)}\n"
        f.writelines(points_start_str)

        for p in points:
            pl = F"{p[0]:.16e} {p[1]:.16e} {p[2]:.16e} 1\n"
            f.writelines(pl)
   
        tet_start_str  = f"Tetrahedra\n"
        tet_start_str += f"{len(tetras)}\n"
        f.writelines(tet_start_str)

        if not filter_tetras_types is None:
            if not (type(filter_tetras_types) is list or type(filter_tetras_types) is np.array):
                filter_tetras_types = [filter_tetras_types]
            filtered_tetras_types = np.empty(shape=[0, ])
            filtered_tetras       = np.empty(shape=[0,4], dtype = int)
            for tt in filter_tetras_types:
                filtered_tetras_ids   = np.where(tetras_types == tt)
                filtered_tetras_types = np.append(filtered_tetras_types, tetras_types[filtered_tetras_ids], axis=0)
                filtered_tetras       = np.append(filtered_tetras      , tetras      [filtered_tetras_ids], axis=0)
        else:
            filtered_tetras_types = tetras_types
            filtered_tetras       = tetras
                

        # points_are_indexed_from_zero = np.min(filtered_tetras)==0
        if points_are_indexed_from_zero:
            for tid, t in enumerate(filtered_tetras):
                tp = t+1
                ttype = filtered_tetras_types[tid]
                tl = F"{tp[0]:d} {tp[1]:d} {tp[2]:d} {tp[3]:d} {ttype}\n"
                f.writelines(tl)
        else:
            for tid, t in enumerate(filtered_tetras):
                ttype = filtered_tetras_types[tid]
                tl = F"{t[0]:d} {t[1]:d} {t[2]:d} {t[3]:d} {ttype}\n"
                f.writelines(tl)
        ## dodanie tych lini wywala GMSH - te linie sa interpretowane jako kolejne czworosciany! Tylko indeksy wierzcholkow sa rowne indeksom z ostatniego czworoscianu minus ofset liczony w liniach!!!
        #end_str  = f"\n"    
        #end_str += f"End\n"
        #f.writelines(end_str)
        
#-----------------------------------------------------------------------------------------
def save_voxelgrid_as_json(out_pth, vx_tm):
    vx_dict = {
        "sparse_indices":   [[int(v) for v in vs] for vs in vx_tm.sparse_indices],
        #"values":           [[int(v) for v in vs] for vs in vx_tm.sparse_indices],
        "translation":      list(vx_tm.translation),
        "scale":            list(vx_tm.scale)
        }
    jsonDumpSafe(out_pth, vx_dict, do_delete_before_dump = True)

def load_voxelgrid_from_json(out_pth):
    with open(out_pth, 'r') as f:
        vx_dict = json.load(f)
        result_VG = trimesh.voxel.base.VoxelGrid(
            #trimesh.voxel.encoding.SparseBinaryEncoding(rem_matrix),# shape = (26, 26, 26)),
            trimesh.voxel.encoding.SparseBinaryEncoding(np.array(vx_dict["sparse_indices"], dtype=int)),
            transform = trimesh.transformations.scale_and_translate(
                scale = vx_dict["scale"], translate = vx_dict["translation"]))
        return result_VG
    
#-----------------------------------------------------------------------------------------
def voxelize_subdivide(mesh, pitch, max_iter=10, edge_factor=2.0):
    """
    Voxelize a surface by subdividing a mesh until every edge is
    shorter than: (pitch / edge_factor)

    Parameters
    -----------
    mesh:        Trimesh object
    pitch:       float, side length of a single voxel cube
    max_iter:    int, cap maximum subdivisions or None for no limit.
    edge_factor: float,

    Returns
    -----------
    VoxelGrid instance representing the voxelized mesh.
    """
    max_edge = pitch / edge_factor

    if max_iter is None:
        longest_edge = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] -
                                      mesh.vertices[mesh.edges[:, 1]],
                                      axis=1).max()
        max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)

    # get the same mesh sudivided so every edge is shorter
    # than a factor of our pitch
    if type(mesh) is meshio._mesh.Mesh:
        v, f = trimesh.remesh.subdivide_to_size(mesh.points,
                                    mesh.cells_dict['triangle'],
                                    max_edge=max_edge,
                                    max_iter=max_iter)
    else:
        v, f = trimesh.remesh.subdivide_to_size(mesh.vertices,
                                    mesh.faces,
                                    max_edge=max_edge,
                                    max_iter=max_iter)

    # convert the vertices to their voxel grid position
    hit = v / pitch

    # Provided edge_factor > 1 and max_iter is large enough, this is
    # sufficient to preserve 6-connectivity at the level of voxels.
    hit = np.round(hit).astype(int)

    # remove duplicates
    unique, inverse = trimesh.grouping.unique_rows(hit)

    # get the voxel centers in model space
    occupied_index = hit[unique]

    origin_index = occupied_index.min(axis=0)
    origin_position = origin_index * pitch

    return trimesh.voxel.base.VoxelGrid(
            trimesh.voxel.encoding.SparseBinaryEncoding(occupied_index - origin_index),# shape = (26, 26, 26)),
            transform = trimesh.transformations.scale_and_translate(scale=pitch, translate=origin_position)
        )

def read_tissues(in_msh_pths):
    
    in_tiss_mshs_tot = {}
    for in_msh_pth in in_msh_pths:
        in_tiss_mshs = {}
        in_msh = trimesh.load(in_msh_pth)

        if(in_msh_pth.find("roi") != -1):
            in_tiss_mshs["roi"] = None
        elif(in_msh_pth.find("skin") != -1):
            in_tiss_mshs["skin"] = None
        if(in_msh_pth.find("bones") != -1):
            in_tiss_mshs["bones"] = None
        if(in_msh_pth.find("vessels") != -1):
            in_tiss_mshs["vessels"] = None
        if(len(in_tiss_mshs) == 0):
            in_tiss_mshs["undef"] = None

        if type(in_msh) is trimesh.Scene:
            t_raw_names  = list(in_msh.geometry.keys())
            geoms       = [in_msh.geometry[t_raw_name] for t_raw_name in t_raw_names]
            t_mass      = [tiss_geom.mass                    for tiss_geom in geoms]
            t_bodies_n  = [tiss_geom.body_count              for tiss_geom in geoms]
            bb_volum    = [tiss_geom.bounding_box.volume     for tiss_geom in geoms] 	
            t_colour    = [tiss_geom.visual.material.ambient for tiss_geom in geoms] 	

            if("skin" in in_tiss_mshs.keys()):
                skin_id = np.argmax(bb_volum)
                in_tiss_mshs["skin"] = geoms[skin_id]
        
            if("vessels" in in_tiss_mshs.keys()):
                t_redness = [float(tc[0])/(float(tc[1])+float(tc[2])) for tc in t_colour]
                vessels_id = np.argmax(t_redness)
                in_tiss_mshs["vessels"] = geoms[vessels_id]
            
            if("bones" in in_tiss_mshs.keys()):
                t_bones = [abs(tc[0]-182) + abs(tc[1]-190)  + abs(tc[0]-194) for tc in t_colour]
                bones_id = np.argmin(t_bones)
                in_tiss_mshs["bones"] = geoms[bones_id]
                
            if("roi" in in_tiss_mshs.keys()):
                skin_id = np.argmax(bb_volum)
                in_tiss_mshs["roi"] = geoms[skin_id]

            if("undef" in in_tiss_mshs.keys()):
                in_tiss_mshs["undef"] = geoms[0]
        else:
            if("roi" in in_tiss_mshs.keys()):
                in_tiss_mshs["roi"] = in_msh
            if("skin" in in_tiss_mshs.keys()):
                in_tiss_mshs["skin"] = in_msh
            if("vessels" in in_tiss_mshs.keys()):
                in_tiss_mshs["vessels"] = in_msh
            if("bones" in in_tiss_mshs.keys()):
                in_tiss_mshs["bones"] = in_msh
            if("undef" in in_tiss_mshs.keys()):
                in_tiss_mshs["undef"] = in_msh

        in_tiss_mshs_tot.update(in_tiss_mshs)

    return in_tiss_mshs_tot

def read_meshio_with_retry(msh_pth, max_tries):
    successful_read = False
    tries_dcnt = max_tries
    while not successful_read:
        try:
            data = meshio.read(msh_pth)
            successful_read = True
        except:
            logging.warning(f'Error reading gmsh file {msh_pth}. Retry after 1 second...')
            tries_dcnt -= 1
            if tries_dcnt == 0:
                logging.error(f'Too many tries. Exit.')
                sys.exit(1)
            time.sleep(1)
    return data

def parse_gmsh_log(log):
    list_of_faulty_verts = []
    stages = [("initial",   ""),
              ("meshing",   "Meshing"),
              ("writting",  "Writing"),
              ("Undefined", "undefined_keyword")
              ]
    sid = 0
    do_look_for_vertices = False
    found_error = False
    for l in log:
        curr_stage_name = stages[sid][0]
        next_stage_key = stages[sid+1][1]
        l = l.replace('\n','')
        if(l.find(next_stage_key) != -1):
            sid += 1
            if stages[sid][0] != "meshing":
                do_look_for_vertices = False
        elif(curr_stage_name == "meshing") and ((l.find(" self-intersecting facets") != -1) or (l.find("intersect at point") != -1)):
            logging.warning(f"During meshing self-intersecting error has been detected: \"{l}\"")
            do_look_for_vertices = True
            found_error = True
        elif(l.find("Error") != -1):
            logging.warning(f"Found error reported in GMSH log at stage {curr_stage_name}: \"{l}\"")
            found_error = True
        if do_look_for_vertices:
            #Info:   Segment: [8462,9285] #-1 (0)
            #Info:   Facet:   [8460,9286,9416] #1                
            if(l.find("Segment:") != -1 or l.find("Facet:") != -1 or l.find("Facet 1:") != -1 or l.find("Facet 2:") != -1 or l.find("1st:") != -1 or l.find("2nd:") != -1 ):
                vl = l.split("[")[1]
                vl = vl.split("]")[0]
                vss = vl.split(",")
                vis = [int(vs) for vs in vss]
                list_of_faulty_verts.extend(vis)

    list_of_faulty_verts_b0 = [vid-1 for vid in list_of_faulty_verts]
    list_of_faulty_verts_b0_unique = list(set(list_of_faulty_verts_b0))

    return found_error, list_of_faulty_verts_b0_unique

def makeVolumeMeshUninven(in_stl_path_base, in_stl_pths_tools, work_dir: str = "", fn_tmpl: str = "_tmp", delete_files: bool = True, gmsh_on_path: bool = True, alg3D: str = 'del', num_threads = None, 
                          size_factor = 1.0, border_element_size = 1.0, fill_element_size = 10.0, use_size_fields=True):
    
    tmp_pths = []
    tmp_gmsh_format = "mesh"
    logging.info(f'>Preparing gmsh script...')
    """
    Import mesh data from file

    Args:
        in_stl_pths: File name<s> with extension
        delete_files: Enable/disable deleting temporary files when finished
        gmsh_on_path: Enable/disable using system gmsh rather than bundled gmsh
        alg3D: choose mesh 3D algorithm for tetras 
            mmg3d - daje nierownomierne czterosciany, wieksze wewnatrz obiektu - potencjalnie szybsza symulacja
            del - domyślny algorytm Delaunay
            initial - tylko oryginalne punkty z powierzchni, bez dodatkowych puntow w srodku obiektu

    Returns:
        Mesh data (points, tris, and tets)
    """

    if not type(in_stl_pths_tools) is list:
        in_stl_pths_tools = [in_stl_pths_tools]
        
        
    # choose mesh 3D algorithm for tetras
    # auto, meshadapt, del2d, front2d, delquad, pack, initial2d, del3d, front3d, mmg3d, hxt, initial3d 
    # 1: Delaunay, 3: Initial mesh only, 4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT
    if alg3D == 'mmg3d':
        alg3D_int = 7  # daje nierownomierne czterosciany, wieksze wewnatrz obiektu - potencjalnie szybsza symulacja
    elif alg3D == 'del':
        alg3D_int = 1  # domyślny algorytm Delaunay
    elif alg3D == 'hxt':
        alg3D_int = 10 #  algorytm Delaunay w wersji rownoleglej
    elif alg3D == 'initial':
        alg3D_int = 3  # tylko oryginalne punkty z powierzchni, bez dodatkowych puntow w srodku obiektu

    geo_string = f"// Gmsh project creating volume from surface\n"
    geo_string += f"//1) Disable size from points and curvatures - size will be set by fields. Explanation form gmsh's t10.get example:\n"
    geo_string += f"//   To determine the size of mesh elements, Gmsh locally computes the minimum of\n"
    geo_string += f"//   1) the size of the model bounding box;\n"
    geo_string += f"//   2) if 'Mesh.MeshSizeFromPoints' is set, the mesh size specified at\n"
    geo_string += f"//      geometrical points;\n"
    geo_string += f"//   3) if 'Mesh.MeshSizeFromCurvature' is positive, the mesh size based on\n"
    geo_string += f"//      curvature (the value specifying the number of elements per 2 * pi rad);\n"
    geo_string += f"//   4) the background mesh size field;\n"
    geo_string += f"//   5) any per-entity mesh size constraint.\n"
    geo_string += f"//   This value is then constrained in the interval ['Mesh.MeshSizeMin',\n"
    geo_string += f"//   'Mesh.MeshSizeMax'] and multiplied by 'Mesh.MeshSizeFactor'.  In addition,\n"
    geo_string += f"//   boundary mesh sizes (on curves or surfaces) are interpolated inside the\n"
    geo_string += f"//   enclosed entity (surface or volume, respectively) if the option\n"
    geo_string += f"//   'Mesh.MeshSizeExtendFromBoundary' is set (which is the case by default).\n"
    geo_string += f"//   When the element size is fully specified by a background mesh size field (as\n"
    geo_string += f"//   it is in this example), it is thus often desirable to set:\n"
    geo_string += f"Mesh.MeshSizeExtendFromBoundary = 0;\n"
    if use_size_fields and len(in_stl_pths_tools)>0:
        geo_string += f"Mesh.MeshSizeFromPoints = 0;\n"
    else:
        geo_string += f"Mesh.MeshSizeFromPoints = 1;\n"
    geo_string += f"Mesh.MeshSizeFromCurvature = 0;\n"
    geo_string += f"//   This will prevent over-refinement due to small mesh sizes on the boundary.\n"
    geo_string += f"\n"

    geo_string += f"//2) ROI surface used for volume definition:\n"
    id = 1
    main_surf_data = read_meshio_with_retry(in_stl_path_base, max_tries = 1)
    in_stl_pth_base_rel = os.path.relpath(in_stl_path_base, work_dir)
    do_use_tm_proximity = True
    if do_use_tm_proximity:
        main_surf_tm = trimesh.load(in_stl_path_base)
    geo_string += f"Merge \"{in_stl_pth_base_rel}\";\n"
    geo_string += f"//RefineMesh;\n"
    geo_string += f"Surface Loop({id}) = {{{id}}};\n"
    geo_string += f"Volume({id}) = {{{id}}};\n"
    
    geo_string += f"\n"
    geo_string += f"//3) Tissues surfaces used for adjusting size of volume elements (tetras) at its boundaries:\n"
    geo_string += f"l{id}() = Point{{:}};\n"
    geo_string += f"max_Point_{id} = #l{id}();\n"
    id += 1
    tools_start_id = id
    if do_use_tm_proximity:
        total_points = np.zeros((0,3))
    else:
        main_surf_data_points = main_surf_data.points
        total_points = main_surf_data_points

    for in_stl_pth_tool in in_stl_pths_tools:
        logging.info(f'> Add Steiner points from {in_stl_pth_tool}...')
        in_stl_pth_tool_rel = os.path.relpath(in_stl_pth_tool, work_dir)
        
        surf_data = read_meshio_with_retry(in_stl_pth_tool, max_tries = 10)
        new_points_candidates = surf_data.points
        if do_use_tm_proximity:
            # Points OUTSIDE the mesh will have NEGATIVE distance
            #candidates2total_min_distance = np.abs(trimesh.proximity.signed_distance(main_surf_tm, new_points_candidates))
            # do not use ABS() -> negative values means that the point is outside of ROI. It is a result of a slighetly different 
            #  remeshing of a ROI and SKIN and a remeshed-skin point will stand outside of a remeshed-roi. If this point will be included it will result in a very flat tetra
            #  and a flat tetra can cause some problems during simulation. Not using ABS() will exclude the points taht are outside ROI.
            candidates2total_min_distance = trimesh.proximity.signed_distance(main_surf_tm, new_points_candidates)
        else:
            dist = scipy.spatial.distance.cdist(new_points_candidates, total_points, metric='euclidean')
            candidates2total_min_distance = np.min(dist, axis = 1)
        pidxs = np.where(candidates2total_min_distance > border_element_size*2/3)
        new_points = new_points_candidates[pidxs]
        logging.info(f'>  From {len(new_points_candidates)} points choose {len(new_points)} that are further from base mesh surface...')
        if do_use_tm_proximity and len(total_points)>0:
            new_points_candidates = new_points
            dist = scipy.spatial.distance.cdist(new_points_candidates, total_points, metric='euclidean')
            candidates2total_min_distance = np.min(dist, axis = 1)
            pidxs = np.where(candidates2total_min_distance > border_element_size*1/3)
            new_points = new_points_candidates[pidxs]
            logging.info(f'>  From {len(new_points_candidates)} points choose {len(new_points)} that are further from already present Stainer points...')
        if(len(new_points)>0):
            total_points = np.append(total_points, new_points, axis=0)
            #main_surf_data.
        
            cells_dict          = [('vertex', np.array([[i,] for i in range(len(new_points))]))]
            #tot_cell_groups_dict    = { 'jakub:tissue_id': [tot_cell_groups],
            #                            'gmsh:ref'       : [tot_cell_groups],}

            point_mesh = meshio.Mesh(points = new_points, cells = cells_dict,  cell_data={})
            in_stl_fn_tool = os.path.basename(in_stl_pth_tool)
            in_stl_fn_tool_points = in_stl_fn_tool.replace('.stl', '_Steiner_points.vtk')
            in_stl_pth_tool_points = os.path.join(work_dir, in_stl_fn_tool_points)
            point_mesh.write(in_stl_pth_tool_points, binary=False)
            tmp_pths.append(in_stl_pth_tool_points)
            in_stl_pth_tool_points_rel = os.path.relpath(in_stl_pth_tool_points, work_dir)
        
            geo_string += f"Merge \"{in_stl_pth_tool_points_rel}\";\n"
            #geo_string += f"Surface Loop({id}) = {{{id}}};\n"
            geo_string += f"l{id}() = Point{{:}};\n"
            geo_string += f"max_Point_{id} = #l{id}();\n"
            #geo_string += f"Printf(\"Max point id: %g \", max_Point_{id} );\n"
            #geo_string += f"Printf(\"Point(1):  %g, %g, %g \", Point{{1}});\n"
            #geo_string += f"Printf(\"Point(%g): %g, %g, %g \", max_Point_{id} , Point{{max_Point_{id} }});\n"
            if not use_size_fields:
                geo_string += f"For i In {{max_Point_{id-1}+1:max_Point_{id}}}\n"
                geo_string += f"//    Printf(\"Point %g \", i);\n"
                if(in_stl_pth_tool.find("bones")!= -1):
                    geo_string += f"    MeshSize{{ Point{{i}} }} = {fill_element_size};\n"
                else:
                    geo_string += f"    MeshSize{{ Point{{i}} }} = {border_element_size};\n"
                geo_string += f"EndFor\n"
            geo_string += f"//Add points to the Volume\n"
            geo_string += f"Printf(\"Add Points %g-%g to Volume \", max_Point_{id-1}+1, max_Point_{id} );\n"
            geo_string += f"Point{{max_Point_{id-1}+1:max_Point_{id}}} In Volume{{1}};\n"
            geo_string += f"\n"
            id += 1
        else:
            geo_string += f"//In \"{in_stl_pth_tool}\" I found no points that are further than {border_element_size} from already placed points so I skip this \n"

    geo_string += f"\n"

    if use_size_fields and len(in_stl_pths_tools)>0:
        tools_id = tools_start_id
        for in_stl_pth_tool in in_stl_pths_tools:
            in_stl_pth_tool_rel = os.path.relpath(in_stl_pth_tool, work_dir)
            geo_string += f"Printf(\"Read surface from {in_stl_pth_tool}\" );\n"
            geo_string += f"Merge \"{in_stl_pth_tool_rel}\";\n"
            geo_string += f"Surface Loop({tools_id}) = {{{tools_id}}};\n"
            tools_id += 1
        geo_string += f"\n"
        field_id = 1
        tool_id  = tools_start_id
        for in_stl_pth_tool in in_stl_pths_tools:
            geo_string += f"Printf(\"Create Field[{field_id}] describing distance from surface {in_stl_pth_tool}\" );\n"
            geo_string += f"Field[{field_id}] = Distance;\n"
            geo_string += f"Field[{field_id}].SurfacesList = {{{tool_id}}};\n"
            geo_string += f"Field[{field_id}].NumPointsPerCurve = 2000;\n" # chyba nie pomaga
            field_id += 1
            tool_id  += 1 

        geo_string += f"Printf(\"Create Field[{field_id}] describing minimum distance from any tissue surface\" );\n"
        geo_string += f"Field[{field_id}] = Min;\n"
        geo_string += f"Field[{field_id}].FieldsList = {{ {','.join([str(id) for id in range(1, field_id)])} }};\n"
        field_id += 1
    
        geo_string += f"Printf(\"Create Field[{field_id}] that thresholds Field[{field_id-1}] to only two gieven values of element size {fill_element_size} and {border_element_size}\" );\n"
        geo_string += f"lc = {border_element_size};\n"
        geo_string += f"Field[{field_id}] = Threshold;\n"
        geo_string += f"Field[{field_id}].InField = {field_id-1};\n"
        geo_string += f"Field[{field_id}].Sigmoid = 0;\n"
        geo_string += f"// It seems that the element size is a length of an edge of a triangle and not its height\n"
        geo_string += f"// therefore a distance should be ...\n"
        geo_string += f"Field[{field_id}].DistMin = lc;\n"
        geo_string += f"Field[{field_id}].DistMax = lc * 1.2;\n"
        geo_string += f"Field[{field_id}].SizeMax = {fill_element_size};\n"
        geo_string += f"Field[{field_id}].SizeMin = lc;\n"
        geo_string += f"\n"
        geo_string += f"Printf(\"Choose Field[{field_id}] as the one used to compute the size of volume elements (tetras)\" );\n"
        geo_string += f"Background Field = {field_id};\n"
        
    geo_string += f"\n"
    geo_string += f"//6) Choose 3D meshing algorithm;\n"
    geo_string += f"Mesh.Algorithm3D = {alg3D_int};\n" 
    geo_string += f"Mesh.Smoothing = 3;\n" # 3 smoothing steps
    geo_string += f"Mesh.OptimizeThreshold = 0.3;\n" #Optimize tetrahedral elements that have a quality less than a threshold. Default value: 0.3
    geo_string += f"//Mesh.OptimizeNetgen = 1;\n"
    
    geo_string += f"\n"
    geo_string += f"//7) Other options.\n"
    geo_string += f"// Display. Show full grid:\n"
    geo_string += f"General.Axes = 3;\n" #Axes (0: none, 1: simple axes, 2: box, 3: full grid, 4: open grid, 5: ruler)
    geo_string += f"General.TextEditor = 'C:\\Program Files\\Notepad++\\notepad++.exe %s';\n"
    ## 1: msh, 2: unv, 10: auto, 16: vtk, 19: vrml, 21: mail, 26: pos stat, 27: stl, 28: p3d, 30: mesh, 31: bdf, 32: cgns, 33: med, 34: diff, 38: ir3, 39: inp, 40: ply2, 41: celum, 42: su2, 47: tochnog, 49: neu, 50: matlab
    #geo_string += f"Mesh.Format = ;\n" 
        
    geo_pth = os.path.join(work_dir, f"{fn_tmpl}.geo")
    with open(geo_pth, 'w') as f:
        f.writelines(geo_string)
    tmp_pths.append(geo_pth)

    if gmsh_on_path:
        command_string = 'gmsh '
    else:
        # Check OS type
        if os.name.startswith('nt'):
            # Windows
            command_string = f'"{os.path.dirname(os.path.realpath(__file__))}\\utils\\gmsh.exe"'
        else:
            # Linux
            command_string = f'"{os.path.dirname(os.path.realpath(__file__))}/utils/gmsh"'

            
    command_string = command_string + f' {geo_pth}'
    # Perform 3D mesh generation, then exit
    command_string = command_string + f' -3'
    # format of the output file
    command_string = command_string + f' -format {tmp_gmsh_format}'
    # multithreading
    if not num_threads is None:
        command_string = command_string + f' -nt {num_threads}'
    err_pth = os.path.join(work_dir, f"_{fn_tmpl}_gmsh_out.log")
    command_string = command_string + f' -log {err_pth}'

    logging.info(f'>Launching gmsh using: {command_string}')
    #args = [l for l in command_string.split(' ') if len(l)>0]
    proc = subprocess.Popen(command_string, shell=True, universal_newlines = True)
    proc.wait()
    with open(err_pth, 'r') as f:
        gmsh_log = f.readlines()
        was_error, error_vertices = parse_gmsh_log(gmsh_log)

    if was_error:
        logging.warning(f'>End up in error :/')
        if(len(error_vertices)>0):
            logging.warning(f'> I have list of faulty vertices: {error_vertices}...')

    msh_pth = os.path.join(work_dir, f"{fn_tmpl}.{tmp_gmsh_format}")
    data = read_meshio_with_retry(msh_pth, max_tries = 10)
    tmp_pths.append(msh_pth)

    if (type(delete_files) is str) and (delete_files=="only_mesh"):
        os.remove(msh_pth)
    elif (type(delete_files) is bool) and delete_files:
        for tmp_pth in tmp_pths:
            os.remove(tmp_pth)

    if was_error:
        return True, error_vertices
    else:
        return False, data


    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def angle_between_unit_vectors(v1_u, v2_u):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def main():
    #----------------------------------------------------------------------------
    # initialize logging 
    script_name = os.path.basename(__file__).split(".")[0]
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    initial_log_fn = f"_initial_{script_name}_{time_str}_pid{os.getpid()}.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    logging.info(f'*' * 50)
    logging.info(f"script {script_name} start @ {time.ctime()}")
    logging.info(f"initial log file is {initial_log_fn}")
    logging.info(f"*" * 50)
    logging.info(f"Parse command line arguments...")
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    start_time = time.time()
    
    #----------------------------------------------------------------------------
    logging.info("Reading configuration...")
    parser = ArgumentParser()
    logging.info(' -' * 25)
    logging.info(" Command line arguments:\n  {}".format(' '.join(sys.argv)))
    _logging_levels = logging._levelToName.keys()

    cfa = parser.add_argument_group('config_file_arguments')
    cfa.add_argument("--cfg" , default=[], action='append', type=str, nargs='*', required=False, metavar="PATH", help="one or more config json filenames. Further config files have higher priority. Command line arguments have the highest priority.", )
    parser.add_argument("--logging_level"                   , default=logging.INFO                      , type=int          , required=False, choices=_logging_levels,     help="")
    
    if not(("-h" in sys.argv) or ("--help" in sys.argv)):
        cfg_fns_args, cmd_line_args_rem = parser.parse_known_args(); # bez error gdy natrafi na nieznany parametr (odwrotnie niż "parse_args()")
        logging.getLogger().setLevel(cfg_fns_args.logging_level)
        cfgs = list(map(str, flatten(cfg_fns_args.cfg)))
        # read dictonaries from config files (create a list of dicts)
        cfg_dicts = convert_cfg_files_to_dicts(cfgs)

        # convert cmd_line_args_rem to dictionary so we can use it to update content of the dictonaries from config files
        cmd_line_args_rem_dict = convert_cmd_line_args_to_dict(cmd_line_args_rem)
        
        logging.info(' -' * 25)
        logging.info(" Merge config files arguments with command line arguments...")
        # merge all config dicts - later ones will overwrite entries with the same keys from the former ones
        cfg_dict_pr = {}
        for cfg_dict in cfg_dicts:
            cfg_dict_pr.update(cfg_dict)
        # finally update with the command line arguments dictionary
        cfg_dict_pr.update(cmd_line_args_rem_dict)
        
        logging.info(" Merged arguments:")
        cfg_d = cfg_dict_pr
        print_cfg_dict(cfg_d, indent = 1, skip_comments = True)

        # parse the merged dictionary
        args_list_to_parse = convert_dict_to_cmd_line_args(cfg_dict_pr)
    #----------------------------------------------------------------------------
    parser = ArgumentParser()

    parser.add_argument("-iDir",  "--in_dir",  default = "as_data/st23_preprocessed_meshes/B000004/000003",   help="input directory with *_skin_bones_vessels_mesh_volume.obj and *_skin_mesh_roi.stl files",   metavar="PATH",required=True)
    parser.add_argument("-iPtrn", "--in_pattern",default = '*_skin_bones_vessels_mesh_volume.obj'         ,   help="output directory for the result remeshed mesh",      metavar="PATH",required=True)
    parser.add_argument("-oDir",  "--out_dir", default = "as_data/st23_remeshed/B000004/000003"           ,   help="output directory for the result remeshed mesh",      metavar="PATH",required=True)
    #parser.add_argument("-ps",    "--pitch_skin", type = float,  help="voxel overall dimension for skin",  default = 1.0, required=False)
    parser.add_argument("-p",     "--pitch"     , type = float,  help="voxel overall dimension",  default = 3.0, required=False)
    parser.add_argument("-mp",    "--max_passes", type = int,    help="Max reparation passes",    default = 10,  required=False)
    parser.add_argument("-daat",  "--do_adjust_z_at_top", default=True,  action=arg2boolAct, help="Adjust Z coordinate at top cut in order to preserve flat cut surface.", required=False)
    parser.add_argument("-ofn",   "--output_file_name",  default = None,   help="Forced outut file name",   metavar="PATH",required=False)
    
    parser.add_argument("-v",     "--verbose",                          help="verbose level",                             required=False)
    
    logging.info('-' * 50)
    if not(("-h" in sys.argv) or ("--help" in sys.argv)): 
        # get training arguments
        args, rem_args = parser.parse_known_args(args_list_to_parse)
        
        logging.info("Parsed configuration arguments:")
        args_d = vars(args)
        print_cfg_dict(args_d, indent = 1, skip_comments = True, max_print_len=8)

        if len(rem_args) > 0:
            logging.warning(f"Unrecognize arguments: {rem_args}")
        
    else: 
        # help
        logging.info("Params:")
        logging.info(parser.format_help())
        sys.exit(1)
        
    verbose 	        = 'off'                 if args.verbose is None else args.verbose
    pitch               = args.pitch
    in_pattern          = args.in_pattern
    do_adjust_z_at_top  = args.do_adjust_z_at_top
    
    iDir 		= os.path.normpath(args.in_dir )
    oDir 		= os.path.normpath(args.out_dir)
    
    ses         = os.path.basename(oDir)
    user        = os.path.basename(os.path.dirname(oDir))
    

    if not os.path.isdir(iDir):
        logging.error(f'Input directory {iDir} with meshes file not found !')
        exit(1)
    logging.info("-="*25)
    logging.info(f"START     : {script_name}")
    logging.info(f"in        : {iDir}")
    logging.info(f"out       : {oDir}")
    logging.info(f"pitch     : {pitch}")
    #----------------------------------------------------------------------------
    # redirect logging file to work directory 
    from as_bin.utils.logging_utils import redirect_log
    lDir = redirect_log(oDir, f"_{script_name}_{time_str}_pid{os.getpid()}.log", f"_{script_name}_last.log")
    logging.info('-' * 50)
    #----------------------------------------------------------------------------
    in_ptrn  = os.path.join(iDir, in_pattern)
    in_pths  = glob.glob(in_ptrn)
    if len(in_pths) == 0:
        if len(in_pths) == 0:
            logging.error(f"Expected to find a file matching '{in_ptrn}'")
        sys.exit(1)

    min_len     = len(in_pths[0])
    in_msh_pths = [in_pths[0]]
    if len(in_pths) > 1:
        logging.warning(f"Multiple files match the pattern '{in_ptrn}': {in_pths}")
        for s in in_pths:
            if len(s) < min_len:
                min_len = len(s)
                in_msh_pths  = [s]
        logging.warning(f" {in_msh_pths[0]} is the shortest file name, therefore I choose this one for further processing.")

    logging.info(f"Read files using trimesh:")
    for in_msh_pth in in_msh_pths:
        logging.info(f"   {in_msh_pth}")

    
    in_fn = os.path.basename(in_msh_pths[0])
    in_fn_s = in_fn.split("_",2)
    out_fn_ptrn = f"{in_fn_s[0]}_{in_fn_s[1]}" if len(in_fn_s)==3 else in_fn.split(".",1)[0]
    
    in_tiss_tms = read_tissues(in_msh_pths)
    ts = list(in_tiss_tms.keys())
    #ts = ['roi']

    logging.info(f"Found the following tissues: {','.join(ts)}")
    
    #-----------------------------------------------------------------------------------------
    logging.info(f"Create local STL files for each input tissue")
    tiss_dict = {}
    for t in ts:

        tiss_dict[t] = {}
        tiss_dict[t]["in_msh"     ] = in_tiss_tms[t]

        out_fn = f"{out_fn_ptrn}_{t}_in.stl"
        out_pth = os.path.join(oDir, out_fn)
        
        msh_in = in_tiss_tms[t]
        logging.info(f"  Exporting to {out_pth} STL file...")
        with open(out_pth, 'bw') as f:
            msh_in.export( f, "stl")
                    
        tiss_dict[t]["in_msh_pth" ] = out_pth


    #-----------------------------------------------------------------------------------------
    # STLs remesh
    logging.info(f"Remesh surfaces in a way that equal-triangle-size surface is obtained...")
    #ts = ['roi']
    for t in ts:
        logging.info(f" {t}...")
        if not args.output_file_name is None:
            out_fn = args.output_file_name
        else:
            out_fn = f"{out_fn_ptrn}_{t}.stl"
        out_pth = os.path.join(oDir, out_fn)
            
        dbg_plots = False
        in_t_msh = pv.read(tiss_dict[t]["in_msh_pth"])
        #in_t_msh.clean()
        clus = pyacvd.Clustering(in_t_msh)

        t_face_n  = in_t_msh.n_faces_strict
        t_area    = in_t_msh.area
        t_face_n_expected = int(np.ceil(t_area / (pitch*3/2)))
        t_face_ratio = t_face_n_expected / t_face_n
        t_max_edge_len  = float(tiss_dict[t]["in_msh"].edges_unique_length.max())
        t_edges_num = len(tiss_dict[t]["in_msh"].edges_unique_length)
        t_long_edge_len = float(tiss_dict[t]["in_msh"].edges_unique_length.max())#t_edges_num*2//3])
        if t_long_edge_len > pitch:
            logging.info(f"  long edge has length = {t_long_edge_len:.1f}, where pitch = {pitch}, therefore subdivision is needed before clustering...")

            ratio = t_long_edge_len / pitch
            ratio = int(np.ceil(np.log2(ratio)))
            logging.info(f"   subdivide {ratio} times...")
            if ratio > 4:
                ratio = 4
                logging.warning(f"    limit the number of subdivision to {ratio} due to excessive memory requirements")

            if dbg_plots:
                # plot original mesh
                in_t_msh.plot(show_edges=True, color='w')
            # mesh is not dense enough for uniform remeshing
            clus.subdivide(ratio)

        logging.info(f"   clustering...")
        clus.cluster(t_face_n_expected)
        if dbg_plots:
            # plot clustered roi
            clus.plot(show_edges=False)
        # remesh
        logging.info(f"   remeshing...")
        remesh = clus.create_mesh()
            
        if dbg_plots:
            # plot uniformly remeshed roi
            remesh.plot(color='w', show_edges=True)

        logging.info(f"  Exporting remeshed {t} to {out_pth} STL file...")
        pv.save_meshio(out_pth, remesh)
                
        tiss_dict[t]["remeshed_surf_pth"] = out_pth
        data = read_meshio_with_retry(out_pth, max_tries = 10)
        tiss_dict[t]["remeshed_surf"    ] = data
        
    #----------------------------------------------------------------------------
    
    # clear ROI remeshed mesh - it happens that a duplicated faces occur after remeshing and also some degenerated faces have been produced
    logging.info(f"Check duplicated points and degenerated faces...")
    check_close_points      = True
    check_duplicated_points = False # i tak nic z tym nie robie 
    check_degenerated_faces = True
    check_overlapping_faces = True
    check_boundary_dangling_faces   = True
    check_boundaryonly_faces        = False
    for t in ts:

        logging.info(f" {t}...")
        check_alternating_faces = t=='skin' or t=='roi' or t=='bones' or t=='undef'
        check_narrow_faces      = t=='skin' or t=='roi' or t=='bones' or t=='undef'
        short_edge_th=pitch/4
        keep_repairing = True
        max_repair_loops = 20
        repair_loop_id = -1
        while keep_repairing:
            repair_loop_id += 1
            keep_repairing = False
            mesh = tiss_dict[t]["remeshed_surf"    ]
            faces = mesh.cells_dict['triangle']
            points = mesh.points
            keep_verts = None
            new_faces, new_points, pass_id, changed = mesh_repair(faces, points,                 
                keep_verts                      = keep_verts, 
                check_close_points              = check_close_points, 
                check_duplicated_points         = check_duplicated_points, 
                check_degenerated_faces         = check_degenerated_faces, 
                check_overlapping_faces         = check_overlapping_faces, 
                check_boundary_dangling_faces   = check_boundary_dangling_faces,
                check_boundaryonly_faces        = check_boundaryonly_faces,
                check_alternating_faces         = check_alternating_faces, 
                check_narrow_faces              = check_narrow_faces, 
                max_passes                      = args.max_passes if (keep_verts is None) else 5, 
                short_edge_th                   = short_edge_th 
                #max_adjecency_angle             = args.max_adjecency_angle, \
                #stop_when_no_degenerated_overlapping_alternating_faces = True, \
                #do_return_merged_points_list    = False, \
                #do_remove_unused_points         = True, \
                #log_level                       = 1 \
                )

            if changed:
                logging.info(f'Overwrite {tiss_dict[t]["remeshed_surf_pth"]} file. Save the mesh without problematic faces...')
                
                #tmesh = trimesh.Trimesh(vertices=points, faces=faces)
                cells_dict          = [('triangle', new_faces   )]
                changed_mesh = meshio.Mesh(points = new_points, cells = cells_dict)
                changed_mesh.write(tiss_dict[t]["remeshed_surf_pth"])
                tiss_dict[t]["remeshed_surf"    ] = changed_mesh

            #----------------------------------------------------------------------------
            # 
            
            #-----------------------------------------------------------------------------------------
            #if do_adjust_z_at_top:
            #    mesh = tiss_dict[t]["remeshed_surf"    ]
            #    faces  = mesh.cells_dict['triangle']
            #    points = mesh.points
            #    if True:
            #        out_fn = f"XXX_dev_input.stl"
            #        out_pth = os.path.join(oDir, out_fn)
            #        tiss_dict[f"rem_z_max_plain"] = out_pth
            #        cells_dict          = [('triangle', new_faces   )]
            #        offseted_mesh = meshio.Mesh(points = points, cells = cells_dict)
            #        offseted_mesh.write(out_pth, binary=True)
            #    new_points, new_faces, changed = adjust_z_at_top(points, faces, pitch/2, max_normal_deviation_deg = 35)
            #    if True:
            #        out_fn = f"XXX_dev_limit_35deg.stl"
            #        out_pth = os.path.join(oDir, out_fn)
            #        tiss_dict[f"rem_z_max_plain"] = out_pth
            #        cells_dict          = [('triangle', new_faces   )]
            #        offseted_mesh = meshio.Mesh(points = new_points, cells = cells_dict)
            #        offseted_mesh.write(out_pth, binary=True)
            #    new_points, new_faces, changed = adjust_z_at_top(points, faces, pitch/2, max_normal_deviation_deg = None)
            #    if True:
            #        out_fn = f"XXX_dev_limit_Nonedeg.stl"
            #        out_pth = os.path.join(oDir, out_fn)
            #        tiss_dict[f"rem_z_max_plain"] = out_pth
            #        cells_dict          = [('triangle', new_faces   )]
            #        offseted_mesh = meshio.Mesh(points = new_points, cells = cells_dict)
            #        offseted_mesh.write(out_pth, binary=True)
            
            if do_adjust_z_at_top and (repair_loop_id < max_repair_loops):
                logging.info(f"Adjust Z coordinate at top cut in order to preserve flat cut surface...")
                
                mesh = tiss_dict[t]["remeshed_surf"    ]
                faces  = mesh.cells_dict['triangle']
                points = mesh.points
                new_points, new_faces, changed = adjust_z_at_top(points, faces, pitch/2, max_normal_deviation_deg = 35)
                #new_points, new_faces, changed = adjust_z_at_top(points, faces, pitch/3, max_normal_deviation_deg = 45)
                if changed:        
                    points = new_points
                    faces  = new_faces
                    keep_repairing = True
                    
                    logging.info(f'  Changed! Save the mesh without Z aligned vertices to {tiss_dict[t]["remeshed_surf_pth"]} file...')
                    #tmesh = trimesh.Trimesh(vertices=points, faces=faces)
                    cells_dict          = [('triangle', faces   )]
                    changed_mesh = meshio.Mesh(points = points, cells = cells_dict)
                    changed_mesh.write(tiss_dict[t]["remeshed_surf_pth"], binary=True)
                    tiss_dict[t]["remeshed_surf"    ] = changed_mesh
                else:
                    logging.info(f"  No points to change. Skip.")

    #----------------------------------------------------------------------------
    logging.info(f"Finished!")
    
if __name__=='__main__':
    main()