"""
Import a mesh file and convert it to tetras.
"""

from distutils.log import info
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
from scipy.spatial import cKDTree

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
from as_bin.utils.mesh_utils import mesh_repair, adjust_z_at_top, mesh_remove_plane_at_coor, smooth_mesh_and_rewrite
#-----------------------------------------------------------------------------------------
def normalize_vectors(vert_normals):
    vert_normals_len = np.linalg.norm(vert_normals, axis = 1)
    zero_len_v = np.where(vert_normals_len==0)
    vert_normals_len[zero_len_v] = 1.0
    vert_normals_n = vert_normals/vert_normals_len[:,np.newaxis]
    return vert_normals_n

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
#----------------------------------------------------------------------------
def update_verts_data(vertex_filter, vert_normals, vert2faceDict, vert_area, face_areas, verts_merging_history):
    if (len(verts_merging_history) > 0):
        for verts_merging_dict in verts_merging_history:
            for vid_red, vid_rem in verts_merging_dict.items():
                v_rem_faces_area = vert_area[vid_rem]
                v_rem_normal = vert_normals[vid_rem]
                v_red_faces_area = vert_area[vid_red]
                v_red_normal = vert_normals[vid_red]
                vert_normals[vid_rem]  = np.average([v_rem_normal, v_red_normal], weights = [v_rem_faces_area, v_red_faces_area],axis=0)
                vert_normals[vid_rem] =  vert_normals[vid_rem]/np.linalg.norm( vert_normals[vid_rem]) 
                vert2faceDict[vid_rem] = np.unique([*vert2faceDict[vid_rem], *vert2faceDict[vid_red]])
                vert_area[vid_rem]     = np.sum(face_areas[vert2faceDict[vid_rem]],axis=0) 
                v_rem_in_filtered = (vid_rem in vertex_filter)
                v_red_in_filtered = (vid_red in vertex_filter)
                if       v_rem_in_filtered and not v_red_in_filtered:
                    _=1
                elif not v_rem_in_filtered and not v_red_in_filtered:
                    _=1
                elif not v_rem_in_filtered and     v_red_in_filtered:
                    vertex_filter[np.where(vertex_filter == vid_red)] = vid_rem
                elif     v_rem_in_filtered and     v_red_in_filtered:
                    vertex_filter = np.delete(vertex_filter, np.where(vertex_filter == vid_red))
                    
#----------------------------------------------------------------------------
def get_triangles_normals(faces, verts, triCells_filter_ids = None):
    if triCells_filter_ids is None:
        tris_vids = faces
    else:
        tris_vids = faces[triCells_filter_ids]
    tris_coors = verts[tris_vids]
    tris_normals = [np.cross(tri_coors[1]-tri_coors[0], tri_coors[2]-tri_coors[0]) for tri_coors in tris_coors]
    tris_normals = [ftn/np.linalg.norm(ftn) for ftn in tris_normals]
    return np.array(tris_normals)

def get_triangles_normal(faces, verts, triCells_filter_ids):
    tris_normals = get_triangles_normals(faces, verts, triCells_filter_ids)
    tris_normal = np.mean(tris_normals, axis=0)
    return np.array(tris_normal)
    
def get_triangles_areas(faces, verts, triCells_filter_ids = None):
    if triCells_filter_ids is None:
        tris_vids = faces
    else:
        tris_vids = faces[triCells_filter_ids]
    tris_coors = verts[tris_vids]
    tris_crosss= [np.cross(tri_coors[1]-tri_coors[0], tri_coors[2]-tri_coors[0]) for tri_coors in tris_coors]
    tris_areas  = [np.linalg.norm(tri_cross)/2 for tri_cross in tris_crosss]
    return np.array(tris_areas)

#-----------------------------------------------------------------------------------------
def offset_verts(in_msh, magnitude, vertex_filter=None, vertex_magnitude = None, normal = None, vertex_normal = None, 
                do_LP_outline_verts_normals = False, LP_filter_outline_normals_N = 9,
                pitch = 1, 
                do_adjust_z_at_top = True,  do_original_z_at_top = True,
                do_keep_outline_verts = False, do_simple = False,
                do_update_vertices_normals_at_each_step = False, smooth_mesh_at_each_step = False,
                dbg_files = False, dbg_o_dir = "./", dbg_o_fn_ptrn = "_dbg_out",
                return_type_is_meshio = False
    ):
               
    #----------------------------------------------------------------------------
    logging.info(f"Assign meshes to tissue's dicts")
    tiss_dict = {}

    verts = in_msh.vertices
    faces = in_msh.faces

    #----------------------------------------------------------------------------
    done_offset = 0.0
    do_use_trimesh_funcs = True
    #----------------------------------------------------------------------------
    if dbg_files:
        out_fn = f"{dbg_o_fn_ptrn}_o{done_offset:.2f}_1_offseted.stl"
        out_pth = os.path.join(dbg_o_dir, out_fn)
        tiss_dict[f"offseted_surf_pth_{done_offset:.2f}"] = out_pth
        cells_dict          = [('triangle', faces   )]
        offseted_mesh = meshio.Mesh(points = verts, cells = cells_dict)
        offseted_mesh.write(out_pth, binary=True)
        tiss_dict["offseted_surf"    ] = offseted_mesh
    #----------------------------------------------------------------------------
    if vertex_filter is None:
        vertex_filter = np.array(range(len(verts)))
    else:
        vertex_filter = np.array(vertex_filter)

    if vertex_magnitude is None:
        vertex_magnitude = np.ones(len(vertex_filter))
    else:
        vertex_magnitude = np.array(vertex_magnitude)
        assert not (vertex_filter is None)
        assert len(vertex_magnitude) == len(vertex_filter)

    #----------------------------------------------------------------------------
    if normal is None and vertex_normal is None:
        logging.info(f"Compute normals for vertices...")

        logging.info(f" Find faces that are using each vertex...")
        if do_use_trimesh_funcs:
            vert2faceDict = {vid: in_msh.vertex_faces[vid][0:vd] for vid, vd in enumerate(in_msh.vertex_degree)}
            logging.info(f" Find face's normals...")
            face_normals = copy.copy(in_msh.face_normals)
            logging.info(f" Find face's areas...")
            tris_crosss= in_msh.triangles_cross
            face_areas  = np.array([np.linalg.norm(tri_cross)/2 for tri_cross in tris_crosss])
            logging.info(f" average normals of vertex's faces weighted by those faces areas in order to find vertex normal...")
            vert_normals = copy.copy(in_msh.vertex_normals)
            if do_original_z_at_top:
                max_z_val = np.max(verts[:,2])
                vert_normals[verts[:,2] == max_z_val, 2] = 0.0
                vert_normals = normalize_vectors(vert_normals)

        else:
            vert2faceDict = {vid:[] for vid in range(len(verts))}
            for fid, f in enumerate(faces):
                vert2faceDict[f[0]].append(fid)
                vert2faceDict[f[1]].append(fid)
                vert2faceDict[f[2]].append(fid)
            logging.info(f" Find face's normals...")
            face_normals = get_triangles_normals(faces, verts)
            logging.info(f" Find face's areas...")
            face_areas   = get_triangles_areas  (faces, verts)
            logging.info(f" average normals of vertex's faces weighted by those faces areas in order to find vertex normal...")
            vert_normals = np.array([np.average(face_normals[vert2faceDict[vid]], weights = face_areas[vert2faceDict[vid]],axis=0) for vid in range(len(verts))])
            vert_normals = normalize_vectors(vert_normals)
            if do_original_z_at_top:
                max_z_val = np.max(verts[:,2])
                vert_normals[verts[:,2] == max_z_val, 2] = 0.0
                vert_normals = normalize_vectors(vert_normals)

        vert_area    = np.array([np.sum(face_areas[vert2faceDict[vid]],axis=0) for vid in range(len(verts))])
        #vert_normals_filtered = np.array([np.average(face_normals[vert2faceDict[vid]], weights = face_areas[vert2faceDict[vid]],axis=0) for vid in vertex_filter])
        #vert_area_filtered    = np.array([np.sum(face_areas[vert2faceDict[vid]],axis=0) for vid in vertex_filter])
    elif not vertex_normal is None :
        logging.info(f"Constant normals for each vertex has been given")
        vert_normals = copy.copy(in_msh.vertex_normals)
        vert_normals[vertex_filter] = np.array(vertex_normal).reshape(-1,3)
        vert_normals = normalize_vectors(vert_normals)
        _=1
    elif not normal is None :
        logging.info(f"Constant normal for vertices has been given")
        vert_normals = np.ones((len(verts),3)) * normal/np.linalg.norm(normal)
        #vert_normals_filtered = np.array([normal for vid in vertex_filter])
        
    #----------------------------------------------------------------------------
    if do_LP_outline_verts_normals:
        
        mo = in_msh.outline()
        has_outline = len(mo.entities) > 0
        outline_vids = []
        if has_outline:
            for olid, ol in enumerate(mo.entities):
                o_len = len(ol.nodes) 
                if o_len > 2:
                    ol = mo.entities[olid]
                    outline_nodes_ids = ol.points[:-1] if (ol.points[0] == ol.points[-1]) else ol.points
                    outline_vids.extend(list(outline_nodes_ids))

        outline_vertex_neighbors  = [[vid, *in_msh.vertex_neighbors[vid]] for vid in outline_vids]
        outline_vertex_normals_LP = copy.copy(vert_normals)
        for i in range(LP_filter_outline_normals_N):
            outline_vertex_normals_LP[outline_vids] = np.array([np.average(outline_vertex_normals_LP[outline_vertex_neighbors[id]], weights =np.linalg.norm(outline_vertex_normals_LP[outline_vertex_neighbors[id]], axis = 1), axis = 0) for id, ovid in enumerate(outline_vids)])
        vert_normals    = outline_vertex_normals_LP

    #----------------------------------------------------------------------------
    vert_normals[vertex_filter] *= vertex_magnitude[:,np.newaxis]
    max_vertex_magnitude = np.max(vertex_magnitude)
    #----------------------------------------------------------------------------
    # clear ROI remashed mesh - it happens that a duplicated facess occure after remeshing and also some degenerated faces have been produced
    if do_keep_outline_verts:
        if do_simple:
            logging.warning(f"do_keep_outline_verts is ignored due to enabled do_simple switch!")
        else:
            mo = in_msh.outline()
            outline_verts_ids_l = []
            for olid, ol in enumerate(mo.entities):
                outline_verts_ids = ol.points
                outline_verts_ids_l.extend(outline_verts_ids)
            keep_verts = np.unique(outline_verts_ids_l)
    else:
        keep_verts = None
    if not do_simple:
        short_edge_th=pitch/2
        logging.info(f"Initial mesh simplification by removing short edges and degenerated faces...")
        faces, verts, pass_id, changed, verts_merging_history = mesh_repair(faces, verts, \
            keep_verts                      = keep_verts, \
            check_narrow_faces              = True, \
            max_passes                      = 100 if (keep_verts is None) else 5, \
            short_edge_th                   = short_edge_th, \
            max_adjecency_angle             = 150, \
            stop_when_no_degenerated_overlapping_alternating_faces = True, \
            do_return_merged_points_list    = True, \
            do_remove_unused_points         = False, \
            log_level                       = 1)

        if normal is None and vertex_normal is None:        
            update_verts_data(vertex_filter, vert_normals, vert2faceDict, vert_area, face_areas, verts_merging_history)

        if dbg_files:
            out_fn = f"{dbg_o_fn_ptrn}_o{done_offset:.2f}_2_repaired.stl"
            out_pth = os.path.join(dbg_o_dir, out_fn)
            tiss_dict[f"repaired_surf_pth_{done_offset:.2f}"] = out_pth
            logging.info(f'Save the mesh without problematic faces to {out_pth} file...')
            cells_dict          = [('triangle', faces   )]
            changed_mesh = meshio.Mesh(points = verts, cells = cells_dict)
            changed_mesh.write(out_pth, binary=True)
    #----------------------------------------------------------------------------
    while(abs(done_offset) < abs(magnitude)):
        if do_simple:
            step_offset = magnitude
            do_repair = False
        else:
            do_compute_safe_step = True
            if False:
                tmp_msh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                vs = tmp_msh.vertices
                graph = tmp_msh.vertex_adjacency_graph 
                min_edge_len_per_vs = np.array([np.min(np.linalg.norm(vs[vid]-vs[list(graph.neighbors(vid))], axis = 1)) for vid in vertex_filter])
                mag2edge = np.linalg.norm(vert_normals[vertex_filter], axis = 1) / min_edge_len_per_vs
                mag2edge_max = max(mag2edge)# * (1-abs(done_offset))/abs(magnitude)

                safe_normal_part_to_shift = (1/mag2edge_max)/2
                step_offset = np.min([safe_normal_part_to_shift, abs(magnitude - done_offset)])
            elif do_compute_safe_step:
                max_recog_dist = pitch*2
                kdtree = cKDTree(verts)
                nid = 2
                vf2v_dists, vf2v_vidx = kdtree.query(verts[vertex_filter], distance_upper_bound=max_recog_dist, k=[nid])
                vf2v_dists = np.array(vf2v_dists.flat)
                vf2v_dists[np.isinf(vf2v_dists)] = max_recog_dist
                zero_dist_vids = np.where(vf2v_dists == 0.0)[0]
                while len(zero_dist_vids) > 0:
                    nid += 1
                    vf2v_distsN, vf2v_vidxN = kdtree.query(verts[vertex_filter][zero_dist_vids], distance_upper_bound=max_recog_dist, k=[nid])
                    zero_dist_vidsN = np.where(vf2v_distsN == 0.0)[0]
                    vf2v_dists[zero_dist_vids] = vf2v_distsN.flat
                    zero_dist_vids = zero_dist_vids[zero_dist_vidsN]

                mag2edge = np.linalg.norm(vert_normals[vertex_filter], axis = 1) / vf2v_dists
                mag2edge_max = max(mag2edge)# * (1-abs(done_offset))/abs(magnitude)

                safe_normal_part_to_shift = (1/mag2edge_max)/2
                step_offset = np.max([safe_normal_part_to_shift, (pitch/max_vertex_magnitude)/4])
                step_offset = np.min([step_offset, abs(magnitude - done_offset)])
            else:
                step_offset = np.min([(pitch/max_vertex_magnitude)/4, abs(magnitude - done_offset)])
            #logging.error(f"-> step_offset {step_offset:.3f}, Offsetting vertices by {done_offset*100:.1f}-{(done_offset+step_offset)*100:.1f}% of required magnitude...")
            if magnitude < 0:
                step_offset *= -1
            do_repair = True
            
        #----------------------------------------------------------------------------
        logging.info(f"Offsetting vertices by {done_offset*100:.1f}-{(done_offset+step_offset)*100:.1f}% of required magnitude...")
        done_offset += step_offset
        verts[vertex_filter] += vert_normals[vertex_filter] * step_offset
        
        is_last = abs(done_offset) >= abs(magnitude)

        #----------------------------------------------------------------------------
        if dbg_files:
            #tmesh = trimesh.Trimesh(vertices=verts, faces=faces)
            out_fn = f"{dbg_o_fn_ptrn}_o{done_offset:.2f}_1_offseted.stl"
            out_pth = os.path.join(dbg_o_dir, out_fn)
            tiss_dict[f"offseted_surf_pth_{done_offset:.2f}"] = out_pth
            cells_dict          = [('triangle', faces   )]
            offseted_mesh = meshio.Mesh(points = verts, cells = cells_dict)
            offseted_mesh.write(out_pth, binary=True)
            tiss_dict["offseted_surf"    ] = offseted_mesh

        #----------------------------------------------------------------------------
        if do_repair:
            # clear ROI remeshed mesh - it happens that a duplicated faces occur after remeshing and also some degenerated faces have been produced
            logging.info(f"Check duplicated verts and degenerated faces...")            

            logging.info(f" repair...")
            keep_repairing = True
            while(keep_repairing):
                keep_repairing = False
                faces, verts, pass_id, changed, verts_merging_history = mesh_repair(faces, verts, \
                    keep_verts                      = keep_verts, \
                    check_duplicated_points         = False, \
                    check_boundary_dangling_faces   = True, \
                    check_boundaryonly_faces        = False, \
                    #check_narrow_faces              = is_last, \
                    check_narrow_faces              = True, \
                    max_passes                      = 100 if (keep_verts is None) else 5, \
                    short_edge_th                   = short_edge_th, \
                    max_adjecency_angle             = 150, \
                    stop_when_no_degenerated_overlapping_alternating_faces = True, \
                    do_return_merged_points_list    = True, \
                    do_remove_unused_points         = False, \
                    log_level                       = 1 )
                
                if not is_last and normal is None and vertex_normal is None:
                    update_verts_data(vertex_filter, vert_normals, vert2faceDict, vert_area, face_areas, verts_merging_history)
                
                if changed:
                    logging.info(f'Reparation done!')
                else:
                    logging.info(f'Reparation not needed. Not changed!')
                    
                if do_adjust_z_at_top:# and not do_original_z_at_top:
                    logging.info(f"Adjust Z coordinate at top cut in order to preserve flat cut surface...")
                    if do_original_z_at_top:
                        force_z_val = max_z_val
                    else:
                        force_z_val = None
                    verts, faces, changed = adjust_z_at_top(verts, faces, pitch/3, force_z_val = force_z_val)
                    if changed:        
                        keep_repairing = True
            if do_update_vertices_normals_at_each_step:
                logging.info(f" Recalculate vertices' normals...")
                tmp_msh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                vert2faceDict = {vid: tmp_msh.vertex_faces[vid][0:vd] for vid, vd in enumerate(tmp_msh.vertex_degree)}
                logging.info(f"  Find face's normals...")
                face_normals = copy.copy(tmp_msh.face_normals)
                logging.info(f"  Find face's areas...")
                tris_crosss= tmp_msh.triangles_cross
                face_areas  = np.array([np.linalg.norm(tri_cross)/2 for tri_cross in tris_crosss])
                logging.info(f"  average normals of vertex's faces weighted by those faces areas in order to find vertex normal...")
                vert_normals = copy.copy(tmp_msh.vertex_normals)
                if do_original_z_at_top:
                    max_z_val = np.max(verts[:,2])
                    vert_normals[verts[:,2] == max_z_val, 2] = 0.0
                    vert_normals = normalize_vectors(vert_normals)

            if dbg_files:
                out_fn = f"{dbg_o_fn_ptrn}_o{done_offset:.2f}_2_repaired.stl"
                out_pth = os.path.join(dbg_o_dir, out_fn)
                tiss_dict[f"repaired_surf_pth_{done_offset:.2f}"] = out_pth
                logging.info(f'Save the mesh without problematic faces to {out_pth} file...')
                cells_dict          = [('triangle', faces   )]
                changed_mesh = meshio.Mesh(points = verts, cells = cells_dict)
                changed_mesh.write(out_pth, binary=True)
    
        if smooth_mesh_at_each_step:
            logging.info(f" Smooth vertices at this step...")
            tmp_msh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            outlines = tmp_msh.outline().entities
            if len(outlines) > 0:
                outlines_vids = np.unique(np.concatenate([o.points for o in outlines]))
                verts_filter_ids = np.setdiff1d(vertex_filter, outlines_vids)
            else:
                verts_filter_ids = vertex_filter
            logging.info(f"  .")
            new_msh = smooth_mesh_and_rewrite(tmp_msh, verts_filter_ids, steps = 1, depth = 0, save_backup_of_input_file = False)
            verts = new_msh.vertices
            logging.info(f"  .")
    #----------------------------------------------------------------------------
    if return_type_is_meshio:
        cells_dict          = [('triangle', faces   )]
        changed_mesh = meshio.Mesh(points = verts, cells = cells_dict)
    else:
        changed_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return changed_mesh
#----------------------------------------------------------------------------
def main():
    #----------------------------------------------------------------------------
    # initialize logging 
    script_name = os.path.basename(__file__).split(".")[0]
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    initial_log_fn = f"_initial_{script_name}_{time_str}_pid{os.getpid()}.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    try:
        os.chmod(initial_log_fn, 0o666)
    except:
        logging.warning(f"Could not change log file permitions {initial_log_fn}. I'm not the owner of the file?")
    
    
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
        cmd_line_args_rem_dict = convert_cmd_line_args_to_dict(cmd_line_args_rem, recognize_negative_values = True)
        
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
        print_cfg_dict(cfg_d, indent = 1, skip_comments = True, max_print_len=8)

        # parse the merged dictionary
        args_list_to_parse = convert_dict_to_cmd_line_args(cfg_dict_pr)
    
    for i, arg in enumerate(args_list_to_parse):
        if (arg[0] == '-') and arg[1].isdigit(): 
            args_list_to_parse[i] = ' ' + arg
    #---------------------------------------------------------------------------- 
    parser = ArgumentParser()

    parser.add_argument("-iDir",  "--in_dir",  default = "as_data/st23_preprocessed_meshes/B000004/000003",   help="input directory with *_skin_bones_vessels_mesh_volume.obj and *_skin_mesh_roi.stl files",   metavar="PATH",required=True)
    parser.add_argument("-iPtrn", "--in_pattern",default = '_skin_bones_vessels_mesh_volume.obj'         ,   help="output directory for the result remeshed mesh",      metavar="PATH",required=True)
    parser.add_argument("-oDir",  "--out_dir", default = "as_data/st23_remeshed/B000004/000003"           ,   help="output directory for the result remeshed mesh",      metavar="PATH",required=True)
    #parser.add_argument("-ps",    "--pitch_skin", type = float,  help="voxel overall dimension for skin",  default = 1.0, required=False)
    parser.add_argument("-p",     "--pitch"     , type = float,  help="voxel overall dimension",  default = 1.0, required=False)
    parser.add_argument("-vf",    "--vertex_filter", type = int,  help="", nargs="*",  default = None, required=False)
    parser.add_argument("-vm",    "--vertex_magnitude", type = float,  help="", nargs="*",  default = None, required=False)
    parser.add_argument("-vn",    "--vertex_normal", type = float, help="",  nargs="*", default = None,  required=False)
    parser.add_argument("-n",     "--normal", type = float, help="",  nargs=3, default = None,  required=False)
    parser.add_argument("-m",     "--magnitude", type = float, help="", default = None,  required=True)
    parser.add_argument("-ds",    "--do_simple", default=False,  action=arg2boolAct, help="Process offset in a single step and do not repair.", required=False)
    parser.add_argument("-duvnaes","--do_update_vertices_normals_at_each_step", default=False,  action=arg2boolAct, help="", required=False)
    parser.add_argument("-smaes", "--smooth_mesh_at_each_step", default=False,  action=arg2boolAct, help="", required=False)
    parser.add_argument("-daat",  "--do_adjust_z_at_top", default=True,  action=arg2boolAct, help="Adjust Z coordinate at top cut in order to preserve flat cut surface.", required=False)
    parser.add_argument("-doat",  "--do_original_z_at_top", default=False,  action=arg2boolAct, help="Adjust Z coordinate at top cut in order to preserve flat cut surface.", required=False)
    parser.add_argument("-dkov",  "--do_keep_outline_verts", default=True,  action=arg2boolAct, help="", required=False)
    parser.add_argument("-dlpon",  "--do_LP_outline_verts_normals", default=False,  action=arg2boolAct, help="Low Pass filtering of outline normals before offsetting", required=False)
    parser.add_argument("-dlponn", "--LP_filter_outline_normals_N", default=9, type = int,  help="", required=False)
    parser.add_argument(          "--dbg_files",    default=False, action=arg2boolAct, help="Leave all files for debug", required=False)
    
    
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

    in_pattern          = args.in_pattern
    
    iDir 		= os.path.normpath(args.in_dir )
    oDir 		= os.path.normpath(args.out_dir)
    
    if not os.path.isdir(iDir):
        logging.error(f'Input directory {iDir} with meshes file not found !')
        exit(1)
    #----------------------------------------------------------------------------
    # redirect logging file to work directory 
    from as_bin.utils.logging_utils import redirect_log
    lDir = redirect_log(oDir, f"_{script_name}_{time_str}_pid{os.getpid()}.log", f"_{script_name}_last.log")
    logging.info('-' * 50)
    #----------------------------------------------------------------------------
    if in_pattern.find("*") == -1:
        in_ptrn  = os.path.join(iDir, f"*{in_pattern}")
    else:
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

    logging.info(f"Read file using trimesh:")
    in_msh_pth = in_msh_pths[0]
    logging.info(f"   {in_msh_pth}")
    
    in_fn = os.path.basename(in_msh_pth)
    in_fn_s = in_fn.split("_",2)
    out_fn_ptrn = in_fn.split(".",1)[0]
    
    in_msh = trimesh.load_mesh(in_msh_pth, process = False)
    if in_msh.body_count > 100:
        logging.warning(f"Read mesh from {in_msh_pth} but got {in_msh.body_count} bodies in that mesh! It can be a result of errorous read. Reading was performed with process set to False.") 
        if not args.vertex_filter is None:
            logging.error(f"Can not re read with 'process' set to True because vertex_filter is set and seting 'process' can re numerate vertices. Exit")
            sys.exit(-1)
        else:
            logging.warning(f"Reading once more with 'process' set to True.") 
            in_msh = trimesh.load_mesh(in_msh_pth, process = True)
            if in_msh.body_count > 100:
                logging.error(f"Still the number of bodies is high: {in_msh.body_count}. Exit")
                sys.exit(-1)
            else:
                logging.info(f" OK")
    
    changed_mesh = offset_verts(in_msh, 
                args.magnitude,
                args.vertex_filter, 
                args.vertex_magnitude,
                args.normal,
                args.vertex_normal,
                args.do_LP_outline_verts_normals,
                args.LP_filter_outline_normals_N,
                
                args.pitch,
                args.do_adjust_z_at_top,
                args.do_original_z_at_top,
                args.do_keep_outline_verts,
                args.do_simple,
                args.do_update_vertices_normals_at_each_step,
                args.smooth_mesh_at_each_step,

                args.dbg_files,
                dbg_o_dir = oDir,
                dbg_o_fn_ptrn = out_fn_ptrn,
                return_type_is_meshio = True
    )
    #----------------------------------------------------------------------------
    #tmesh = trimesh.Trimesh(vertices=verts, faces=faces)
    out_fn = f"{out_fn_ptrn}.stl"
    out_pth = os.path.join(oDir, out_fn)
    logging.info(f'Writing to {out_pth}!')
    changed_mesh.write(out_pth, binary=True)

    #----------------------------------------------------------------------------
    logging.info(f"Finished!")
    
if __name__=='__main__':
    main()