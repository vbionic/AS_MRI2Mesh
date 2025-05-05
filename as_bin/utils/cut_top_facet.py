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
from as_bin.utils.mesh_utils import mesh_repair, adjust_z_at_top, mesh_remove_plane_at_coor
#-----------------------------------------------------------------------------------------
    

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

    cfa = parser.add_argument_group('config_file_arguments')
    cfa.add_argument("--cfg" , default=[], action='append', type=str, nargs='*', required=False, metavar="PATH", help="one or more config json filenames. Further config files have higher priority. Command line arguments have the highest priority.", )
    
    if not(("-h" in sys.argv) or ("--help" in sys.argv)):
        cfg_fns_args, cmd_line_args_rem = parser.parse_known_args(); # bez error gdy natrafi na nieznany parametr (odwrotnie niż "parse_args()")
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
        print_cfg_dict(cfg_d, indent = 1, skip_comments = True)

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
    parser.add_argument(          "--dbg_files",    default=False, action=arg2boolAct, help="Leave all files for debug", required=False)
    
    parser.add_argument("-v",     "--verbose",                          help="verbose level",                             required=False)

    logging.info('-' * 50)
    if not(("-h" in sys.argv) or ("--help" in sys.argv)): 
        # get training arguments
        args, rem_args = parser.parse_known_args(args_list_to_parse)
        
        logging.info("Parsed configuration arguments:")
        args_d = vars(args)
        print_cfg_dict(args_d, indent = 1, skip_comments = True)

        if len(rem_args) > 0:
            logging.warning(f"Unrecognize arguments: {rem_args}")
        
    else: 
        # help
        logging.info("Params:")
        logging.info(parser.format_help())
        sys.exit(1)

    verbose 	        = 'off'                 if args.verbose is None else args.verbose
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
    #out_fn_ptrn = f"{in_fn_s[0]}_{in_fn_s[1]}" if len(in_fn_s)==3 else in_fn.split(".",1)[0]
    out_fn_ptrn = in_fn.split(".",1)[0]
    
    in_msh = trimesh.load(in_msh_pth)
    
    #-----------------------------------------------------------------------------------------
    logging.info(f"Assign meshes to tissue's dicts")
    tiss_dict = {}
    tiss_dict["in_msh"     ] = in_msh
    tiss_dict["in_msh_pth" ] = in_msh_pth

    verts = in_msh.vertices
    faces = in_msh.faces

    #----------------------------------------------------------------------------
    max_z = max(verts[:,2])
    logging.info(f"Cutting at Z = {max(verts[:,2])}...")
    faces, verts, changed = mesh_remove_plane_at_coor(
        faces, verts, \
        coordinates_to_remove = [None, None, max_z] )
    in_msh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out_fn = f"{out_fn_ptrn}.stl"
    out_pth = os.path.join(oDir, out_fn)
    logging.info(f"Saving to {out_pth}...")
    tiss_dict[f"rem_z_max_plain"] = out_pth
    cells_dict          = [('triangle', faces   )]
    offseted_mesh = meshio.Mesh(points = verts, cells = cells_dict)
    offseted_mesh.write(out_pth, binary=True)
    tiss_dict["rem_z_max_plain"    ] = offseted_mesh

    #----------------------------------------------------------------------------
    logging.info(f"Finished!")
    
if __name__=='__main__':
    main()