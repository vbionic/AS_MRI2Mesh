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
    
def read_tissues(in_msh_pths):
    
    in_tiss_mshs_tot = {}
    for in_msh_pth in in_msh_pths:
        in_tiss_mshs = {}
        in_msh = trimesh.load(in_msh_pth)

        if(in_msh_pth.find("omega") != -1):
            in_tiss_mshs["omega"] = None
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
                
            if("omega" in in_tiss_mshs.keys()):
                skin_id = np.argmax(bb_volum)
                in_tiss_mshs["omega"] = geoms[skin_id]

            if("undef" in in_tiss_mshs.keys()):
                in_tiss_mshs["undef"] = geoms[0]
        else:
            if("roi" in in_tiss_mshs.keys()):
                in_tiss_mshs["roi"] = in_msh
            if("omega" in in_tiss_mshs.keys()):
                in_tiss_mshs["omega"] = in_msh
            if("skin" in in_tiss_mshs.keys()):
                in_tiss_mshs["skin"] = in_msh
            if("vessels" in in_tiss_mshs.keys()):
                in_tiss_mshs["vessels"] = in_msh
            if("bones" in in_tiss_mshs.keys()):
                in_tiss_mshs["bones"] = in_msh

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
    parser.add_argument("-p",     "--pitch"     , type = float,  help="voxel overall dimension",  default = 1.0, required=False)
    parser.add_argument("-maa",   "--max_adjecency_angle", type = float,  help="",  default = 150.0, required=False)
    parser.add_argument("-mp",    "--max_passes", type = int,    help="Max reparation passes",    default = 20,  required=False)
    parser.add_argument("-daat",  "--do_adjust_z_at_top", default=True,  action=arg2boolAct, help="Adjust Z coordinate at top cut in order to preserve flat cut surface.", required=False)
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
    pitch               = args.pitch
    in_pattern          = args.in_pattern
    do_adjust_z_at_top  = args.do_adjust_z_at_top
    
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
    logging.info(f"Assign meshes to tissue's dicts")
    tiss_dict = {}
    for t in ts:

        tiss_dict[t] = {}
        tiss_dict[t]["in_msh"     ] = in_tiss_tms[t]

        out_fn = f"{out_fn_ptrn}_{t}_in.stl"
        out_pth = os.path.join(oDir, out_fn)
        
        msh_in = in_tiss_tms[t]
        if args.dbg_files:
            logging.info(f"  Exporting to {out_pth} STL file...")
            with open(out_pth, 'bw') as f:
                msh_in.export( f, "stl")
                        
            tiss_dict[t]["in_msh_pth" ] = out_pth

    #----------------------------------------------------------------------------
    
    # clear ROI remashed mesh - it happens that a duplicated facess occure after remeshing and also some degenerated faces have been produced
    logging.info(f"Merge close points and remove degenerated faces...")
    check_close_points           = True
    check_duplicated_points      = False # i tak nic z tym nie robie
    check_degenerated_faces      = True
    check_overlapping_faces      = True
    check_boundary_dangling_faces= True
    check_boundaryonly_faces     = False
    for t in ts:

        logging.info(f" {t}...")
        check_alternating_faces = t=='skin' or t=='roi' or t=='omega' or t=='bones'
        check_narrow_faces      = t=='skin' or t=='roi' or t=='omega' or t=='bones'
        short_edge_th=0.001
        mesh = tiss_dict[t]["in_msh"     ]#tiss_dict[t]["remeshed_surf"    ]
        faces = mesh.faces#mesh.cells_dict['triangle']
        points = mesh.vertices#mesh.points
        keep_repairing = True
        max_repair_loops = 20
        repair_loop_id = -1
        while(keep_repairing):
            keep_verts_local = None
            if do_adjust_z_at_top:
                ps_z = points[:,2]
                max_z = np.max(ps_z)
                dz_th = pitch/3
                vid_ids = np.where(np.logical_and((ps_z > (max_z - dz_th)), (ps_z < max_z)))[0]
                keep_verts_local = vid_ids
            keep_repairing = False
            repair_loop_id += 1
            new_faces, new_points, pass_id, changed = mesh_repair( faces, points,                 
                keep_verts                      = keep_verts_local, 
                check_close_points              = check_close_points, 
                check_duplicated_points         = check_duplicated_points, 
                check_degenerated_faces         = check_degenerated_faces, 
                check_overlapping_faces         = check_overlapping_faces, 
                check_boundary_dangling_faces   = check_boundary_dangling_faces,
                check_boundaryonly_faces        = check_boundaryonly_faces,
                check_alternating_faces         = check_alternating_faces, 
                check_narrow_faces              = check_narrow_faces, 
                max_passes                      = args.max_passes if (keep_verts_local is None) else 5, 
                short_edge_th                   = short_edge_th, 
                max_adjecency_angle             = args.max_adjecency_angle, 
                stop_when_no_degenerated_overlapping_alternating_faces = True, 
                do_return_merged_points_list    = False, 
                do_remove_unused_points         = True, 
                log_level                       = 1 )

            if len(ts)==1:
                out_fn = in_fn
            else:
                out_fn = f"{out_fn_ptrn}_{t}.stl"
            out_pth = os.path.join(oDir, out_fn)
            tiss_dict[t]["remeshed_surf_pth"] = out_pth
            
            if changed:
                logging.info(f'Changed! Save the mesh without problematic faces to {tiss_dict[t]["remeshed_surf_pth"]} file...')
            else:
                logging.info(f'Not changed! Save the mesh without problematic faces to {tiss_dict[t]["remeshed_surf_pth"]} file...')
                
            if do_adjust_z_at_top and (repair_loop_id < max_repair_loops):
                logging.info(f"Adjust Z coordinate at top cut in order to preserve flat cut surface...")
                new_points, new_faces, changed = adjust_z_at_top(new_points, new_faces, pitch/3)
                if changed:        
                    points = new_points
                    faces  = new_faces
                    keep_repairing = True
                    logging.info(f'  Changed! Save the mesh without Z aligned vertices to {tiss_dict[t]["remeshed_surf_pth"]} file...')
                else:
                    logging.info(f"  No points to change. Skip.")

        #tmesh = trimesh.Trimesh(vertices=points, faces=faces)
        cells_dict          = [('triangle', new_faces   )]
        changed_mesh = meshio.Mesh(points = new_points, cells = cells_dict)
        changed_mesh.write(tiss_dict[t]["remeshed_surf_pth"], binary=True)
        tiss_dict[t]["remeshed_surf"    ] = changed_mesh

    #----------------------------------------------------------------------------
    logging.info(f"Finished!")
    
if __name__=='__main__':
    main()