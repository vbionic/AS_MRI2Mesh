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
from as_bin.utils.offset_verts import get_triangles_normals
#----------------------------------------------------------------------------
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
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
        if (arg[0] == '-'): 
            if arg[1].isdigit(): 
                args_list_to_parse[i] = ' ' + arg
    #----------------------------------------------------------------------------
    parser = ArgumentParser()

    parser.add_argument("-iDir",  "--in_dir",  default = "as_data/st23_preprocessed_meshes/B000004/000003",   help="input directory with *_skin_bones_vessels_mesh_volume.obj and *_skin_mesh_roi.stl files",   metavar="PATH",required=True)
    parser.add_argument("-iPtrn", "--in_pattern",default = '_skin_bones_vessels_mesh_volume.obj'         ,   help="output directory for the result remeshed mesh",      metavar="PATH",required=True)
    parser.add_argument("-oDir",  "--out_dir", default = "as_data/st23_remeshed/B000004/000003"           ,   help="output directory for the result remeshed mesh",      metavar="PATH",required=True)
    parser.add_argument("-r" ,  "--radius"    ,  default = "8", help="smooth radius",                          required=False)
    parser.add_argument("-s" ,  "--sign"      ,  help="offset sign [+1, -1]",                   required=True)
    parser.add_argument("-th",  "--threads"   ,type = int , default = -2       ,help="Number of simultaneous processes",       required=False)
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
    radius      = args.radius
    sign        = args.sign
    threads     = args.threads
    
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
        
    in_fn = os.path.basename(in_msh_pths[0])
    in_fn_s = in_fn.split("_",2)
    #out_fn_ptrn = f"{in_fn_s[0]}_{in_fn_s[1]}" if len(in_fn_s)==3 else in_fn.split(".",1)[0]
    out_fn_ptrn = in_fn.split(".",1)[0]
    out_fn = f"{out_fn_ptrn}.stl"
    out_pth = os.path.join(oDir, out_fn)


    cmdline   = f'python3 ./as_bin/utils/smoothenSTL.py'
    cmdline  += f' -if {in_msh_pths[0]}'
    cmdline  += f' -of {out_pth}'
    cmdline  += f' -r {radius}'
    cmdline  += f" -s {sign}"
    cmdline  += f" -th {threads}"
       
    logging.info(f"   {cmdline}")

    ret = os.system(cmdline)


    logging.info(f"Finished!")
    sys.exit(ret)
    
if __name__=='__main__':
    main()