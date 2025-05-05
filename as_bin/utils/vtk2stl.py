import sys, os
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from datetime import datetime
from pandas.core.common import flatten
from pathlib import Path    
import glob
import logging
import random
import copy
import time
import json
#---------------------------------------------------------
import numpy as np
import trimesh
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_arg import arg2boolAct
from v_utils.v_arg import convert_dict_to_cmd_line_args, convert_cmd_line_args_to_dict, convert_cfg_files_to_dicts
from v_utils.v_arg import print_cfg_list, print_cfg_dict

#---------------------------------------------------------
from argparse   import ArgumentParser

from sfepy.discrete.fem import Mesh
from sfepy.postprocess.utils_vtk import get_vtk_from_mesh,\
    get_vtk_surface, write_vtk_to_file,\
    tetrahedralize_vtk_mesh
from as_bin.utils.dir_utils import createDirIfNeeded
from as_bin.utils.fem_utils import create_surf_from_mesh
from as_bin.utils.fem_utils import tid2tname, tname2tid
#----------------------------------------------------------------------------
def main():
    
    # initialize logging 
    script_name = os.path.basename(__file__).split(".")[0]
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    from sfepy.base.base import output
    #output = Output('my sfepy:')
    output.set_output_prefix('    sfp:')
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

    parser.add_argument("-iVPre", "--in_vtk_prefix",default = "as_data/st28_voxels/",  help="input VTK file prefix or path",   metavar="PATH",required=True)    
    parser.add_argument("-oDir",  "--out_dir",      default = None,      help="if the output directory is given then the results are copied to the directory, if not then STL file is created in the VTK file directory",metavar="PATH",required=False)
    parser.add_argument("-ts",    "--tissue_names", default = ["omega"], nargs='*',    help="Name of tissues ", required=False)
    parser.add_argument(          "--dbg_files",    default=False, action=arg2boolAct, help="Leave all files for debug", required=False)
    
    
    logging.info('-' * 50)
    if not(("-h" in sys.argv) or ("--help" in sys.argv)): 
        # get training arguments
        args, rem_args = parser.parse_known_args(args_list_to_parse)
        
        logging.info("Parsed configuration arguments:")
        args_d = vars(args)
        print_cfg_dict(args_d, indent = 1, skip_comments = True)
        
    else: 
        # help
        logging.info("Params:")
        logging.info(parser.format_help())
        sys.exit(1)
    
    #----------------------------------------------------------------------------
    doCreateAtODir = not args.out_dir is None
    if doCreateAtODir:
        oDir = os.path.normpath(args.out_dir)
    else:
        oDir = os.path.normpath(os.getcwd())
    #----------------------------------------------------------------------------
    # redirect logging file to work directory 
    from as_bin.utils.logging_utils import redirect_log
    lDir = redirect_log(oDir, f"_{script_name}_{time_str}_pid{os.getpid()}.log", f"_{script_name}_last.log")
    logging.info('-' * 50)
    #----------------------------------------------------------------------------
    logging.info("Processing arguments...")

    pth = args.in_vtk_prefix#sys.argv[1:]  # the first argument is the script itself
    if os.path.isfile(pth) :
        logging.info(f" path is already a file")
        in_mesh_pths = [pth]
    else:#elif os.path.isdir(pth) :
        logging.info(f" path is not a file -> search for vtk file recursively at this path")
        if doCreateAtODir:
            pth_ptrns = [f"{pth}*.vtk"]
        else:
            pth_ptrns = [f"{pth}*.vtk",f"{pth}/**/*.vtk"]
        in_mesh_pths = []
        for pth_ptrn in pth_ptrns:
            logging.info(f"  search files using pattern: {pth_ptrn}")
            in_mesh_pths_c = glob.glob(pth_ptrn, recursive=True)
            in_mesh_pths.extend(in_mesh_pths_c)
        in_mesh_pths = list(set(in_mesh_pths))
    logging.info(f" Found: {in_mesh_pths}")
        
    for in_mesh_pth in in_mesh_pths:

        input_mesh = Mesh.from_file(in_mesh_pth)
        logging.info(f"{in_mesh_pth}")

        for tissue_name in args.tissue_names:
            if doCreateAtODir:
                in_mesh_fn = Path(in_mesh_pth).name
                out_fn = in_mesh_fn.replace(".vtk", f"_{tissue_name}.stl")
                out_pth = os.path.join(oDir, out_fn)
            else:
                out_pth = in_mesh_pth.replace(".vtk", f"_{tissue_name}.stl")

            logging.info(f"  {in_mesh_pth} -> {out_pth}")

            surf_mesh = create_surf_from_mesh(input_mesh, tissue_name, tname2tid, work_dir = oDir, dbg_files=False, out_fn_prefix = None)

            verts = surf_mesh.coors
            faces = surf_mesh.get_conn('2_3')

            meshes_surf = trimesh.Trimesh(vertices=verts, faces=faces)#, vertex_normals = vertex_normals, face_normals = face_normals)
            meshes_surf.export(out_pth)
    
if __name__ == '__main__':
    main()