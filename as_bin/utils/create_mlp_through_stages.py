#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy
import trimesh
from argparse import ArgumentParser
import os, sys
import pathlib
from os import listdir
from os.path import isfile, isdir
import glob
import time
from time import gmtime, strftime
from datetime import datetime
import logging
from pandas.core.common import flatten
import shutil

#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_contour import *
from v_utils.v_polygons import *
from v_utils.v_arg import print_cfg_list, print_cfg_dict
from v_utils.v_arg import arg2boolAct
from v_utils.v_arg import convert_dict_to_cmd_line_args, convert_cmd_line_args_to_dict, convert_cfg_files_to_dicts
from as_bin.utils.meshlab_utils import mlp_parse, export_meshlab_mlp
from as_bin.utils.logging_utils import redirect_log
from v_utils.v_dataset import expand_session_dirs
##############################################################################
# MAIN
##############################################################################
if __name__ == '__main__':

    np.random.seed(13)
    start_time_total = time.time()
    #----------------------------------------------------------------------------
    # initialize logging 
    script_name = os.path.basename(__file__).split(".")[0]
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    initial_log_fn = f"_initial_{script_name}_{time_str}_pid{os.getpid()}.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])

    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    logging.info("initial log file is {}".format(initial_log_fn))
    #----------------------------------------------------------------------------
    parser = ArgumentParser()
    logging.info('- ' * 25)
    logging.info("Reading configuration...")
    logging.info("Command line arguments:")
    logging.info(" {}".format(' '.join(sys.argv)))
    cmd_line_args_rem = sys.argv[1:]
    cfgfl_dict_prs = {}
    #-----------------------------------------------------------
    # get middle priority arguments from config files
    logging.info('- ' * 25)
    logging.info("Try read config files with middle level priority arguments...")    
    cfa = parser.add_argument_group('config_file_arguments')
    cfa.add_argument("--cfg" , default=[], action='append', type=str, nargs='*', required=False, metavar="PATH", help="one or more config json filenames. Further config files have higher priority. Command line arguments have the highest priority.", )
    
    if not(("-h" in sys.argv) or ("--help" in sys.argv)):
        cfg_fns_args, cmd_line_args_rem = parser.parse_known_args(cmd_line_args_rem); # bez error gdy natrafi na nieznany parametr (odwrotnie niż "parse_args()")
            
        cfgs = list(map(str, flatten(cfg_fns_args.cfg)))
        # read dictonaries from config files (create a list of dicts)
        cfg_dicts = convert_cfg_files_to_dicts(cfgs)

        # merge all config dicts - later ones will overwrite entries with the same keys from the former ones
        for cfg_dict in cfg_dicts:
            cfgfl_dict_prs.update(cfg_dict)
    
    #----------------------------------------------------------------------------
    # gather all the arguments
    logging.info('- ' * 25)
    logging.info("Collect all the arguments from the model file, config files and command line...")

    # convert cmd_line_args_rem to dictionary so we can use it to update content of the dictonaries from config files
    cmd_line_args_rem_dict = convert_cmd_line_args_to_dict(cmd_line_args_rem)

    # update argument dictionary with the increasing priority:
    params_dict = {}

    # 1) delete keys that should use defaut values:
    logging.debug(" 1) delete keys that should use defaut values...")
    # 2) middle priority arguments from config files 
    logging.debug(" 2) middle priority arguments from config files ...")
    params_dict.update(cfgfl_dict_prs)
    # 3) high priority command line arguments
    logging.debug(" 3) high priority command line arguments ...")
    params_dict.update(cmd_line_args_rem_dict)

    logging.info("Merged arguments:")
    cfg_d = params_dict
    print_cfg_dict(cfg_d, indent = 1, skip_comments = True)

    # parse the merged arguments dictionary
    args_list_to_parse = convert_dict_to_cmd_line_args(params_dict)
    #----------------------------------------------------------------------------
    # add parser parameters that are specific to this script  
    _logging_levels = logging._levelToName.keys()
    _engines = ["blender", "scad"]

    parser = ArgumentParser()
    parser.add_argument("-iDir",  "--in_dir"                , default="as_data/"                        , type=str          , required=False, metavar="PATH", help="shell stl directory root")
    parser.add_argument("--include_patterns"                , default=[".stl"]                         , type=str,nargs='+', required=False, metavar="P", help="filename must include one of those")
    parser.add_argument("--exclude_patterns"                , default=["_in.stl", "ball1", "box1", "cylinder1"]    , type=str,nargs='+', required=False, metavar="P", help="filename must NOT include any of those")
    parser.add_argument("--exclude_subses_dir_patterns"     , default=["_tmp", "_log"]    , type=str,nargs='+', required=False, metavar="P", help="dir must NOT include any of those")
    #parser.add_argument("--include_stage_dirs_filter"       , default=None, type=str,nargs='+', required=False, metavar="P", help="filename must include one of those")
    parser.add_argument("--include_stage_dirs_filter"       , default=["st63", "st73", "st8", "st9", "st100",], type=str,nargs='+', required=False, metavar="P", help="filename must include one of those")
    parser.add_argument("--in_dir_filter"                   , default="st"                           , type=str          , required=False, metavar="PATH", help="")
    parser.add_argument("-ses",  "--session_id"             , default = ["B*/*", "X*/*"]   ,help="session id", nargs='+',  metavar="PATH", required=False)
    parser.add_argument("--out_dir",   "-od"                , default="as_data/stX_mlp_through_stages" , type=str          , required=False, metavar="PATH", help="directory in which directorys with output files will be saved")
    parser.add_argument("--logging_level"                   , default=logging.INFO                      , type=int          , required=False, choices=_logging_levels,     help="")
    parser.add_argument("--only_one_bones_mesh"             , default=True   , action=arg2boolAct, required=False, metavar='B'   , help="pokazuje kolejne kroki tworzonych obiektow")
    parser.add_argument("--copy_meshes_local"               , default=True   , action=arg2boolAct, required=False, metavar='B'   , help="pokazuje kolejne kroki tworzonych obiektow")
                                                            
    #-----------------------------------------------------------------------------------------
    parser.add_argument("--show_steps"                      , default=False   , action=arg2boolAct, required=False, metavar='B'   , help="pokazuje kolejne kroki tworzonych obiektow")
    parser.add_argument(          "--dbg_files",    default=False, action=arg2boolAct, help="Leave all files for debug", required=False)
    #-----------------------------------------------------------------------------------------

    if (("-h" in sys.argv) or ("--help" in sys.argv)):
        # help
        logging.info("Params:")
        logging.info(parser.format_help())
        sys.exit(1)
    else:
        # get evaluation arguments
        logging.info('- ' * 25)
        logging.info("Parse the arguments...")
        args, rem_args = parser.parse_known_args(args_list_to_parse)
        
    #----------------------------------------------------------------------------
    if(len(rem_args) != 0):
        logging.error("After all modules parsed own arguments some arguments are left: {}.".format(rem_args))
        sys.exit(10)
        
    #----------------------------------------------------------------------------
    include_patterns     = args.include_patterns
    rootDir 		   = os.path.normpath(args.in_dir )
    oDir 		   = os.path.normpath(args.out_dir)
    
    if not os.path.isdir(rootDir):
        logging.error(f'Input directory {rootDir} not found !')
        exit(1)
    #----------------------------------------------------------------------------
    # redirect logging file to work directory 
    logging.info('-' * 50)
    work_dir = os.path.normpath(oDir)
    logging.info('Redirect logging file to directory {}'.format(work_dir))
    # create work dir
    try:
        if not os.path.isdir(work_dir):
            pathlib.Path(work_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
            logging.info("Created dir {}".format(work_dir))
    except Exception as err:
        logging.error("Creating dir ({}) IO error: {}".format(work_dir, err))
        sys.exit(1)
        
    #----------------------------------------------------------------------------
    # new logging file handler    
    lDir = redirect_log(work_dir, f"_{script_name}_{time_str}.log", f"_{script_name}_last.log")
    logging.info('-' * 50)
    
    #----------------------------------------------------------------------------
    logging.info(f" Input dir                        : {rootDir   }")
    logging.info(f" Output dir                       : {work_dir   }")
    #----------------------------------------------------------------------------
    logging.info('-' * 50)
    output_dir      =  work_dir
    
    onlydirs        = [os.path.join(rootDir, f) for f in listdir(rootDir) if isdir(os.path.join(rootDir, f))]
    onlydirsSplited = [os.path.split(d)[1] for d in       onlydirs]
    onlydirsSplited_match = np.zeros(len(onlydirsSplited), dtype=bool)
    for did, cdir in enumerate(onlydirsSplited):
        for sp in os.path.split(cdir):
            if(sp.find(args.in_dir_filter)==0):
                onlydirsSplited_match[did] = True
                break
    logging.info("Dirs fould at root:")
    for did, d in enumerate(onlydirs):
        logging.info(f"   {d}\t\t{'*match in_dir_filter' if onlydirsSplited_match[did] else ''}")
        
    onlydirsfilter  = [d for did, d in enumerate(onlydirsSplited) if onlydirsSplited_match[did]]
    onlydirsfilter  = [d for d in onlydirsfilter if d.find(output_dir)==-1]

    if not args.include_stage_dirs_filter is None:
        logging.info("Dirs filtered by include_stage_dirs_filter:")
        
        onlydirsfilter_filter = np.zeros(len(onlydirsfilter), dtype=bool)
        for fid, f in enumerate(onlydirsfilter):
            onlydirsfilter_filter[fid] = np.any([f.find(ep)!=-1 for ep in args.include_stage_dirs_filter])
            
        onlydirsfilter = [f for fid, f in enumerate(onlydirsfilter) if onlydirsfilter_filter[fid]]
        for did, d in enumerate(onlydirsfilter):
            logging.info(f"   {d}")

    onlydirsfilter_sorted = []
    onlydirsfilter_ids = np.ones(len(onlydirsfilter))*-1
    for stDir_id, stDir in enumerate(onlydirsfilter):
        if stDir.find("st")==0:
            end = stDir.find("_")
            try:
                stId_str = stDir[2:end]
                stId = int(stId_str)
                onlydirsfilter_ids[stDir_id] = stId
            except:
                onlydirsfilter_ids[stDir_id] = -1
    
    onlydirsfilter_idxed = [(onlydirsfilter_ids[i], onlydirsfilter[i]) for i in range(len(onlydirsfilter_ids))]
    onlydirsfilter_idxed = sorted(onlydirsfilter_idxed, key=lambda tup: tup[0])

    logging.info("Dirs found at root filtered and sorted:")
    for did, d in onlydirsfilter_idxed:
        logging.info(f"  {did:.0f}: {d}")
    #----------------------------------------------------------------------------
    logging.info('-' * 50)
    logging.info(f"Walking through dirs at root...")

    database_dict = {}
    for stId, stDir in onlydirsfilter_idxed:
        logging.info(f" {stDir}...")
        
        stPth = os.path.join(rootDir, stDir)
        session_l = expand_session_dirs(args.session_id, stPth)

        for ses in session_l:
            logging.info(f"  {ses}...")
            ses_database_dict = {}
            ses_database_dict["stage_pth"] = stPth
            ses_database_dict["stage_dir"] = stDir
            ses_database_dict["stage_id" ] = stId
            

            ses_pth = os.path.join(stPth, ses)
            ses_database_dict["ses_pth"] = ses_pth
            ses_database_dict["ses_dir"] = ses
            ses_database_dict["ses_name"] = ses.replace("\\", "_").replace("/", "_").replace("__", "_")
            ses_database_dict["file_fn_l"      ] = []
            ses_database_dict["file_fn_short_l"] = []
            ses_database_dict["file_suf_d_l"   ] = []
            ses_database_dict["file_pth_l"     ] = []
            ses_database_dict["mesh_name_l"    ] = []
            depth = 3
            while depth > 0:
                ptrn = f"{ses_pth}" + depth*"/*"
                logging.info(f"   searching {ptrn}...")
                files = glob.glob(ptrn)
                if len(files) > 0:
                    subSes_file_l = [f for f in files if isfile(f)]
                    subSes_file_fn_l = [os.path.split(f)[1] for f in subSes_file_l]
                    subSes_file_d_l  = [os.path.split(os.path.relpath(f, ses_pth))[0] for f in subSes_file_l]
                    subSes_file_l_inc_filter = np.zeros(len(subSes_file_fn_l), dtype=bool)
                    subSes_file_l_exc_filter = np.zeros(len(subSes_file_fn_l), dtype=bool)
                    subSes_dir_l_exc_filter = np.zeros(len(subSes_file_d_l), dtype=bool)
                    for fid, f in enumerate(subSes_file_fn_l):
                        subSes_file_l_inc_filter[fid] = np.any([f.find(ep)!=-1 for ep in args.include_patterns])
                        subSes_file_l_exc_filter[fid] = np.any([f.find(ep)!=-1 for ep in args.exclude_patterns])
                    for did, d in enumerate(subSes_file_d_l):
                        subSes_dir_l_exc_filter [did] = np.any([d.find(ep)!=-1 for ep in args.exclude_subses_dir_patterns])
                    _=1
                    for fid, _ in enumerate(subSes_file_l_inc_filter):
                        filter = subSes_file_l_inc_filter[fid] and not subSes_file_l_exc_filter[fid] and not subSes_dir_l_exc_filter[fid]
                        if filter:
                            fn = subSes_file_fn_l[fid]
                            ses_database_dict["file_fn_l"   ].append(fn)
                            fn_short = fn.replace(ses_database_dict["ses_name"],"")
                            if(fn_short[0]=="_"):
                                fn_short = fn_short[1:]
                            dot_pos = fn_short.find(".")
                            fn_short = fn_short[:dot_pos]
                            ses_database_dict["file_fn_short_l"].append(fn_short)

                            sufDir = subSes_file_d_l [fid]
                            ses_database_dict["file_suf_d_l"].append(sufDir)
                            fpth = os.path.join(ses_pth, sufDir, fn)
                            ses_database_dict["file_pth_l"  ].append(fpth)

                            sufDir_name = sufDir.replace("\\", "_").replace("/", "_").replace("__", "_")
                            #mesh_name_short = f"{stId:.0f}_{sufDir_name}_{fn_short}"
                            mesh_name_short = f"{stDir}__{sufDir_name}__{fn_short}"
                            ses_database_dict["mesh_name_l"].append(mesh_name_short)
                        
                        logging.info(f"   {subSes_file_l[fid]}\t\t{'*satisfy include_patterns and exclude_patterns' if filter else ''}")
                depth -= 1
            if not ses in database_dict.keys():
                database_dict[ses] = []
            database_dict[ses].append(ses_database_dict)

    #----------------------------------------------------------------------------
    logging.info('=' * 50)
    logging.info(f"Removing empty results...")
    for ses in database_dict.keys():
        logging.info(f' Remove empty stages for {ses}: {[sd["ses_pth"] for sd in database_dict[ses] if len(sd["file_fn_l"]) == 0]}') 
        database_dict[ses] = [sd for sd in database_dict[ses] if len(sd["file_fn_l"]) > 0]
            
    sess2remove = [ses for ses,sesd_l in database_dict.items() if len(sesd_l) == 0]
    database_dict = {ses:sesd_l for ses,sesd_l in database_dict.items() if not ses in sess2remove}
    for ses in sess2remove:
        logging.info(f' Removed whole {ses} due to it empty file list') 

    #----------------------------------------------------------------------------
    logging.info('=' * 50)
    logging.info(f"Filter results by sessions")
    for ses in database_dict.keys():
        logging.info(f"{ses}")
        ses_database_dict = database_dict[ses]
        ses_o_dir = os.path.join(work_dir, ses)
        if len(ses_database_dict) == 0:
            continue
        # create work dir
        try:
            if not os.path.isdir(ses_o_dir):
                pathlib.Path(ses_o_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            sys.exit(1)
        ses_name = ses_database_dict[0]["ses_name"]
        out_ml_features_pth = os.path.join(ses_o_dir, f"{ses_name}.mlp")
        ses_mlp_meshes = []
        has_bones = False
        for ses_stage_dict in ses_database_dict:
            has_liner_or_shell = False
            for mid in range(len(ses_stage_dict['file_pth_l'])):
                mname = ses_stage_dict['mesh_name_l'][mid]
                if mname.find("liner") != -1 or mname.find("shell") != -1:
                    has_liner_or_shell = True
            for mid in range(len(ses_stage_dict['file_pth_l'])):
                mpth  = ses_stage_dict['file_pth_l' ][mid]
                mname = ses_stage_dict['mesh_name_l'][mid]
                mname_bez_kropki = mname.replace(".","")

                is_bone = mname.find("bones") != -1
                is_vessels = mname.find("vessels") != -1

                if args.only_one_bones_mesh and has_bones and is_bone:
                    continue

                if mname.find("liner") != -1:
                    color= "139 69 19 255" #saddlebrown
                    is_solid = True
                    visible = "1"
                elif mname.find("shell") != -1:
                    color= "50 50 50 255" #dark
                    is_solid = True
                    visible = "1"
                elif is_vessels:
                    color= "153 0 0 255" #red darker,
                    is_solid = True
                    visible = "0"
                elif is_bone:
                    color= "200 200 200 255"
                    is_solid = True
                    visible = "1"
                else:
                    color= "100 100 100 255" #grey
                    is_solid = not has_liner_or_shell
                    visible = "0"

                if args.copy_meshes_local:
                    local_mpth = os.path.join(ses_o_dir, mpth)
                    os.makedirs(os.path.dirname(local_mpth), exist_ok=True)
                    shutil.copy(mpth, local_mpth)
                    mpth = local_mpth

                logging.info(f"   {mname_bez_kropki}:\t{mpth}")
                ses_mlp_meshes.append({
                    "color": color,
                    "filename": mpth,
                    "visible": visible,
                    #"is_wireframe": True,
                    "is_solid": is_solid,
                    "named_data_serie": {
                        mname_bez_kropki: [0,0,0],
                        }
                })
                if is_bone:
                    has_bones = True

        logging.info(f" Exporting MeshLab project (MLP) to {out_ml_features_pth}...")
        export_meshlab_mlp(out_ml_features_pth, primitives_dicts = [*ses_mlp_meshes])
        logging.info('=' * 50)

    
    ##############################################################################
    elapsed_time = time.time() - start_time_total
    logging.info("Total time " + str(round(elapsed_time, 2)) + "s")

    logging.info("="*50)
    

