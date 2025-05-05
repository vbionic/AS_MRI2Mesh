import sys, os
import pathlib
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
import os
import glob
import tracemalloc
import multiprocessing as mp
import timeit
import time
import shutil
import logging
import json 
import cv2
#-----------------------------------------------------------------------------------------
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe
from v_utils.v_dataset import expand_session_dirs
#---------------------------------------------------------
from argparse           import ArgumentParser
#--------------------------------------------------------
input_dir       = 'as_input'
stage03_dir     = 'as_data/st03_preprocessed_images'
stage06_dir     = 'as_data/st06_shape_correction'
stage07_dir     = 'as_data/st07_postprocess'
stage08_dir     = 'as_data/st08_filling_voids'
#--------------------------------------------------------
run_list        = []
err_list        = []
#--------------------------------------------------------

def log_err(name, ses):
    err 			= {}
    err['plugin'] 	= name
    err['session'] 	= ses
        
    err_list.append(err)

def log_run(name, ses):
    run 			= {}
    run['plugin'] 	= name
    run['session'] 	= ses
        
    run_list.append(run)
    
#-----------------------------------------------------------

#-----------------------------------------------------------
def Run_fill_voids(verbose, ses):

    plpath  = os.path.normpath('as_bin/st08_filling_voids/as_fillVoids.py ')

    idpath  = os.path.normpath(os.path.join(stage07_dir, ses))
    ippath  = os.path.normpath(os.path.join(stage06_dir, ses)) #directory with probablility images and grayscale images - no sense in copying them to st07, so they stay in st06
    odpath  = os.path.normpath(os.path.join(stage08_dir, ses))
    
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)
        
    

    logging.info('path to script: {}'.format(plpath))

    log_run ('Filling voids', ses)

    cmd    = 'python3 '    + plpath 
    cmd   += '-v '         + verbose
    cmd   += ' -oDir '     + odpath
    cmd   += ' -iDir '     + idpath
    cmd   += ' -pDir '     + ippath
    cmd   += ' -conf '     + 'as_cfg/st08_filling_voids/configuration.json'

    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('Filling voids', ses)
   

    return
    
    
def Run_detect_straight(verbose, ses):
    
    plpath  = os.path.normpath('as_bin/st08_filling_voids/as_detectStraight.py ')

    idpath  = os.path.normpath(os.path.join(stage08_dir, ses))
    odpath  = os.path.normpath(os.path.join(stage08_dir, ses))
    parampath  = os.path.normpath(os.path.join(stage03_dir, ses))
    
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)

    logging.info('path to script: {}'.format(plpath))

    log_run ('Detect straight lines', ses)

    cmd    = 'python3 '    + plpath 
    cmd   += '-v '         + verbose
    cmd   += ' -oDir '     + odpath
    cmd   += ' -iDir '     + idpath
    cmd   += ' -pDir '     + parampath


    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('Detecting straight lines', ses)
   

    return
    
    
def Run_show_straight(verbose, ses):
    
    plpath  = os.path.normpath('as_bin/st08_filling_voids/as_showStraight.py ')

    idpath  = os.path.normpath(os.path.join(stage08_dir, ses))


    logging.info('path to script: {}'.format(plpath))

    log_run ('Show straight lines', ses)

    cmd    = 'python3 '    + plpath 
    cmd   += '-v '         + verbose
    cmd   += ' -iDir '     + idpath


    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('Showing straight lines', ses)
   

    return
    
#-----------------------------------------------------------
def Run_create_polys(verbose, ses):

    odpath  = os.path.normpath(os.path.join(stage08_dir, ses))
    
    logging.info(f'Convert labels PNG to polygons JSON for {odpath}')

    log_run ('create polys', ses)
    
    
    for cls in ["bones", "fat", "muscles", "skin", "vessels", "roi", "unknown"]:
        logging.info(f' Class "{cls}"')
        
        odpath_cls  = os.path.normpath(os.path.join(odpath, cls))
        
        if not os.path.isdir(odpath):
            logging.warning(f' Class folder "{odpath_cls}" not found! Put it on the error list')
            log_err ('create polys', ses + f"_{cls}")
        else:
            labels_ptrn = os.path.join(odpath_cls, "*labels.png")
            labels_fns = glob.glob(labels_ptrn)
            
            labels_fns.sort()
            logging.info(f'  Found {len(labels_fns)} images to convert...')
            if(len(labels_fns) == 0):
                logging.warning(f' 0 labels found in "{odpath_cls}"! Put it on the error list')
                log_err ('create polys', ses + f"_{cls}")
            
            for label_fn in labels_fns:
                
                img_f = Image.open(label_fn)
                my_polygons = v_polygons.from_image(img_f)
                
                poly_fn = label_fn.replace("_labels.png", "_polygons.json")
                
                jsonDumpSafe(poly_fn, my_polygons.as_dict())

    return



#---------------------------------------------------------
# main
#---------------------------------------------------------
def main():

    parser          = ArgumentParser()

    parser.add_argument("-ses",    "--session_id"       , default = "*/*"   ,help="session id", nargs='+',	metavar="PATH",	required=False)
                                                                                                                                   
    parser.add_argument("-op",     "--operation"        , dest="op"      ,help="operation",                                        required=False)
    parser.add_argument("-v",      "--verbose"          , dest="verbose" ,help="verbose level <off, on/dbg>",                      required=False)

    args            = parser.parse_args()
    #---------------------------------------------------------------------------------------------------------------------------------------
    # initialize logging 
    data_dir = stage08_dir
    script_name = os.path.basename(__file__).split(".")[0]
    from datetime import datetime
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    log_dir = f"{data_dir}/_log"
    try:
        if not os.path.isdir(log_dir):
            pathlib.Path(log_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(log_dir, err))
        exit(1)
    initial_log_fn = f"{log_dir}/_dbg_{script_name}_{time_str}_pid{os.getpid()}.log"
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging_level = logging.INFO if ((args.verbose is None) or (args.verbose == "off")) else logging.DEBUG
    logging.basicConfig(level=logging_level, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    try:
        os.chmod(initial_log_fn, 0o666)
    except:
        logging.warning(f"Could not change log file permitions {initial_log_fn}. I'm not the owner of the file?")
    
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")

    #---------------------------------------------------------------------------------------------------------------------------------------
    op_l = ['all', 'fill_voids', 'create_polys', 'detect_straight', 'show_straight']

    verbose         = 'off'             if args.verbose is None else args.verbose
    op              = 'all'             if args.op      is None else args.op
    
    if not (op in op_l):
        logging.error("operation filter {} does not match any of expected vals: {}".format(op, op_l))
        exit(1)
        
    #---------------------------------------------------------------------------------------------------------------------------------------

    session_l = expand_session_dirs(args.session_id, stage07_dir)

    #---------------------------------------------------------------------------------------------------------------------------------------

    logging.info('      > -----------------------------------------------------')
    logging.info('      > sessions list to process:')
    for ses in session_l:
        logging.info('      >    '+ ses)
    logging.info('      > -----------------------------------------------------')

    for ses in session_l:
        logging.info('      > starting to process : '+ ses)

        logging.info('=' * 50)
        logging.info(' working set   : %s'%(ses))
        logging.info(' current dir   : %s'%os.getcwd())
        logging.info(' process       : %s'%op)
        logging.info(' verbose level : %s'%verbose)
        logging.info('-' * 50)

        if op == 'all':
            Run_fill_voids       (verbose, ses)
            Run_create_polys     (verbose, ses)
            #Run_detect_straight  (verbose, ses)
            #Run_show_straight    (verbose, ses)
            
        elif op == 'create_polys':
            Run_create_polys(verbose, ses)
            
        elif op == 'fill_voids':
            Run_fill_voids(verbose, ses)
            
        elif op == 'detect_straight':
            Run_detect_straight(verbose, ses)
            
        elif op == 'show_straight':
            Run_show_straight(verbose, ses)
                
            


         
    logging.info('-' * 50)
    logging.info("RUNS   : {}".format(len(run_list)))
    logging.info("ERRORS : {}".format(len(err_list)))
    if len(err_list):
        logging.error("ERROR LIST : ")
        for e in err_list:
            logging.error(e)
    logging.info('*' * 50)
    if len(err_list):
        exit(-1)

#-----------------------------------------------------------
    
if __name__ == '__main__':
    main()
