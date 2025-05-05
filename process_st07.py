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
stage07_dir     = 'as_data/st07_postprocess'
stage06_dir     = 'as_data/st06_shape_correction'
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
def Run_postprocess_tissues(verbose, ses):
    plpath  = os.path.normpath('as_bin/st07_postprocess/as_postprocessTissues.py ')

    idpath  = os.path.normpath(os.path.join(stage06_dir, ses))
    odpath  = os.path.normpath(os.path.join(stage07_dir, ses))
    
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)
    

    logging.info('path to script: {}'.format(plpath))

    log_run ('CorrectShapes - Skin', ses)

    cmd    = 'python3 '    + plpath 
    cmd   += '-v '         + verbose
    cmd   += ' -oDir '     + odpath
    cmd   += ' -iDir '     + idpath
    cmd   += ' -conf '     + 'as_cfg/st07_postprocess/configuration.json'

    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('Postprocess Tissues', ses)
   
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
    data_dir = stage07_dir
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
    op_l = ['all', 'skin', 'position']

    verbose         = 'off'             if args.verbose is None else args.verbose
    op              = 'all'             if args.op      is None else args.op
    
    if not (op in op_l):
        logging.error("operation filter {} does not match any of expected vals: {}".format(op, op_l))
        exit(1)
        
    #---------------------------------------------------------------------------------------------------------------------------------------

    session_l = expand_session_dirs(args.session_id, 'as_data/st06_shape_correction/')

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
            Run_postprocess_tissues(verbose, ses)
            
        elif op == 'postprocess':
            Run_postprocess_tissues(verbose, ses)


         
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
