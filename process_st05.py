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
import tracemalloc
import multiprocessing
import timeit
import shutil
import logging
import copy
import time
import json
#---------------------------------------------------------
from argparse   import ArgumentParser
#-----------------------------------------------------------------------------------------
from v_utils.v_json import jsonDumpSafe
from v_utils.v_dataset import MRIDataset, expand_session_dirs
from v_utils.v_arg import print_cfg_list, print_cfg_dict
#--------------------------------------------------------
stage03_dir     = 'as_data/st03_preprocessed_images'
stage05_dir     = 'as_data/st05_evaluated_shapes/'
#--------------------------------------------------------
run_list        = []
err_list        = []
#--------------------------------------------------------

def log_err(name, pth, ses):
    err 			= {}
    err['plugin'] 	= name
    err['pth']  	= pth
    err['session'] 	= ses
        
    err_list.append(err)

def log_run(name, pth, ses):
    run 			= {}
    run['plugin'] 	= name
    run['pth']  	= pth
    run['session'] 	= ses
        
    run_list.append(run)

#---------------------------------------------------------

def copy_roi(iDir, oDir):

  roi_src = os.path.normpath(iDir + "/roi")
  roi_dst = os.path.normpath(oDir + "/roi")
  
  if os.path.isdir(roi_dst):
      shutil.rmtree(roi_dst)
  shutil.copytree(roi_src, roi_dst, symlinks=False, ignore=None)
  
#---------------------------------------------------------

def execute(iDir, oDir, pDir, gpu_id):
     
    log_run ('EVAL', pDir, oDir)

    pth_cfg_f 	= open(pDir + "/" + "eval.json","r");     
    pth_cfg     = json.load(pth_cfg_f)

    for ev in pth_cfg["process"]:        
    
        pth_name  = ev["pth"];    
        pth_out   = ev["out"]
    
        logging.info('      > pth: {}'.format(pth_name))
        logging.info('      > classes: {}'.format(pth_out))
        
        cmdline   = 'python flexnet/evaluation/eval_pth.py'
        cmdline  += ' -oDir {} '.format(os.path.normpath(oDir))
        cmdline  += ' -pth {} '.format(os.path.normpath(pDir+"/"+pth_name))
        cmdline  += ' -sP '
        cmdline  += ' -sN '
        cmdline  += ' -sL '
        cmdline  += ' -sC '
        cmdline  += ' -oShp ' + " ".join(pth_out)  
        cmdline  += ' -iDir {} '.format(os.path.normpath(iDir))

        if "roi" in ev["out"]:
            cmdline  += ' -iDir {} '.format(os.path.normpath("as_data/st03_preprocessed_images"))
            cmdline  += ' -plim 1 '
        else:
            cmdline  += ' -plim 32 '

        cmdline  += ' --force_gpu_id ' + str(gpu_id) + ' '

        logging.info('      > cmdline: {}'.format(cmdline))

        log_run ('EVAL', pth_name, oDir)

        ret = os.system(cmdline)

        if ret:
            log_err ('EVAL', pth_name, oDir)
        
#---------------------------------------------------------
# main
#---------------------------------------------------------

def main():

    parser          = ArgumentParser()

    parser.add_argument("-ses",    "--session_id"       , default = "*/*"   ,help="session id", nargs='+',	metavar="PATH",	required=False)

    parser.add_argument("-v",      "--verbose"          , dest="verbose"    ,help="verbose level",							required=False)

    args            = parser.parse_args()
    #---------------------------------------------------------------------------------------------------------------------------------------
    # initialize logging 
    data_dir = stage05_dir
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

    session_l = expand_session_dirs(args.session_id, stage03_dir)

    #---------------------------------------------------------------------------------------------------------------------------------------

    logging.info('      > -----------------------------------------------------')
    logging.info('      > sessions list to process:')
    for ses in session_l:
        logging.info('      >    '+ ses)
    logging.info('      > -----------------------------------------------------')

    for ses in session_l:
        logging.info('      > starting to process : '+ ses)

        imgDir 	    = os.path.normpath(os.path.join(stage03_dir, ses))
        outDir 		= os.path.normpath(os.path.join(stage05_dir, ses))

        try:
            if not os.path.isdir(outDir):
                pathlib.Path(outDir).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('      > creating "%s" directory failed, error "%s"'%(outDir, err))
            sys.exit(1)

        logging.info('      > -----------------------------------------------------')
        logging.info('      > working set   : %s'%(ses))
        logging.info('      > current dir   : %s'%os.getcwd())
        logging.info('      > verbose level : %s'%args.verbose)
        logging.info('      > subdirectory  :')
        logging.info('      >    input images (png from dicom) : '+ imgDir)
        logging.info('      >    output shapes and metadata    : '+ outDir)

        # kopiowanie ROI z st03 (wczesniej wygenerowane ROI (w st02) i przyciete w st03)
        copy_roi(imgDir, outDir) 
        execute(imgDir, outDir, "as_pth/st05_pth", 0)

    logging.info("RUNS   : "+str(len(run_list)))
    logging.info("ERRORS : "+str(len(err_list)))
    if len(err_list):
        logging.info("LIST OF SESSIONS ENDED WITH ERRORS: ")
        for e in err_list:
            logging.info(e)
    if len(err_list):
        exit(-1)

#-----------------------------------------------------------
    
if __name__ == '__main__':
    main()
