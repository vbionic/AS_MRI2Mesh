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
stage05_dir     = 'as_data/st05_evaluated_shapes'
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
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        pathlib.Path(dst).mkdir(mode=0o775, parents=True, exist_ok=True)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=symlinks, ignore=ignore)




#-----------------------------------------------------------
def Run_correct_position(verbose, ses, move_step, PerformCorrections, decim):
    # plpath  = os.path.normpath('as_bin/st06_correct_shapes/as_correctPosition2.py')

    # idpath         = os.path.normpath(os.path.join(stage05_dir, ses, "roi"))
    # idshapepath    = os.path.normpath(os.path.join(stage06_dir, ses, "skin_closed"))
    # odpath         = os.path.normpath(os.path.join(stage06_dir, ses))
    # description_fn = os.path.normpath(os.path.join(stage03_dir, ses, 'description.json'))
    
    # imgpath        = os.path.normpath(os.path.join(stage03_dir, ses))
    
    # try:
        # description_h = open(description_fn)
        # description = json.load(description_h)
        # description_h.close()
    # except Exception as err:
        # logging.error('description file not found: "%s", error "%s"'%(description_fn,err))
        # exit(1)
        
        
    # try:
        # if not os.path.isdir(odpath):
            # pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    # except Exception as err:
        # logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        # exit(1)
    
    # logging.info('path to script: {}'.format(plpath))

    # log_run ('CorrectPosition - ROI based', ses)

    # cmd    = 'python3 '     + plpath 
    # cmd   += ' -v '     + verbose
    # cmd   += ' -q '     + 'on'
    # cmd   += ' -oDir '     + odpath
    # cmd   += ' -iDir '     + idpath
    # cmd   += ' -ishDir '     + idshapepath
    # cmd   += ' -imgDir '     + imgpath
    # cmd   += ' -ss '     + "{}".format(description["distance_between_slices"])
    # cmd   += ' -psx '     + "{}".format(description["pixel_spacing_x"])
    # cmd   += ' -psy '     + "{}".format(description["pixel_spacing_y"])
    # cmd   += ' --perform_cor ' + PerformCorrections

    # logging.info(cmd)
    # ret = os.system(cmd)
    
    # if ret:
        # log_err ('CorrectPosition - ROI based', ses)
    plpath  = os.path.normpath('as_bin/st06_correct_shapes/as_correctPosition2.py')

    idpath         = os.path.normpath(os.path.join(stage05_dir, ses, "roi"))
    idshapepath    = os.path.normpath(os.path.join(stage06_dir, ses, "skin_closed"))
    odpath         = os.path.normpath(os.path.join(stage06_dir, ses))
    description_fn = os.path.normpath(os.path.join(stage03_dir, ses, 'description.json'))
    
    imgpath        = os.path.normpath(os.path.join(stage03_dir, ses))
    
    try:
        description_h = open(description_fn)
        description = json.load(description_h)
        description_h.close()
    except Exception as err:
        logging.error('description file not found: "%s", error "%s"'%(description_fn,err))
        exit(1)
        
        
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)
    logging.info('path to script: {}'.format(plpath))

    log_run ('CorrectPosition - NORM 4', ses)

    cmd    = 'python3 '     + plpath 
    cmd   += ' -v '     + verbose
    cmd   += ' -q '     + 'on'
    cmd   += ' -oDir '     + odpath
    cmd   += ' -iDir '     + idpath
    cmd   += ' -ishDir '     + idshapepath
    cmd   += ' -imgDir '     + imgpath
    cmd   += ' -ss '     + "{}".format(description["distance_between_slices"])
    cmd   += ' -psx '     + "{}".format(description["pixel_spacing_x"])
    cmd   += ' -psy '     + "{}".format(description["pixel_spacing_y"])
    cmd   += ' -N 4'
    cmd   += ' -s ' + move_step
    cmd   += ' --perform_cor ' + PerformCorrections
    cmd   += ' --decim ' + decim

    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('CorrectPosition - NORM 4', ses)
   
    return

def Run_correct_position2(verbose, ses, move_step, PerformCorrections, decim):
    plpath  = os.path.normpath('as_bin/st06_correct_shapes/as_correctPosition2.py')

    idpath         = os.path.normpath(os.path.join(stage05_dir, ses, "roi"))
    idshapepath    = os.path.normpath(os.path.join(stage06_dir, ses, "skin_closed"))
    odpath         = os.path.normpath(os.path.join(stage06_dir, ses, "NORM2"))
    description_fn = os.path.normpath(os.path.join(stage03_dir, ses, 'description.json'))
    
    imgpath        = os.path.normpath(os.path.join(stage03_dir, ses))
    
    try:
        description_h = open(description_fn)
        description = json.load(description_h)
        description_h.close()
    except Exception as err:
        logging.error('description file not found: "%s", error "%s"'%(description_fn,err))
        exit(1)
        
        
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)
    
    logging.info('path to script: {}'.format(plpath))

    log_run ('CorrectPosition - NORM 2', ses)

    cmd    = 'python3 '     + plpath 
    cmd   += ' -v '     + verbose
    cmd   += ' -q '     + 'on'
    cmd   += ' -oDir '     + odpath
    cmd   += ' -iDir '     + idpath
    cmd   += ' -ishDir '     + idshapepath
    cmd   += ' -imgDir '     + imgpath
    cmd   += ' -ss '     + "{}".format(description["distance_between_slices"])
    cmd   += ' -psx '     + "{}".format(description["pixel_spacing_x"])
    cmd   += ' -psy '     + "{}".format(description["pixel_spacing_y"])
    cmd   += ' -N 2'
    cmd   += ' -s ' + move_step
    cmd   += ' --perform_cor ' + PerformCorrections
    cmd   += ' --decim ' + decim

    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('CorrectPosition - NORM 2', ses)
   
    return

def Run_correct_position3(verbose, ses, move_step, PerformCorrections, decim):
    plpath  = os.path.normpath('as_bin/st06_correct_shapes/as_correctPosition2.py')

    idpath         = os.path.normpath(os.path.join(stage05_dir, ses, "roi"))
    idshapepath    = os.path.normpath(os.path.join(stage06_dir, ses, "skin_closed"))
    odpath         = os.path.normpath(os.path.join(stage06_dir, ses, "NORM3"))
    description_fn = os.path.normpath(os.path.join(stage03_dir, ses, 'description.json'))
    
    imgpath        = os.path.normpath(os.path.join(stage03_dir, ses))
    
    try:
        description_h = open(description_fn)
        description = json.load(description_h)
        description_h.close()
    except Exception as err:
        logging.error('description file not found: "%s", error "%s"'%(description_fn,err))
        exit(1)
        
        
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)
    logging.info('path to script: {}'.format(plpath))

    log_run ('CorrectPosition - NORM 3', ses)

    cmd    = 'python3 '     + plpath 
    cmd   += ' -v '     + verbose
    cmd   += ' -q '     + 'on'
    cmd   += ' -oDir '     + odpath
    cmd   += ' -iDir '     + idpath
    cmd   += ' -ishDir '     + idshapepath
    cmd   += ' -imgDir '     + imgpath
    cmd   += ' -ss '     + "{}".format(description["distance_between_slices"])
    cmd   += ' -psx '     + "{}".format(description["pixel_spacing_x"])
    cmd   += ' -psy '     + "{}".format(description["pixel_spacing_y"])
    cmd   += ' -N 3'
    cmd   += ' -s ' + move_step
    cmd   += ' --perform_cor ' + PerformCorrections
    cmd   += ' --decim ' + decim

    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('CorrectPosition - NORM 3', ses)
   
    return

def Run_correct_position4(verbose, ses, move_step, PerformCorrections, decim):
    plpath  = os.path.normpath('as_bin/st06_correct_shapes/as_correctPosition2.py')

    idpath         = os.path.normpath(os.path.join(stage05_dir, ses, "roi"))
    idshapepath    = os.path.normpath(os.path.join(stage06_dir, ses, "skin_closed"))
    odpath         = os.path.normpath(os.path.join(stage06_dir, ses, "NORM4"))
    description_fn = os.path.normpath(os.path.join(stage03_dir, ses, 'description.json'))
    
    imgpath        = os.path.normpath(os.path.join(stage03_dir, ses))
    
    try:
        description_h = open(description_fn)
        description = json.load(description_h)
        description_h.close()
    except Exception as err:
        logging.error('description file not found: "%s", error "%s"'%(description_fn,err))
        exit(1)
        
        
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)
    logging.info('path to script: {}'.format(plpath))

    log_run ('CorrectPosition - NORM 4', ses)

    cmd    = 'python3 '     + plpath 
    cmd   += ' -v '     + verbose
    cmd   += ' -q '     + 'on'
    cmd   += ' -oDir '     + odpath
    cmd   += ' -iDir '     + idpath
    cmd   += ' -ishDir '     + idshapepath
    cmd   += ' -imgDir '     + imgpath
    cmd   += ' -ss '     + "{}".format(description["distance_between_slices"])
    cmd   += ' -psx '     + "{}".format(description["pixel_spacing_x"])
    cmd   += ' -psy '     + "{}".format(description["pixel_spacing_y"])
    cmd   += ' -N 4'
    cmd   += ' -s ' + move_step
    cmd   += ' --perform_cor ' + PerformCorrections
    cmd   += ' --decim ' + decim

    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('CorrectPosition - NORM 4', ses)
   
    return

def Run_correct_position5(verbose, ses, move_step, PerformCorrections, decim):
    plpath  = os.path.normpath('as_bin/st06_correct_shapes/as_correctPosition2.py')

    idpath         = os.path.normpath(os.path.join(stage05_dir, ses, "roi"))
    idshapepath    = os.path.normpath(os.path.join(stage06_dir, ses, "skin_closed"))
    odpath         = os.path.normpath(os.path.join(stage06_dir, ses, "NORM5"))
    description_fn = os.path.normpath(os.path.join(stage03_dir, ses, 'description.json'))
    
    imgpath        = os.path.normpath(os.path.join(stage03_dir, ses))
    
    try:
        description_h = open(description_fn)
        description = json.load(description_h)
        description_h.close()
    except Exception as err:
        logging.error('description file not found: "%s", error "%s"'%(description_fn,err))
        exit(1)
        
        
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)
    logging.info('path to script: {}'.format(plpath))

    log_run ('CorrectPosition - NORM 4', ses)

    cmd    = 'python3 '     + plpath 
    cmd   += ' -v '     + verbose
    cmd   += ' -q '     + 'on'
    cmd   += ' -oDir '     + odpath
    cmd   += ' -iDir '     + idpath
    cmd   += ' -ishDir '     + idshapepath
    cmd   += ' -imgDir '     + imgpath
    cmd   += ' -ss '     + "{}".format(description["distance_between_slices"])
    cmd   += ' -psx '     + "{}".format(description["pixel_spacing_x"])
    cmd   += ' -psy '     + "{}".format(description["pixel_spacing_y"])
    cmd   += ' -N 5'
    cmd   += ' -s ' + move_step
    cmd   += ' --perform_cor ' + PerformCorrections
    cmd   += ' --decim ' + decim

    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('CorrectPosition - NORM 5', ses)
   
    return

def Run_correct_position6(verbose, ses, move_step, PerformCorrections, decim):
    plpath  = os.path.normpath('as_bin/st06_correct_shapes/as_correctPosition2.py')

    idpath         = os.path.normpath(os.path.join(stage05_dir, ses, "roi"))
    idshapepath    = os.path.normpath(os.path.join(stage06_dir, ses, "skin_closed"))
    odpath         = os.path.normpath(os.path.join(stage06_dir, ses, "NORM6"))
    description_fn = os.path.normpath(os.path.join(stage03_dir, ses, 'description.json'))
    
    imgpath        = os.path.normpath(os.path.join(stage03_dir, ses))
    
    try:
        description_h = open(description_fn)
        description = json.load(description_h)
        description_h.close()
    except Exception as err:
        logging.error('description file not found: "%s", error "%s"'%(description_fn,err))
        exit(1)
        
        
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)
    logging.info('path to script: {}'.format(plpath))

    log_run ('CorrectPosition - NORM 4', ses)

    cmd    = 'python3 '     + plpath 
    cmd   += ' -v '     + verbose
    cmd   += ' -q '     + 'on'
    cmd   += ' -oDir '     + odpath
    cmd   += ' -iDir '     + idpath
    cmd   += ' -ishDir '     + idshapepath
    cmd   += ' -imgDir '     + imgpath
    cmd   += ' -ss '     + "{}".format(description["distance_between_slices"])
    cmd   += ' -psx '     + "{}".format(description["pixel_spacing_x"])
    cmd   += ' -psy '     + "{}".format(description["pixel_spacing_y"])
    cmd   += ' -N 6'
    cmd   += ' -s ' + move_step
    cmd   += ' --perform_cor ' + PerformCorrections
    cmd   += ' --decim ' + decim

    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('CorrectPosition - NORM 6', ses)
   
    return

    
def Run_copytree(verbose, ses):
    idpath         = os.path.normpath(os.path.join(stage05_dir, ses))
    odpath         = os.path.normpath(os.path.join(stage06_dir, ses))
    imgpath        = os.path.normpath(os.path.join(stage03_dir, ses))
        
    try:
        if not os.path.isdir(odpath):
            pathlib.Path(odpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(odpath,err))
        exit(1)

    
    logging.info("===============copying files===================")
    copytree(idpath + "/bones", odpath + "/bones")
    copytree(idpath + "/fat", odpath + "/fat")
    copytree(idpath + "/muscles", odpath + "/muscles")
    copytree(idpath + "/vessels", odpath + "/vessels")
    copytree(idpath + "/roi", odpath + "/roi")
    copytree(imgpath, odpath + "/images")
    copytree(odpath + "/skin_closed", odpath + "/skin")
    
    
    
    
    
#-----------------------------------------------------------
def Run_correct_skin(verbose, ses):

    plpath  = os.path.normpath('as_bin/st06_correct_shapes/as_correctShapes2.py ')

    idpath  = os.path.normpath(os.path.join(stage05_dir, ses, "skin"))
    odpath  = os.path.normpath(os.path.join(stage06_dir, ses, "skin_closed"))
    
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

    logging.info(cmd)
    ret = os.system(cmd)
    
    if ret:
        log_err ('CorrectShapes - Skin', ses)
   
    return
    
#---------------------------------------------------------
# main
#---------------------------------------------------------
def main():

    parser          = ArgumentParser()

    parser.add_argument("-ses",    "--session_id"       , default = "*/*"   ,help="session id", nargs='+',	metavar="PATH",	required=False)
                                                                                                                                   
    parser.add_argument("-op",     "--operation"        , dest="op"      ,help="operation",                                        required=False)
    parser.add_argument("-v",      "--verbose"          , dest="verbose" ,help="verbose level <off, on/dbg>",                      required=False)
    parser.add_argument("-s",    "--move_step"      ,     dest="move_step",    help="movement made in each step"   ,                    required=False)
    parser.add_argument("-c"     ,    "--perform_cor"    ,     dest="PerformCorrections",    help="perform correctins on the set"   ,   required=False)
    parser.add_argument("-d"     ,    "--decim"    ,     dest="decim",    help="decimation step"   ,   required=False)


    args            = parser.parse_args()
    #---------------------------------------------------------------------------------------------------------------------------------------
    # initialize logging 
    data_dir = stage06_dir
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
    op_l = ['all', 'skin', 'position', 'postprocess', 'positionM2', 'positionM3', 'positionM4', 'positionM5', 'positionM6']
    
    verbose         = 'off'             if args.verbose   is None else args.verbose
    op              = 'all'             if args.op        is None else args.op
    move_step       = '1.0'             if args.move_step is None else args.move_step
    decim       = '1'             if args.decim is None else args.decim
    PerformCorrections = 'True'         if args.PerformCorrections is None else args.PerformCorrections
    
    if not (op in op_l):
        logging.error("operation filter {} does not match any of expected vals: {}".format(op, op_l))
        exit(1)
        
    #---------------------------------------------------------------------------------------------------------------------------------------

    session_l = expand_session_dirs(args.session_id, 'as_data/st05_evaluated_shapes/')

    #---------------------------------------------------------------------------------------------------------------------------------------

    logging.info('      > -----------------------------------------------------')
    logging.info('      > sessions list to process:')
    for ses in session_l:
        logging.info('      >    '+ ses)
    logging.info('      > -----------------------------------------------------')

    for ses in session_l:
    
        do_correct_position = True
    
        #checking the session specific parameters
        try:
            conffn = ses.replace('/','_').replace('\\','_') + ".cfg"
            conffh = open(os.path.join("as_cfg",conffn))
            configuration = json.load(conffh)
            conffh.close()
        except Exception as err:
            logging.info("Configuration file IO error: {}".format(err))
            logging.info("Proceeding with default settings")
            configuration = None
        if (configuration is not None):
            if ("st06" in configuration):
                #we have specific settings
                if "correct_position" in configuration["st06"]:
                    do_correct_position = configuration["st06"]["correct_position"]
                    logging.info(f"Will skip position correction: {not do_correct_position}")
                    
        logging.info('      > starting to process : '+ ses)

        logging.info('=' * 50)
        logging.info(' working set   : %s'%(ses))
        logging.info(' current dir   : %s'%os.getcwd())
        logging.info(' process       : %s'%op)
        logging.info(' verbose level : %s'%verbose)
        logging.info('-' * 50)

        if op == 'lsc':
            pass
            
        elif op == 'position':
            Run_correct_position    (verbose, ses, move_step, PerformCorrections, decim)
            
        elif op == 'positionM2':
            Run_correct_position2   (verbose, ses, move_step, PerformCorrections, decim)
            
        elif op == 'positionM3':
            Run_correct_position3   (verbose, ses, move_step, PerformCorrections, decim)
        
        elif op == 'positionM4':
            Run_correct_position4   (verbose, ses, move_step, PerformCorrections, decim)
            
        elif op == 'positionM5':
            Run_correct_position5   (verbose, ses, move_step, PerformCorrections, decim)
            
        elif op == 'positionM6':
            Run_correct_position6   (verbose, ses, move_step, PerformCorrections, decim)
            
        elif op == 'skin':
            Run_correct_skin        (verbose, ses)
            
        elif op == 'all':
            Run_correct_skin        (verbose, ses)
            if do_correct_position:
                Run_correct_position    (verbose, ses, move_step, PerformCorrections, decim)
            else:
                Run_copytree(verbose, ses)
            

         
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
