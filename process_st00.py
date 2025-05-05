import sys, os
import pathlib
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
import glob
import tracemalloc
import multiprocessing
import timeit
import time
import shutil
import logging
import json
#---------------------------------------------------------
from argparse   import ArgumentParser
#-----------------------------------------------------------------------------------------
from v_utils.v_dataset import expand_session_dirs
#--------------------------------------------------------
input_dir       = 'as_input'
stage00_dir     = 'as_data/st00_dicom_processing'
dicom_dir       = 'images'
nrrd_dir        = 'nrrd'
upsampled_dir   = 'upsampled'
upsampled_rectified_dir   = 'upsampled_rectified'
rectified_dir   = 'rectified'
superpixel_dir  = 'superpixel'
superpixel_rectified_dir  = 'superpixel_rectified'
brightness_dir  = 'brightness'
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
    
#---------------------------------------------------------
def Run_DICOMCHECK(verbose, invertDir, forceFlip, ses):

    plpath  = os.path.normpath('as_bin/st00_dicom_processing/as_dicom_check.py ')

    idpath  = os.path.normpath(os.path.join(input_dir   , ses))
    dppath  = os.path.normpath(os.path.join(stage00_dir , ses, dicom_dir))
    
    cmd     = 'python3 ' + plpath 
    cmd   +=  '-v '      + verbose
    cmd   += ' -idDir '  + idpath
    cmd   += ' -ckDir '  + dppath
    cmd   += ' -invD '   + str(invertDir)
    cmd   += ' -fF '   + str(forceFlip)
    


    log_run ('DICOM check', ses)

    logging.info(cmd)
    ret = os.system(cmd)

    if ret:
        log_err ('DICOM check', ses)

    return

#---------------------------------------------------------
def Run_DICOM2PNG(verbose, ses, min_no, max_no, cropcenterx, cropcentery, cropradius, cropx1, cropy1, cropx2, cropy2):
    
    plpath  = os.path.normpath('as_bin/st00_dicom_processing/as_dicom2png.py ')

    idpath  = os.path.normpath(os.path.join(input_dir  , ses))
    dppath  = os.path.normpath(os.path.join(stage00_dir, ses, dicom_dir))
    lspath  = os.path.normpath(os.path.join(stage00_dir, ses, dicom_dir))

    cmd    = 'python3 '   + plpath 
    cmd   +=  '-v '       + verbose
    cmd   += ' -idDir '   + idpath
    cmd   += ' -dpDir '   + dppath
    cmd   += ' -lsDir '   + lspath
    cmd   += ' --min_no ' + str(min_no)
    cmd   += ' --max_no ' + str(max_no)
    if cropcenterx is not None:
        cmd   += ' --cropcenterx ' + str(cropcenterx)
    if cropcentery is not None:
        cmd   += ' --cropcentery ' + str(cropcentery)
    if cropradius is not None:
        cmd   += ' --cropradius ' + str(cropradius)
    if cropx1 is not None:
        cmd   += ' --cropx1 ' + str(cropx1)
    if cropy1 is not None:
        cmd   += ' --cropy1 ' + str(cropy1)
    if cropx2 is not None:
        cmd   += ' --cropx2 ' + str(cropx2)
    if cropy2 is not None:
        cmd   += ' --cropy2 ' + str(cropy2)

    log_run ('DICOM2PNG', ses)

    logging.info(cmd)
    ret = os.system(cmd)

    if ret:
        log_err ('DICOM2PNG', ses)

    return

#---------------------------------------------------------
def Run_DICOM2NRRD(verbose, ses):
    
    plpath  = os.path.normpath('as_bin/st00_dicom_processing/as_dicom2nrrd.py ')

    idpath  = os.path.normpath(os.path.join(input_dir  , ses))
    orpath  = os.path.normpath(os.path.join(stage00_dir, ses, nrrd_dir))
    lspath  = os.path.normpath(os.path.join(stage00_dir, ses, dicom_dir))

    cmd    = 'python3 ' + plpath 
    cmd   +=  '-v '     + verbose
    cmd   += ' -idDir ' + idpath
    cmd   += ' -orDir ' + orpath
    cmd   += ' -lsDir ' + lspath

    log_run ('DICOM2NRRD', ses)

    logging.info(cmd)
    ret = os.system(cmd)

    if ret:
        log_err ('DICOM2nrrd', ses)

    return

#---------------------------------------------------------
def Run_ChangeResolution(verbose, ses):
    
    plpath  = os.path.normpath('as_bin/st00_dicom_processing/as_change_resolution.py ')
    pixspacing = 0.5

    imgpath     = os.path.normpath(os.path.join(stage00_dir, ses, dicom_dir))
    rchgpath    = os.path.normpath(os.path.join(stage00_dir, ses, upsampled_dir))

    cmd    = 'python3 '   + plpath 
    cmd   +=  '-v '       + verbose
    cmd   += ' -dpDir '   + imgpath
    cmd   += ' -dprDir '  + rchgpath
    cmd   += ' -ps {}'.format(pixspacing)

    log_run ('change resolution', ses)

    logging.info(cmd)
    ret = os.system(cmd)

    if ret:
        log_err ('ChangeResolution', ses)

    return

#---------------------------------------------------------
def Run_ChangeBrightness(verbose, ses):
    
    plpath  = os.path.normpath('as_bin/st00_dicom_processing/as_change_brightness.py ')

    imgpath     = os.path.normpath(os.path.join(stage00_dir, ses, upsampled_dir))
    rchgpath    = os.path.normpath(os.path.join(stage00_dir, ses, upsampled_dir))

    cmd    = 'python3 '   + plpath 
    cmd   +=  '-v '       + verbose
    cmd   += ' -imgDir '  + imgpath
    cmd   += ' -outDir '  + rchgpath
 
    log_run ('change brightness', ses)

    logging.info(cmd)
    ret = os.system(cmd)

    if ret:
        log_err ('ChangeBrightness', ses)

    return

#---------------------------------------------------------
# main
#---------------------------------------------------------
def main():

    parser          = ArgumentParser()

    parser.add_argument("-ses",    "--session_id"       , default = "*/*"   ,help="session id", nargs='+',	metavar="PATH",	required=False)

    parser.add_argument("-op",     "--operation"        , dest="op"         ,help="operation",                              required=False)
    parser.add_argument("-v",      "--verbose"          , dest="verbose"    ,help="verbose level",                          required=False)

    args            = parser.parse_args()
    #---------------------------------------------------------------------------------------------------------------------------------------
    # initialize logging 
    data_dir = stage00_dir
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
    

    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    #---------------------------------------------------------------------------------------------------------------------------------------
    
    session_l = expand_session_dirs(args.session_id, input_dir)

    #---------------------------------------------------------------------------------------------------------------------------------------

    verbose         = 'off'             if args.verbose is None else args.verbose
    op              = 'all'             if args.op      is None else args.op

    #---------------------------------------------------------------------------------------------------------------------------------------

    
    logging.info('      > -----------------------------------------------------')
    logging.info('      > sessions list to process:')
    for ses in session_l:
        logging.info('      >    '+ ses)
    logging.info('      > -----------------------------------------------------')
    
    
    
    

    for ses in session_l:
        logging.info('      > starting to process : '+ ses)

        dicomDir    = os.path.normpath(os.path.join(input_dir,   ses))
        outDir 		= os.path.normpath(os.path.join(stage00_dir, ses))
        
        #checking the session specific parameters
        min_no = 0
        max_no = -1
        cropcenterx = None
        cropcentery = None
        cropradius = None
        invertDir = False
        forceFlip = False
        cropx1 = None
        cropy1 = None
        cropx2 = None
        cropy2 = None
        
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
            if ("st00" in configuration):
                #we have specific settings
                if "do_not_process" in configuration["st00"]:
                    if configuration["st00"]["do_not_process"]:
                        logging.info(f"Skipping sequence {ses} - forced by config file")
                        continue
                if "min_no" in configuration["st00"]:
                    min_no = configuration["st00"]["min_no"]
                    logging.info(f"specific settings found: min_no = {min_no}")
                if "max_no" in configuration["st00"]:
                    max_no = configuration["st00"]["max_no"]
                    logging.info(f"specific settings found: max_no = {max_no}")
                if "cropcenterx" in configuration["st00"]:
                    cropcenterx = configuration["st00"]["cropcenterx"]
                    logging.info(f"specific settings found: cropcenterx = {cropcenterx}")
                if "cropcentery" in configuration["st00"]:
                    cropcentery = configuration["st00"]["cropcentery"]
                    logging.info(f"specific settings found: cropcentery = {cropcentery}")
                if "cropradius" in configuration["st00"]:
                    cropradius = configuration["st00"]["cropradius"]
                    logging.info(f"specific settings found: cropradius = {cropradius}")
                if "invertDir" in configuration["st00"]:
                    invertDir = configuration["st00"]["invertDir"]
                    logging.info(f"specific settings found: invertDir = {invertDir}")
                if "forceFlip" in configuration["st00"]:
                    forceFlip = configuration["st00"]["forceFlip"]
                    logging.info(f"specific settings found: forceFlip = {forceFlip}")
                if "cropx1" in configuration["st00"]:
                    cropx1 = configuration["st00"]["cropx1"]
                    logging.info(f"specific settings found: cropx1 = {cropx1}")
                if "cropy1" in configuration["st00"]:
                    cropy1 = configuration["st00"]["cropy1"]
                    logging.info(f"specific settings found: cropy1 = {cropy1}")
                if "cropx2" in configuration["st00"]:
                    cropx2 = configuration["st00"]["cropx2"]
                    logging.info(f"specific settings found: cropx2 = {cropx2}")
                if "cropy2" in configuration["st00"]:
                    cropy2 = configuration["st00"]["cropy2"]
                    logging.info(f"specific settings found: cropy2 = {cropy2}")

        try:
            if (op == 'all') and os.path.isdir(outDir):
                shutil.rmtree(outDir)
            if not (os.path.isdir(outDir)):
                pathlib.Path(outDir).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('      > creating "%s" directory failed, error "%s"'%(outDir, err))
            sys.exit(1)

        logging.info('          > -----------------------------------------------------')
        logging.info('          > working set   : %s'%(ses))
        logging.info('          > current dir   : %s'%os.getcwd())
        logging.info('          > process       : %s'%op)
        logging.info('          > verbose level : %s'%verbose)
        logging.info('          > subdirectory  :')
        logging.info('          >    input dicoms               : '+ dicomDir)
        logging.info('          >    output images and metadata : '+ outDir)

        if op == 'dicom2png':
            Run_DICOM2PNG           (verbose, ses, min_no, max_no, cropcenterx, cropcentery, cropradius, cropx1, cropy1, cropx2, cropy2)
                
        elif op == 'dicomcheck':
            Run_DICOMCHECK          (verbose, invertDir, forceFlip, ses)
                
        elif op == 'change-brightness':
            Run_ChangeBrightness    (verbose, ses)
                
        elif op == 'change-resolution':
            Run_ChangeResolution    (verbose, ses)
                
        elif op == 'all':
            Run_DICOMCHECK          (verbose, invertDir, forceFlip, ses)
            Run_DICOM2PNG           (verbose, ses, min_no, max_no, cropcenterx, cropcentery, cropradius, cropx1, cropy1, cropx2, cropy2)
            Run_ChangeResolution    (verbose, ses)
            Run_ChangeBrightness    (verbose, ses)

                
        else:
            logging.info("ERROR    >  unknown plugin selection with option(%s)"%op)
            exit(1)

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
