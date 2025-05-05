import os, sys
import pathlib
import glob
import tracemalloc
import timeit
import shutil
import logging
import copy
import time
import json
import multiprocessing as mp
import pydicom
#---------------------------------------------------------
from argparse   import ArgumentParser
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#--------------------------------------------------------
from v_utils.v_json import jsonDumpSafe
#--------------------------------------------------------
raw_dicom_dir = 'as_input_dicom'
input_dir = 'as_input'

#--------------------------------------------------------
run_list        = []
err_list        = []
#--------------------------------------------------------

def log_err(name, userID, sesID):
    err 			= {}
    err['plugin'] 	= name
    err['user'] 	= userID
    err['session'] 	= sesID
        
    err_list.append(err)

def log_run(name, userID, sesID):
    run 			= {}
    run['plugin'] 	= name
    run['user'] 	= userID
    run['session'] 	= sesID
        
    run_list.append(run)
    
#-----------------------------------------------------------------------------------
def getDICOMheader_noexit(inputfile):
    try:        
        dataset = pydicom.dcmread(inputfile)
    except Exception as err:
        return None
    return dataset
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------


def Run_import_data(verbose, userID, destintation_dir):
    
#-----------------------------------------------------------
# construct cmd line
    
    plpath 	= os.path.normpath('as_bin/utils/as_import_data.py ')

    cmd    = 'python '   + plpath
    cmd   +=  '-v '     + verbose
    cmd   += ' -src '    + userID
    cmd   += ' -dst '     + destintation_dir

    log_run ('import dicom data', userID, destintation_dir)

    print(cmd)
    ret = os.system(cmd)

    if ret:
        log_err ('import dicom data', userID, destintation_dir)

    return

#-----------------------------------------------------------
def Run_explode_data(verbose, userID, destintation_dir):
    
#-----------------------------------------------------------
# construct cmd line
    
    plpath 	= os.path.normpath('as_bin/utils/as_import_data_explode_dicom.py ')

    cmd    = 'python '   + plpath
    cmd   +=  '-v '     + verbose
    cmd   += ' -src '    + userID
    cmd   += ' -dst '     + destintation_dir

    log_run ('explode dicom data', userID, destintation_dir)

    print(cmd)
    ret = os.system(cmd)

    if ret:
        log_err ('explode dicom data', userID, destintation_dir)

    return
#---------------------------------------------------------
# main
#---------------------------------------------------------

def main():

    parser          = ArgumentParser()

    parser.add_argument("-usrID",  "--user_id"          , dest="usr_id"      ,help="user id",    nargs='+', metavar="PATH",	required=False)
    parser.add_argument("-v",      "--verbose"          , dest="verbose"     ,help="verbose level",							required=False)

    args            = parser.parse_args()

    #---------------------------------------------------------------------------------------------------------------------------------------

    logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler("dbg_import_dicom_data.log",mode='w'),logging.StreamHandler(sys.stdout)])

    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    #---------------------------------------------------------------------------------------------------------------------------------------

    usr_dir     = raw_dicom_dir + '/'
    usr_dir     = os.path.normpath(usr_dir)
    usr_list    = [os.path.basename(f.path) for f in os.scandir(usr_dir) if f.is_dir()]

    #---------------------------------------------------------------------------------------------------------------------------------------

    verbose         = 'off'   		    if args.verbose         is None else args.verbose

    #---------------------------------------------------------------------------------------------------------------------------------------

    userID = []

    if args.usr_id  is None:
        userID = usr_list
    else:
        for user in usr_list:
           if user in args.usr_id:
               userID.append(user)

    #---------------------------------------------------------------------------------------------------------------------------------------

    logging.info('INFO      > ------------------------------import_dicom_data_newdicom.py-----------------')
    logging.info('INFO      > user list to process:')
    for user in userID:
        logging.info('INFO      >    '+ user)
    logging.info('INFO      > -----------------------------------------------------')

    for user in userID:
        logging.info('INFO      > starting to process '+ user)
        
        Run_explode_data(verbose, usr_dir + '/'+ user,  usr_dir + '/'+ user+'XXX/DICOM')
        
        destination_dirname = []

        if user[0] == 'X':
            destination_dirname = user.split('_',1)[0]
        elif user[0] == 'B':
            destination_dirname = user.split('_',1)[0]
        elif user[0] == 'T':
            destination_dirname = user.split('_',1)[0]
        elif user[0] == 'C':
            destination_dirname = user.split('_',1)[0]
        else:
            logging.error("Unknown user category (B, T, X, C): {}".format(user))
            sys.exit(1)
        
        user_path = os.path.normpath(input_dir + '/' + destination_dirname)
        
        if os.path.isdir(user_path):
            logging.error("Directory exists, while it should not: {}".format(user_path))
            sys.exit(1)

        try:
            if not os.path.isdir(user_path):
                pathlib.Path(user_path).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('Creating "%s" directory failed, error "%s"'%(user_path,err))
            sys.exit(1)

        Run_import_data(verbose, usr_dir + '/'+ user+'XXX', user_path)

    logging.info("RUNS   : "+str(len(run_list)))
    logging.info("ERRORS : "+str(len(err_list)))
    if len(err_list):
        logging.info("LIST OF SESSIONS ENDED WITH ERRORS: ")
        for e in err_list:
            logging.info(e)

#-----------------------------------------------------------
    
if __name__ == '__main__':
    main()
