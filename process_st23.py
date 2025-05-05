import sys, os
from multiprocessing import Process, Queue
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
#-----------------------------------------------------------------------------------------
from v_utils.v_dataset import expand_session_dirs
#---------------------------------------------------------
from argparse   import ArgumentParser
#--------------------------------------------------------
stage03_dir        = "as_data/st03_preprocessed_images"
ax_shapes_dir_cor  = "as_data/st08_filling_voids"
ax_shapes_dir      = "as_data/st05_evaluated_shapes"
stage23_dir        = "as_data/st23_preprocessed_meshes"
#--------------------------------------------------------
run_list        = []
err_list        = []
#--------------------------------------------------------

def log_err(name, ses):
    err             = {}
    err['plugin']   = name
    err['session']  = ses
        
    return err

def log_run(name, ses):
    run             = {}
    run['plugin']   = name
    run['session']  = ses
        
    return run

#---------------------------------------------------------
def Run_CreateMesh(q, verbose, skinDir, iDir, ses, name_sfx, skin_outer_only, detect_lines = False, do_plugs = True):
    
    plpath  = os.path.normpath('as_bin/st23_mesh_processing/as_gen_mesh_v5.py ')

    imgpath     = os.path.normpath(iDir        + '/' + ses)
    skinpath    = os.path.normpath(skinDir     + '/' + ses)
    meshpath    = os.path.normpath(stage23_dir + '/' + ses)
    dpath       = os.path.normpath(stage03_dir + '/' + ses)
    
    try:
        if not os.path.isdir(meshpath):
            pathlib.Path(meshpath).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('      > creating "%s" directory failed, error "%s"'%(meshpath, err))
        sys.exit(1)

    cmd    = 'python '   + plpath 
    cmd   += ' -v '      + verbose
    cmd   += ' -iDir '  + imgpath
    cmd   += ' -sDir '  + skinpath
    cmd   += ' -oDir '  + meshpath
    cmd   += ' -dDir '  + dpath
    cmd   += ' -nSfx '  + name_sfx
    cmd   += ' --obj_smooth F '
    if skin_outer_only:
        cmd   += ' --ignore_holes T '
        cmd   += ' --ds_polygon_clss skin '
    else:
        cmd   += ' --ignore_holes F '
        cmd   += ' --ds_polygon_clss skin bones vessels '
    if do_plugs:
        cmd   += ' --do_plugs T ' 
    else:
        cmd   += ' --do_plugs F ' 
    
    if detect_lines:
        cmd   += ' --detect_lines T '
 
    log = log_run (f'CreateMesh profile_name = {name_sfx}, skin_outer_only = {skin_outer_only}, do_plugs = {do_plugs}', ses)

    logging.info(cmd)
    ret = os.system(cmd)

    if ret:
        err = log_err (f'CreateMesh profile_name = {name_sfx}, skin_outer_only = {skin_outer_only}, do_plugs = {do_plugs}', ses)
    else:
        err = None

    q.put((log, err))

#---------------------------------------------------------
# main
#---------------------------------------------------------
def main():

    parser          = ArgumentParser()

    parser.add_argument("-ses",    "--session_id"       , default = "*/*"   ,help="session id", nargs='+',	metavar="PATH",	required=False)

    parser.add_argument("-op",     "--operation"        , default = "all"   ,help="operation",                              required=False)
    parser.add_argument("-v",      "--verbose"          , dest="verbose"    ,help="verbose level",                          required=False)
    parser.add_argument("-th",  "--threads"  , type = int  , default = -2      ,help="Number of simultaneous processes",       required=False)
    
    parser.add_argument("-cor",    "--corrected"  	    , dest="cor"     	,help="corrected", action='store_true', default=True, required=False)

    args            = parser.parse_args()

    #---------------------------------------------------------------------------------------------------------------------------------------
    
    verbose         = 'off'             if args.verbose is None else args.verbose
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    
    op_l = ['all', 'outer', 'volume', 'roi']
    op   = args.operation
    
    if not (op in op_l):
        logging.error("operation filter {} does not match any of expected vals: {}".format(op, op_l))
        exit(1)

    #---------------------------------------------------------------------------------------------------------------------------------------
    # initialize logging 
    data_dir = stage23_dir
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

    if args.cor:
        in_shapes_dir = ax_shapes_dir_cor
    else:
        in_shapes_dir = ax_shapes_dir

    threads = args.threads 
    if args.threads <= 0:
        threads = max(1, (os.cpu_count() - abs(args.threads)))
    #---------------------------------------------------------------------------------------------------------------------------------------
     
    session_l = expand_session_dirs(args.session_id, ax_shapes_dir)
    session_la = []
    if op == 'all':
        for o in ['outer', 'volume', 'roi']:  
            session_la.extend([(l,o) for l in session_l]) 
    else:
        session_la.extend([(l,op) for l in session_l]) 
    session_l = session_la
    
    logging.info(f'      > -----------------------------------------------------')
    logging.info(f'      > threads {threads} / {os.cpu_count()}')
    logging.info(f'      > sessions list to process:')
    for ses in session_l:
        logging.info(f'      >    {ses}')
    logging.info(f'      > -----------------------------------------------------')

    #---------------------------------------------------------------------------------------------------------------------------------------


    th_pending_num = 0
    th_done_num    = 0 
    session_l_id   = 0
    th_list        = []
    
    while (th_done_num < len(session_l)):
    
        changed = False
        
        # check if started processes have finished 
        for pid, (p, q, ses) in enumerate(th_list):
        
            p.join(timeout=0)
            if not p.is_alive():
                
                (log, err) = q.get()
                q.close()
                run_list.append(log)
                if not err is None:
                    err_list.append(err)
                
                th_list.pop(pid)
                
                changed = True
                th_pending_num -= 1
                th_done_num    += 1
                
                logging.info('          > done  set   : %s'%(ses))
                break
        
        # start a new process
        if (session_l_id < len(session_l)) and (th_pending_num < threads):
            ses, op = session_l[session_l_id]
            logging.info('          > -----------------------------------------------------')
            logging.info('          > start set     : %s'%(ses))
            logging.info('          > current dir   : %s'%os.getcwd())
            logging.info('          > verbose level : %s'%args.verbose)
            logging.info('          > use cor. input: %s'%args.cor)
        
            queue = Queue()
            if op == 'outer':#                                   q, verbose,       skinDir,           iDir, ses, name_sfx, skin_outer_only, detect_lines = False, do_plugs = True                      
                p = Process(target=Run_CreateMesh, args=    (queue, verbose, in_shapes_dir,  in_shapes_dir, ses, "outer" ,           True ,                False,           False ))
            if op == 'volume':                                                                                                                                              
                p = Process(target=Run_CreateMesh, args=    (queue, verbose, in_shapes_dir,  in_shapes_dir, ses, "volume",           False,                False,           True ))
            if op == 'roi':                                                                                                                                                 
                p = Process(target=Run_CreateMesh, args=    (queue, verbose, in_shapes_dir,  in_shapes_dir, ses, "roi"   ,           True ,                False,           True ))

            p.start()
            th_list.append((p, queue, ses))
            
            session_l_id   += 1
            th_pending_num += 1
            changed = True
            
        if not changed:
            time.sleep(3)

    logging.info("RUNS   : "+str(len(run_list)))
    logging.info("ERRORS : "+str(len(err_list)))
    if len(err_list):
        logging.info("LIST OF SESSIONS ENDED WITH ERRORS: ")
        for e in err_list:
            logging.info(e)

#-----------------------------------------------------------
    
if __name__ == '__main__':
    main()
