import sys, os
from multiprocessing import Process, Queue
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
import os
import pathlib
import glob
import logging
import copy
import time
import json
#---------------------------------------------------------
from argparse   import ArgumentParser
#-----------------------------------------------------------------------------------------
from v_utils.v_dataset import expand_session_dirs
from v_utils.v_arg import arg2boolAct
#--------------------------------------------------------
stage23_dir     = 'as_data/st23_preprocessed_meshes/'
stage24_dir     = 'as_data/st24_remeshed'
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

def execute_remesh(q, ses, inDir, outDir, pitch, fn_ptrn, args_rem_line):
 
    inDirSes  = os.path.normpath(os.path.join(inDir,  ses))
    outDirSes = os.path.normpath(os.path.join(outDir, ses))
    
    cmdline   = f'python ./as_bin/st24_remesh/remesh.py'
    cmdline  += f' --in_dir {inDirSes}'
    cmdline  += f' --out_dir {outDirSes} '
    cmdline  += f' --pitch {pitch} '
    cmdline  += f' --in_pattern {fn_ptrn} '
    cmdline  += f" {args_rem_line}"

    logging.info(cmdline)

    log = log_run ('REMESH', (ses, inDir, outDir, pitch, fn_ptrn))

    #cmdline = f"echo {ses}, {inDir}, {outDir}, {pitch}, {fn_ptrn}"
    ret = os.system(cmdline)

    if ret:
        err = log_err ('REMESH', (ses, inDir, outDir, pitch, fn_ptrn))
    else:
        err = None
        
    q.put((log, err))

#---------------------------------------------------------
# main
#---------------------------------------------------------

def main():

    parser          = ArgumentParser()

    parser.add_argument("-ses", "--session_id"             , default = "*/*"   ,help="session id", nargs='+',  metavar="PATH", required=False)
    parser.add_argument("-op",  "--operation"              , default = "all"   ,help="operation",                              required=False)
    parser.add_argument("-th",  "--threads"  , type = int  , default = -2      ,help="Number of simultaneous processes",       required=False)
                                                           
    parser.add_argument("-v",   "--verbose"                                    ,help="verbose level",                          required=False)
    parser.add_argument("-p",   "--pitch"             , type = float, default = 1.0 , help="Voxel overall dimension",          required=False)
    
    args, args_rem = parser.parse_known_args()
    args_rem_line = ' '.join(args_rem)
    #---------------------------------------------------------------------------------------------------------------------------------------
    
    pitch       = args.pitch
    #---------------------------------------------------------------------------------------------------------------------------------------
    inDir  		= os.path.normpath(stage23_dir)
    outDir 		= os.path.normpath(stage24_dir)
    logDir 		= os.path.normpath(os.path.join(outDir,"_log"))
    
    try:
        if not os.path.isdir(logDir):
            pathlib.Path(logDir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(logDir, err))
        exit(1)
    #---------------------------------------------------------------------------------------------------------------------------------------
    # initialize logging 
    script_name = os.path.basename(__file__).split(".")[0]
    from datetime import datetime
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    initial_log_fn = f"{logDir}/_dbg_{script_name}_{time_str}_pid{os.getpid()}.log"
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
    
    op_l = ['all', 'remesh', 'roi_remesh']
    op   = args.operation
    
    if not (op in op_l):
        logging.error("operation filter {} does not match any of expected vals: {}".format(op, op_l))
        exit(1)
     
    threads = args.threads 
    if args.threads <= 0:
        threads = max(1, (os.cpu_count() - abs(args.threads)))
    #---------------------------------------------------------------------------------------------------------------------------------------

    session_l = expand_session_dirs(args.session_id, stage23_dir)

    #fn_ptrns = [('*_skin_bones_vessels_mesh_volume.obj', 'volume'),
    #            ('*_skin_mesh_roi.stl',                  'volume'),
    #            ('*_skin_mesh_outer.stl',                'outer' )  ]
    if op == 'roi_remesh':
        fn_ptrns = [('*_skin_mesh_roi.stl',       'volume')  ]
    else:
        fn_ptrns = [('*_skin_mesh_volume.stl',    'volume'),
                    ('*_bones_mesh_volume.stl',   'volume'),
                    ('*_vessels_mesh_volume.stl', 'volume'),
                    ('*_skin_mesh_roi.stl',       'volume'),
                    ('*_skin_mesh_outer.stl',     'outer' )  ]
            
    th_to_start_num = len(session_l)*len(fn_ptrns)
    
    logging.info(f'      > -----------------------------------------------------')
    logging.info(f'      > threads {threads} / {os.cpu_count()}')
    logging.info(f'      > total tasks {th_to_start_num}')
    logging.info(f'      > sessions list to process ({len(session_l)}):')
    for ses in session_l:
        logging.info(f'      >    {ses}')
    logging.info(f'      > files patterns ({len(fn_ptrns)}):')
    for fn_ptrn in fn_ptrns:
        logging.info(f'      >    {fn_ptrn}')
    logging.info(f'      > -----------------------------------------------------')

    #---------------------------------------------------------------------------------------------------------------------------------------


    th_pending_num = 0
    th_done_num    = 0 
    th_started_num = 0
    th_list        = []
    
    
    while (th_done_num < th_to_start_num):
    
        changed = False
        
        # check if started processes have finished 
        for pid, (p, q, ses, ptrn) in enumerate(th_list):
        
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
                
                logging.info(f'          > done  set   : {ses}, {ptrn}')
                break
        
        # start a new process
        if (th_started_num < th_to_start_num) and (th_pending_num < threads):
            session_l_id = th_started_num  % len(session_l)
            fn_ptrn_l_id = th_started_num // len(session_l)
            ses     = session_l[session_l_id]
            fn_ptrn,oDirApx = fn_ptrns [fn_ptrn_l_id]

            outDirS 		= os.path.normpath(os.path.join(outDir, oDirApx))
            logDirS 		= os.path.normpath(os.path.join(outDirS,"_log"))
            
            try:
                if not os.path.isdir(logDirS):
                    pathlib.Path(logDirS).mkdir(mode=0o775, parents=True, exist_ok=True)
            except Exception as err:
                logging.error('creating "%s" directory failed, error "%s"'%(logDirS, err))
                exit(1)
                
            logging.info(f'          > -----------------------------------------------------')
            logging.info(f'          > start set {th_started_num+1}/{th_to_start_num}  : {ses}, {fn_ptrn}, pitch {pitch}')
                
            queue = Queue()
            if op == 'remesh' or op == 'roi_remesh' :              
                p = Process(target=execute_remesh, args=(queue, ses, inDir, outDirS, pitch, fn_ptrn, args_rem_line), daemon = True)
            elif op == 'all':                                                              
                p = Process(target=execute_remesh, args=(queue, ses, inDir, outDirS, pitch, fn_ptrn, args_rem_line), daemon = True)
            
            p.start()
            th_list.append((p, queue, ses, fn_ptrn))
            
            th_started_num += 1
            th_pending_num += 1
            changed = True
            
        if not changed:
            time.sleep(1)

    logging.info(f"RUNS   : {len(run_list)}")
    logging.info(f"ERRORS : {len(err_list)}")
    if len(err_list):
        logging.info("LIST OF SETS ENDED WITH ERRORS: ")
        for task_descr in err_list:
            logging.info(f'{task_descr}')

#-----------------------------------------------------------
    
if __name__ == '__main__':
    main()
