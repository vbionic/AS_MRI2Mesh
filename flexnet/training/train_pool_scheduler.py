import os, sys
import glob
import json
import logging
import time
import queue
import re
import json
import shutil
import torch
import numpy as np
#---------------------------------------------------------
from datetime           import datetime
from multiprocessing    import Process, Queue
from argparse           import ArgumentParser
#--------------------------------------------------------

def end_process(process):
    process.join()
    logging.info(" Joined process {} with exit code = {}...".format(process.name, process.exitcode))
    if(process.exitcode != 0):
        logging.error(' Child process returned with error {} (process = {})'.format(process.exitcode, process))

#-----------------------------------------------------------

def execute(task_box, id):
 
    args        = ""
    cfgs        = " --cfg"

    fname       = task_box["file_name"]
    out_dir     = task_box["sch_cfg"]["pool"]
    oname       = fname.split("#")[1][1:]

    for p in task_box["task"]:
        switch  = list(p.keys())[0]
        arg     = p[switch]

        if switch[0] != '#':
            if switch=="--cfg":
                cfgs    = cfgs + " " + arg
            else:
                switch = switch.replace("++","--")    
                print(switch,arg)
                args    += ' ' + switch + ' ' + str(arg)

    args += cfgs 

    cmdline     = 'python flexnet/training/pytorch_model_train.py '
    cmdline    += '--out_dir {} '.format(os.path.normpath(os.path.join(out_dir, oname)))
    cmdline    += args + ' ' 
    cmdline    += '--force_gpu_id ' + str(id) + ' '

    print(cmdline)
    
    ret = 1#os.system(cmdline)
    
    return ret

#-----------------------------------------------------------

def worker(qin,qout,id):

    logging.info("worker %d started"%(id))

    os.chdir("as")
    print(os.getcwd())
    
    while True:

        if qin.empty():
            time.sleep(0.1)
        else:
            cmd = qin.get()
            #logging.info("cmd:{}".format(cmd[0]))
        
            if cmd[0] == 'quit':
                qout.put(["quit",None])
                break
            if cmd[0] == 'ready':
                qout.put(["ready",None])
            if cmd[0] == 'execute':
                #logging.info("arg:{}".format(cmd[1]))
                ret  = execute(cmd[1],id)
                qout.put(["executed",ret,cmd[1],id])
            else:
                time.sleep(0.1)
        
    return
    
#-----------------------------------------------------------
# load queues
#-----------------------------------------------------------

def q_load_from_waiting_dir(root_dir, sch_cfg):
    
        q_list      = []
        q_dir       = root_dir + "/" + sch_cfg["scheduler_dir"] + "/" + sch_cfg["waiting_dir"] + "/q_*.json"
        qfile_list  = glob.glob(q_dir)

        for qname in qfile_list:

            tlist = {}
            
            qfile	= open(qname,"r")
            qjson   = json.load(qfile)
            
            xname           = os.path.basename(qname)
            fname, fext     = os.path.splitext(xname)

            params = {}
            for p in fname.split("#")[0].split("_"):
                tmp  = p.split("[")
                if len(tmp)>1:
                    arg  = tmp[1][:-1]
                    name = tmp[0]
                    params[name] = arg
            
            tlist["params"]         =  params
            tlist["sch_cfg"]        =  sch_cfg
            tlist["root_dir"]       =  root_dir
            tlist["file_name"]      =  os.path.basename(qname)
            tlist["task"]           =  qjson["task"]
            q_list.append(tlist)
            qfile.close()
        
            # print()
            # print("params",tlist["params"])
            # print("root_dir",tlist["root_dir"])
            # print("file_dir",tlist["file_dir"])
            # print("file_name",tlist["file_name"])
        return(q_list)

#-----------------------------------------------------------
# make task list from all qeueues 
#-----------------------------------------------------------

def q_get_task_list(work_dirs,dir_cfg):

    q_list = []
    
    for i in range(len(work_dirs)):
        q_list = q_load_from_waiting_dir(work_dirs[i],dir_cfg)

    return(q_list)

#-----------------------------------------------------------
# get one task from queues
#-----------------------------------------------------------

def q_get_one_task(q_list):

    task = None
    if (len(q_list)) > 0:
        task    = q_list.pop(0)
        
    return(q_list,task)
    
#-----------------------------------------------------------
# move task files
#-----------------------------------------------------------

def move_file(task,src,dst,file):

    print(src)
    print(dst)
    sheduler_root_dir = task["root_dir"] + "/" + task["sch_cfg"]["scheduler_dir"]
    if os.path.isdir(sheduler_root_dir):
        try:
            if not os.path.isdir(dst):
                os.makedirs(dst,0o775)
        except Exception as err:
            print('ERROR > creating "%s" directory failed, error "%s"'%(dst,err))
            sys.exit(1)
    else:
        print('ERROR > cannot find scheduler path "%s"'%(sheduler_root_dir))
        exit(1)

    src = src + "/" + file
    trg = dst + "/" + file
    if os.path.isfile(trg):
        os.remove(trg)
    dest = shutil.move(src, dst)  

    return

def q_move_task_to_pending(task):

    print("-"*32)
    print("move to pending")

    path_src = task["root_dir"] + "/" + task["sch_cfg"]["scheduler_dir"] + "/" + task["sch_cfg"]["waiting_dir"]
    path_dst = task["root_dir"] + "/" + task["sch_cfg"]["scheduler_dir"] + "/" + task["sch_cfg"]["pending_dir"]

    print("src",path_src)
    print("dst",path_dst)

    move_file(task,path_src,path_dst,task["file_name"])
    
    print("-"*32)
    
    return(0)

def q_move_task_to_passed(task):

    print("-"*32)
    print("move to passed")

    path_src = task["root_dir"] + "/" + task["sch_cfg"]["scheduler_dir"] + "/" + task["sch_cfg"]["pending_dir"]
    path_dst = task["root_dir"] + "/" + task["sch_cfg"]["scheduler_dir"] + "/" + task["sch_cfg"][ "passed_dir"]

    print("src",path_src)
    print("dst",path_dst)

    move_file(task,path_src,path_dst,task["file_name"])

    print("-"*32)
    
    return(0)

def q_move_task_to_failed(task):

    print("-"*32)
    print("move to failed")
 
    path_src = task["root_dir"] + "/" + task["sch_cfg"]["scheduler_dir"] + "/" + task["sch_cfg"]["pending_dir"]
    path_dst = task["root_dir"] + "/" + task["sch_cfg"]["scheduler_dir"] + "/" + task["sch_cfg"][ "failed_dir"]

    print("src",path_src)
    print("dst",path_dst)

    move_file(task,path_src,path_dst,task["file_name"])

    print("-"*32)
    
    return(0)
    
#-----------------------------------------------------------

def main():

    parser                      = ArgumentParser()

    parser.add_argument("-pDir",   "--project_dir" ,  dest="pdir"   ,help="project directory",            	metavar="PATH",     required=True)
    parser.add_argument("-sDir",   "--scheduler_dir", dest="sdir"   ,help="scheduler directory",            	metavar="PATH",     required=True)
    parser.add_argument("-ug",     "--use_gpu"      , dest="ug"     	,help="specify which gpu should be used", nargs='*', default=[]  , type=int,  metavar='I',    required=False)
    parser.add_argument("-v",      "--verbose"    	, dest="verbose"	,help="verbose level",		                  	        required=False)

    args            	        = parser.parse_args()

    #-------------------------------------------------------------------------------------------------

    prj_dir            	        = args.pdir
    prj_path                    = os.path.normpath(prj_dir)
    
    sch_dir            	        = args.sdir
    sch_path                    = os.path.normpath(prj_path + "/" + sch_dir)

    sch_cfg                     = {}
    sch_cfg["pool"]             = sch_dir
    sch_cfg["scheduler_dir"]    = "scheduler"
    sch_cfg["waiting_dir"]      = "waiting"
    sch_cfg["passed_dir"]       = "passed"
    sch_cfg["pending_dir"]      = "pending"
    sch_cfg["failed_dir"]       = "failed"


    print('*' * 50)
    print("main dirs  :")
    print("  project    :",prj_dir)
    print("  pool       :",sch_dir)
    print("  scheduler  :",sch_path)
    print("sub dirs  (scheduler/) :")
    print("  waiting    :",sch_cfg["waiting_dir"])
    print("  pending    :",sch_cfg["pending_dir"])
    print("  passed     :",sch_cfg["passed_dir"])
    print("  failed     :",sch_cfg["failed_dir"])
    print('*' * 50)

    if not os.path.isdir(sch_path):
        print('ERROR > cannot find "%s" directory'%(sch_path))
        exit(-1)

    #-------------------------------------------------------------------------------------------------

    dcnt                = torch.cuda.device_count()
    
    for did in range(0,dcnt):
        dname = torch.cuda.get_device_name(did)
        print("device %d -> %s"%(did,dname))

    print('*' * 50)
    if dcnt==0:
        exit(1)

    #-------------------------------------------------------------------------------------------------
    # start workers
    #-------------------------------------------------------------------------------------------------
    
    dev_num 	= 1#torch.cuda.device_count()
    dev_list 	= np.arange(0,dev_num)
    req_num 	= len(args.ug)
    req_list 	= args.ug

    if req_num>0:
        dev_num  = req_num
        tmp_list = []
        for r in req_list:
            if r in dev_list:
                 tmp_list.append(r)
            else:
                print("gpu id '%d' is not available!"%r) 
                exit(1)

        dev_list = tmp_list
        dev_num  = len(tmp_list)
    
    print("active devs :",dev_list)

    #-------------------------------------------------------------------------------------------------
    # start workers
    #-------------------------------------------------------------------------------------------------

    worker_handle = []
    
    for pid in range(dev_num):
        
        wqueue_in           = Queue()
        wqueue_out          = Queue()
       
        req_id              = dev_list[pid]
        logging.info("New process on GPU id = %d"%req_id) 
        wprocess            = Process(target=worker, args=(wqueue_out, wqueue_in, req_id), name="worker_{:02}".format(pid))
        wprocess.daemon     = True
        worker_handle.append({ "worker":wprocess, "pipe_out":wqueue_out, "pipe_in":wqueue_in})

    for pr in worker_handle:
        pr["worker"].start()

    for pr in worker_handle:
        pr["pipe_out"].put(["ready",None])

    #-------------------------------------------------------------------------------------------------
    # main loop
    #-------------------------------------------------------------------------------------------------

    finish = False
    
    while not finish:
    
        for pr in worker_handle:
            if not pr["pipe_in"].empty(): 

                res = pr["pipe_in"].get()

                if res[0] == "executed" :

                    if res[1] != 0:
                        q_move_task_to_failed(task) 
                    else:
                        q_move_task_to_passed(task) 

                if (res[0] == "ready" or res[0] == "executed"):
                    q_list          = q_get_task_list([sch_path],sch_cfg)
                    q_list, task    = q_get_one_task(q_list)

                    print(task)
                    if task is not None:
                        err     = q_move_task_to_pending(task)
                        pr["pipe_out"].put(["execute",task])
                    else:
                        finish = True
                    
    time.sleep(0.5)

    #-------------------------------------------------------------------------------------------------

    for pr in worker_handle:
        pr["pipe_out"].put(["quit",None])

    for pr in worker_handle:
        pr["worker"].join()

#-----------------------------------------------------------

if __name__ == '__main__':
    main()

