import os, sys
import pathlib
import glob
import json
import logging
import time
import torch
import queue
import re
import json
import numpy as np
from datetime import datetime
#---------------------------------------------------------
from multiprocessing    import Process, Queue
from argparse           import ArgumentParser
#--------------------------------------------------------

def end_process(process):
    process.join()
    logging.info(" Joined process {} with exit code = {}...".format(process.name, process.exitcode))
    if(process.exitcode != 0):
        logging.error(' Child process returned with error {} (process = {})'.format(process.exitcode, process))

#-----------------------------------------------------------

def execute(cmd, od, id):
 
    json_list = []
    param_list = []

    for prm in cmd:
        if 'json' in prm:
            json_list.append(prm)
        else:
            param_list.append(prm)

    cfg_args    = ''
    prm_args    = ''
    oname       = ''
    
    for prm in json_list: 
        print(prm)
        cfg_args    += prm + ' '
        re_res       = re.search(r"\[([A-Za-z0-9_+&@]+)\]", prm)
        if not re_res is None:
            tmpx = re_res.group()[1:-1]
            tmpp = tmpx.split("+")
            for pr in tmpp:
               tmpa      = pr.split("@") 
               oname    += tmpa[0] + "[" + tmpa[1] + "]" + '_'
        
    for prm in param_list:
        prm_args    += '--' + prm[2:] + ' '
        res         =  prm[2:].split('_')
        sc          = ''
        
        for w in res:
            sc += w[0].lower()

        res   = prm[1:].split(' ')

        if prm[0:2]=="--":
            oname += sc + '[' + res[1] + ']' + '_'
        elif prm[0:2]!="++":
            logging.error('unknown parameter type (%s)'%prm)
            exit(-1)
            

    oname = oname[:-1]
    
    cmdline   = 'python flexnet/training/pytorch_model_train.py '
    cmdline  += '--out_dir {} '.format(os.path.normpath(os.path.join(od, oname)))
    cmdline  += '--cfg ' + cfg_args + ' ' 
    cmdline  += prm_args    
    cmdline  += '--force_gpu_id ' + str(id) + ' '
    cmdline  += '--session_dirs "*/*" '

    logging.info(cmdline)
    
    ret = os.system(cmdline)
    
    return ret

#-----------------------------------------------------------

def recurent_loop(list, cfg, cmd, tasks_queue):

    name     = list[0]
    listx    = list[1:]
    lf       = False
    emit     = True

    if name[0] == '@' or name[0:2] == '++':
        name = name[1:]
        lf   = True
        if name[0] == '#':
            emit = False

    if lf:
        if name[0:2] == '--' or name[0:2] == '++':
            setx    = [name.split(" ")[1]] 
            name    = name.split(" ")[0]
        else:
            setx    = [name] 
    else:
        setx        = cfg[name] 

    if name[0:2] == '--' or name[0:2] == '++':
    
        res      = 0
        params   = setx 

        for x in params:
            cmdx = cmd.copy()
            text = name + ' ' + x
            cmdx.append(text)  

            if len(listx)!=0:
                res += recurent_loop(listx,cfg,cmdx,tasks_queue)
            else:
                tasks_queue.put(cmdx)
                #execute(cmdx)
                #logging.info("----------------------------")
                res += 1

    else:
    
        json_list   = setx 
        res         = 0
    
        for x in json_list:
            cmdx = cmd.copy()
            path = x
            cmdx.append(path)  

            if len(listx)!=0:
                res += recurent_loop(listx,cfg,cmdx,tasks_queue)
            else:
                
                tasks_queue.put(cmdx)
                #execute(cmdx)
                #logging.info("----------------------------")
                res += 1

    return res

#-----------------------------------------------------------

def cfg_check_loop(list, cfg):

    errf = []
    okf  = []
    ovf  = []
    res  = 0
    
    for name in list:

        lf   = False
        
        if name[0] == '@':
            name = name[1:]
            lf   = True

        if lf:
            if name[0:2] == '--' or name[0:2] == '++':
                setx    = [name.split(" ")[1]] 
            else:
                setx    = [name] 
        else:
            setx        = cfg[name] 

        if name[0:2] == '--' or name[0:2] == '++':

            s = [str(i) for i in setx] 

                  
            ovf.append(name + ' with set [' + ','.join(s) + ']' )
            res += 1
        
        elif name[0] != '-' and name[0] != '+':
        
            for item in setx:
                path        = item
     
                if path[0:2]!="--" and path[0:2]!="++":
                    try:
                        cfg_file	 	= open(path,"r");   
                        try:
                            cfg_json	= json.load(cfg_file);     
                            okf.append(path)
                        except:
                            errf.append(path)
                    except:
                        errf.append(path)
    
                    res += 1
                else:
                    ovf.append(path)
                    res += 1

        else:
            
            logging.error('unknown config type (%s)'%name)
            exit(-1)


    logging.info('*' * 50)
    logging.info('config files : %d'%res)
    logging.info('-' * 50)

    for oname in ovf:
        logging.info("configuration switch %s is OK"%oname)

    for oname in okf:
        logging.info("configuration file %s is OK"%oname)

    for ename in errf:
        logging.error("configuration file %s is WRONG"%ename)

    logging.info('*' * 50)
            
    return len(errf)

#-----------------------------------------------------------

def worker(qin,qout,od,id):

    logging.info("worker %d started"%(id))
    
    while True:

        if qin.empty():
            time.sleep(0.1)
        else:
            cmd = qin.get()
            logging.info("cmd:{}".format(cmd[0]))
        
            if cmd[0] == 'quit':
                qout.put(["quit",None])
                break
            if cmd[0] == 'ready':
                qout.put(["ready",None])
            if cmd[0] == 'execute':
                logging.info("arg:{}".format(cmd[1]))
                ret  = execute(cmd[1],od,id)
                qout.put(["executed",ret,cmd[1],od,id])
            else:
                time.sleep(0.1)
        
    return

#-----------------------------------------------------------

def save_queue(q,odir,fname):

    spath = os.path.normpath(os.path.join(odir,fname))

    q_list           = list(q.queue)
    z 				 = []

    print("updating log file : ",spath," with ",len(q_list)," records")
        
    for x in q_list:
        a = []        
        for n in x:
            n = '@'+n
            a.append(n)
        z.append(a)

    dict = {}
    dict["process"]   = z

    #print(dict)

    with open(spath, 'w') as f:
        json.dump(dict, f, indent=4)

    return

#-----------------------------------------------------------

def talk_to_bot(file, ca, th, tot, pas, fai, cur, ph, dr):

    now   = datetime.now()
    
    text  = "time:"     +  now.strftime("%d-%m-%Y %H.%M.%S")
    text += ",proc:"    + "training"
    text += ",cards:"   + str(ca)  
    text += ",threads:" + str(th)  
    text += ",tot:"     + str(tot)  
    text += ",pass:"    + str(pas)  
    text += ",fail:"    + str(fai)  
    text += ",curr:"    + str(cur)  
    text += ",phase:"   + str(ph)  
    text += ",dir:"     + str(dr)  
    text += "\n"

    file.write(text)
    file.flush()
    
#-----------------------------------------------------------

def main():

    wr_waiting 	    = queue.Queue() 
    wr_passed 	    = queue.Queue() 
    wr_failed 	    = queue.Queue() 
    wr_inprogress 	= queue.Queue() 

    parser              = ArgumentParser()

    parser.add_argument("-cfg",    "--cfg" 	     	, dest="cfg"     	,help="json file with configs pool",	metavar="PATH",     required=True)
    parser.add_argument("-oDir",   "--output_dir"  	, dest="odir"    	,help="output directory",            	metavar="PATH",     required=False)
    parser.add_argument("-bch",    "--bot_channel"  , dest="bch"        ,help="bot channel path",  			                        required=True)
    parser.add_argument("-ug",     "--use_gpu"      , dest="ug"     	,help="specify which gpu should be used", nargs='*', default=[]  , type=int,  metavar='I',    required=False)
    parser.add_argument("-v",      "--verbose"    	, dest="verbose"	,help="verbose level",		                  	        required=False)
    parser.add_argument("-to",      "--test_only"  	, dest="to"      	,help="for test only",		                  	        required=False)

    args            	= parser.parse_args()

    #-------------------------------------------------------------------------------------------------
    
    cfg_path 			= args.cfg
    bot_path 			= args.bch
    out_dir            	= "out_temp" if args.odir is None else args.odir

    out_path            = os.path.normpath(out_dir)

    try:
        if not os.path.isdir(out_path):
            pathlib.Path(out_path).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('Creating "%s" directory failed, error "%s"'%(out_path,err))
        sys.exit(1)

    #-------------------------------------------------------------------------------------------------

    log_format          = "%(asctime)s [%(levelname)s] %(message)s"
    initial_log_fn      = out_path + "/dbg_training_pool.log"
    logging_level       = logging.INFO if ((args.verbose is None) or (args.verbose == "off")) else logging.DEBUG
    
    logging.basicConfig(level=logging_level, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()

    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    logging.info('*' * 50)

    #-------------------------------------------------------------------------------------------------

    bot_file            = open(bot_path, "w+")

    #-------------------------------------------------------------------------------------------------

    dcnt                = torch.cuda.device_count()
    
    for did in range(0,dcnt):
        dname = torch.cuda.get_device_name(did)
        logging.info("device %d -> %s"%(did,dname))

    logging.info('*' * 50)
    if dcnt==0:
        exit(-1)

    #-------------------------------------------------------------------------------------------------
        
    cfg_path            = os.path.normpath(cfg_path)
    
    try:
        cfg_file	 	= open(cfg_path,"r");   
        logging.info("opening training config file %s"%cfg_path)
    except:
        logging.error("cannot open training config file %s"%cfg_path)
        exit(1)

    #-------------------------------------------------------------------------------------------------
  
    cfg_json	 	    = json.load(cfg_file);     
    
    #-------------------------------------------------------------------------------------------------
    #loop section
    #-------------------------------------------------------------------------------------------------
    
    loop_list           = cfg_json["process"]
    res = 0
    
    for x in loop_list:
        logging.info('*' * 50)

        res            += cfg_check_loop(x,cfg_json)
      
        if res!=0:
            logging.error("configuration files error(s)")
            exit(1)
    
    tasks_queue         = queue.Queue() 

    res = 0
    for x in loop_list:
        logging.info('*' * 50)

        res            += recurent_loop(x,cfg_json,[],tasks_queue)

    task_list           = list(tasks_queue.queue)

    logging.info("generated tasks : %d"%res)

    #-------------------------------------------------------------------------------------------------
    # start workers
    #-------------------------------------------------------------------------------------------------
    
    worker_handle = []
    
    dev_num 	= torch.cuda.device_count()
    dev_list 	= np.arange(0,dev_num)
    req_num 	= len(args.ug)
    req_list 	= args.ug

    if req_num>0:
        dev_num  = req_num
        tmp_list = []
        for r in req_list:
            if r in dev_list:
                 tmp_list.append(r)
#                if r in tmp_list:
#                    logging.error("gpu id '%d' is called more than once!"%r) 
#                    exit(1)
#                else:
#                    tmp_list.append(r)
            else:
                logging.error("gpu id '%d' is not available!"%r) 
                exit(1)

        dev_list = tmp_list
        dev_num  = len(tmp_list)

    for pid in range(dev_num):
        
        wqueue_in           = Queue()
        wqueue_out          = Queue()
       
        req_id              = dev_list[pid]
        logging.info("New process on GPU id = %d"%req_id) 
        wprocess            = Process(target=worker, args=(wqueue_out, wqueue_in, out_dir, req_id), name="worker_{:02}".format(pid))
        wprocess.daemon     = True
        worker_handle.append({ "worker":wprocess, "pipe_out":wqueue_out, "pipe_in":wqueue_in})

    for pr in worker_handle:
        pr["worker"].start()

    for pr in worker_handle:
        pr["pipe_out"].put(["ready",None])

    if args.to is not None:
        exit(-1)
 
    tq_len = tasks_queue.qsize()

    b_pass      = 0
    b_fail      = 0
    b_curr      = 0
    b_time_out  = 0
    b_send_f    = 0
    
    tr_name = os.path.basename(out_dir)
    talk_to_bot(bot_file, ca=dev_num, th=1, tot=tasks_queue.qsize(), pas=0, fai=0, cur=0, ph="begin", dr=tr_name)    

    #-------------------------------------------------------------------------------------------------
    # main loop
    #-------------------------------------------------------------------------------------------------
    
    while not tasks_queue.empty():
    
        for pr in worker_handle:
            if not pr["pipe_in"].empty(): 

                save_queue(tasks_queue,     out_path,"training_waiting_jobs.log")
                save_queue(  wr_failed,     out_path,"training_failed_jobs.log")
                save_queue(  wr_passed,     out_path,"training_passed_jobs.log")
                save_queue(  wr_inprogress, out_path,"training_inprogress_jobs.log")

                res = pr["pipe_in"].get()
                logging.info(res)

                if res[0] == "executed" :
                    b_curr -= 1
                    wr_inprogress.get()
                    if res[1] != 0:
                        wr_failed.put(res[2]) 
                    else:
                        wr_passed.put(res[2]) 

                if (res[0] == "ready" or res[0] == "executed") and (not tasks_queue.empty()):
                    b_curr += 1
                    exe = tasks_queue.get()
                    pr["pipe_out"].put(["execute",exe])
                    wr_inprogress.put(exe)

                b_pass = wr_passed.qsize()
                b_fail = wr_failed.qsize()
        
                talk_to_bot(bot_file, ca=dev_num, th=1, tot=tq_len, pas=b_pass, fai=b_fail, cur=b_curr, ph="work", dr=tr_name)
                b_send_f = 1

        if b_send_f:
            b_time_out  = 0
            b_send_f    = 0
        else:
            b_time_out += 1
        
        if b_time_out>60:
         
                b_time_out  = 0
                b_pass      = wr_passed.qsize()
                b_fail      = wr_failed.qsize()
        
                talk_to_bot(bot_file, ca=dev_num, th=1, tot=tq_len, pas=b_pass, fai=b_fail, cur=b_curr, ph="work", dr=tr_name)

        #logging.info(len(tasks_queue.queue))
        time.sleep(0.5)

    #-------------------------------------------------------------------------------------------------

    save_queue(tasks_queue,out_path,"training_waiting_jobs.log")
    save_queue(  wr_failed,out_path,"training_failed_jobs.log")
    save_queue(  wr_passed,out_path,"training_passed_jobs.log")
    save_queue(  wr_inprogress ,out_path,"training_inprogress_jobs.log")

    rem = tq_len - wr_passed.qsize() - wr_failed.qsize()

    while rem > 0:
        for pr in worker_handle:
            if not pr["pipe_in"].empty(): 
                res = pr["pipe_in"].get()
                logging.info(res)

                if res[0] == "executed" :
                    if res[1]:
                        wr_failed.put(res[2]) 
                    else:
                        wr_passed.put(res[2]) 
            
        rem = tq_len - wr_passed.qsize() - wr_failed.qsize()

    for pr in worker_handle:
        pr["pipe_out"].put(["quit",None])

    for pr in worker_handle:
        pr["worker"].join()

    print("total  : ",tq_len)
    print("passed : ",wr_passed.qsize())
    print("failed : ",wr_failed.qsize())

    print("list of passed tasks :")
    p_list           = list(wr_passed.queue)
    for x in p_list:
        print(x) 

    if len(list(wr_failed.queue)) >0 :
        print("list of failed tasks :")
        f_list           = list(wr_failed.queue)
        for x in f_list:
            print(x) 

    save_queue(tasks_queue,out_path,"training_waiting_jobs.log")
    save_queue(  wr_failed,out_path,"training_failed_jobs.log")
    save_queue(  wr_passed,out_path,"training_passed_jobs.log")
    save_queue(  wr_inprogress ,out_path,"training_inprogress_jobs.log")

#-----------------------------------------------------------

if __name__ == '__main__':
    main()

