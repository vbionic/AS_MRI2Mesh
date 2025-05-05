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
#-----------------------------------------------------------------------------------------
from multiprocessing    import Process, Queue
import argparse     
#-----------------------------------------------------------------------------------------
curr_script_path = os.path.dirname(os.path.abspath(__file__))
flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
from v_utils.v_dataset import MRIDataset, expand_session_dirs
#-----------------------------------------------------------------------------------------

def end_process(process):
    process.join()
    logging.info(" Joined process {} with exit code = {}...".format(process.name, process.exitcode))
    if(process.exitcode != 0):
        logging.error(' Child process returned with error {} (process = {})'.format(process.exitcode, process))

#-----------------------------------------------------------

# def execute(cmd, odir, idir, did):
 
    # xdir    = cmd[0]
    # xpth    = cmd[1]    

    # cmdline   = 'python process_st05_eval_pth.py '
    # cmdline  += '-oDir {} '.format(os.path.normpath(os.path.join(odir, xdir)))
    # cmdline  += '-iDir {} '.format(os.path.normpath(os.path.join(idir, xdir)))
    # cmdline  += '-pth {} '.format(xpth)
    # cmdline  += '--force_gpu_id ' + str(did) + ' '

    # logging.info(cmdline)
    
    # ret = os.system(cmdline)
    
    # return ret

def execute(pdir, ses, iDir, oDir, pPath, gpu_id, oshp, plim, iSfx, flg, upo):
  
    cmdline   = 'python flexnet/evaluation/eval_pth.py'
    cmdline  += ' -pth {} '.format(os.path.normpath(pdir + "/" + pPath))

    if flg[0]:
        cmdline  += ' -sB '
    if flg[1]:             
        cmdline  += ' -sL '
    if flg[2]:             
        cmdline  += ' -sP '
    if flg[3]:
        cmdline  += ' -sN '
    if flg[4]:
        cmdline  += ' -sC '
    if flg[5]:
        cmdline  += ' -sS '

    if plim is not None:
        cmdline  += ' -plim {} '.format(plim)

    if iSfx is not None:
        cmdline  += ' -iSfx {} '.format(os.path.normpath(iSfx))
    
    if oshp is not None:
        cmdline  += ' -oShp ' + " ".join(oshp)  
        
    cmdline  += ' --force_gpu_id ' + str(gpu_id) + ' '

    if flg[6] is not None:
       cmdline  += ' -refo '
    
    if flg[7]:
        cmdline  += ' -fH '

    ret     = 0
        
    dst_dir  = os.path.normpath(os.path.join(oDir, ses))
            
    if upo==False or (upo and os.path.exists(dst_dir)==False): 
        
        # sessions loop
        session_l = expand_session_dirs(args.session_id, iDir)

        for ses in session_l:
            cmdX     = cmdline
            cmdX    += ' -oDir {} '.format(oDir)
            cmdX    += ' -iDir {} '.format(iDir)
            cmdX    += ' -sesID {} '.format(ses)

            logging.info(cmdX)

            ret      += os.system(cmdX)

    return ret


#-----------------------------------------------------------

def worker(qin,qout,idir,odir,did,wid):

    logging.info("worker %d started on device %d"%(wid,did))

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
                logging.info("arg:{} @ {}".format(cmd[1],cmd[2]))
                
                ses_id = cmd[2][3]
                oshp   = cmd[2][4]
                plim   = cmd[2][5]
                isfx   = cmd[2][6]

                pdir   = cmd[2][0] + "/" + cmd[1][0]
                idir   = cmd[2][1]
                odir   = cmd[2][2] + "/" + cmd[1][0]
                
                ppat   = cmd[1][1]
              
                ret    = execute(pdir,ses_id,idir,odir,ppat,did,oshp,plim,isfx,cmd[3],cmd[4])

                qout.put(["executed",ret,cmd[1],odir,idir])
            else:
                time.sleep(0.1)
        
    return

#-----------------------------------------------------------

def save_queue(q,odir,fname):

    spath = os.path.normpath(os.path.join(odir,fname))
    print("updating log file : ",spath)

    q_list           = list(q.queue)
    z                = []

    for x in q_list:
        z.append(x)

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
    text += ",proc:"    + "eval"
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
class NegateAction(argparse.Action):
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, option[1:3] != 'no')
        

def main():

    wr_waiting  = queue.Queue() 
    wr_passed   = queue.Queue() 
    wr_failed   = queue.Queue() 

    parser              = argparse.ArgumentParser()
    
    parser.add_argument("-ses",    "--session_id"       , default = "*/*"   ,help="session id", nargs='+',  metavar="PATH", required=False)

    parser.add_argument("-bDir",    "--bot_dir"         , dest="bdir"       ,help="bot dir",                                    required=False)
    parser.add_argument("-iDir",    "--input_dir"       , dest="idir"       ,help="input dir",                                  required=True)
    parser.add_argument("-oDir",    "--output_dir"      , dest="odir"       ,help="output dir",                                 required=False)
    parser.add_argument("-pDir",    "--pth_dir"         , dest="pdir"       ,help="pth dir",                                    required=True)

    parser.add_argument("-updo",    "--update_only"     , dest="upo"        ,help="force only update",    action='store_true',   default=False,                         required=False)
    parser.add_argument("-refo",    "--ref_only"        , dest="refo"       ,help="ref only",             action='store_true',   default=False,                         required=False)

    parser.add_argument("-ug",     "--use_gpu"          , dest="ug"         ,help="specify which gpu should be used", nargs='*', default=[0], type=int,  metavar='I',   required=False)
    parser.add_argument("-tn",     "--thread_num"       , dest="tn"         ,help="thread number",                    nargs=1,   default=1  , type=int,  metavar='I',   required=False)

    parser.add_argument("-sB",      "-no-sB"            , dest="f_box"      ,help="generate box file",       action=NegateAction,  nargs=0, default=True,   required=False)
    parser.add_argument("-sL",      "-no-sL"            , dest="f_label"    ,help="generate label file",     action=NegateAction,  nargs=0, default=True,   required=False)
    parser.add_argument("-sP",      "-no-sP"            , dest="f_prob"     ,help="generate prob file",      action=NegateAction,  nargs=0, default=False,  required=False)
    parser.add_argument("-sN",      "-no-sN"            , dest="f_prob_nl"  ,help="generate prob nlfile",    action=NegateAction,  nargs=0, default=False,  required=False)
    parser.add_argument("-sC",      "-no-sC"            , dest="f_poly"     ,help="generate polygons file",  action=NegateAction,  nargs=0, default=False,  required=False)
    parser.add_argument("-sS",      "-no-sS"            , dest="f_stat"     ,help="generate stats file",     action=NegateAction,  nargs=0, default=True,   required=False)

    parser.add_argument("-plim",    "--polygons_lim"    , dest="plim"       ,help="limit of polygons number",               required=False)
    parser.add_argument("-fH",      "--fill_holes"      , dest="fholes"     ,help="fill holes of polygons, masks and labels", action='store_true', default=False, required=False)
    parser.add_argument("-oShp",    "--output_shapes"   , dest="oshp"       ,help="output shapes names",  nargs='+',        required=False)
    parser.add_argument("-iSfx",    "--input_sufix"     , dest="isfx"       ,help="input sufix",                            required=False)
    parser.add_argument("-v",       "--verbose"         , dest="verbose"    ,help="verbose level",                          required=False)

    args                = parser.parse_args()
    timestamp           = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    
    #--------------------------------------------------------------------------------------

    pdir                = args.pdir
    idir                = args.idir
    odir                = args.idir if args.odir == None else args.odir
    bdir                = "as_data/bot_channels/"     if args.bdir == None else args.bdir + "/"

    ses_id              = args.session_id
    
    bot_file            = open(bdir + "bot_channel_eval_%s.log"%timestamp, "w+")
    
    #--------------------------------------------------------------------------------------

    output_path = os.path.normpath(odir)

    try:
        if not os.path.isdir(output_path):
            pathlib.Path(output_path).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('Creating "%s" directory failed, error "%s"'%(output_path,err))
        sys.exit(1)

    #--------------------------------------------------------------------------------------

    log_format          = "%(asctime)s [%(levelname)s] %(message)s"
    initial_log_fn      = output_path +  "/dbg_eval_pool_%s.log"%timestamp
    logging_level       = logging.INFO if ((args.verbose is None) or (args.verbose == "off")) else logging.DEBUG
    
    logging.basicConfig(level=logging_level, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()

    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    logging.info('*' * 50)

    #--------------------------------------------------------------------------------------

    xpth_path  = os.path.normpath(pdir)
  
    dcnt                = torch.cuda.device_count()
    
    for did in range(0,dcnt):
        dname = torch.cuda.get_device_name(did)
        logging.info("device %d -> %s"%(did,dname))

    logging.info('*' * 50)
    if dcnt==0:
        exit(-1)

    #create list of directories 

    glob_list   = glob.glob( os.path.normpath(xpth_path + '/*'))
    dir_list    = []

    for d in glob_list:
        if os.path.isdir(d):
             dir_list.append(d)

    task_list   = []
    
    for d in dir_list:
        dc = d.replace('[', '[[]')
        pth_list   = glob.glob(dc + "/*.pth")
        for p in pth_list:
            task_list.append([os.path.basename(d),os.path.basename(p)])

    for p in task_list:
        logging.info("dir(%s), pth(%s) "%(p[0],p[1]))

    logging.info("dirs number : %d"%len(dir_list))
    logging.info("pths number : %d"%len(task_list))

    tasks_queue         = queue.Queue() 
    
    for p in task_list:
        tasks_queue.put(p)

    task_list           = list(tasks_queue.queue)

    logging.info("generated tasks : %d"%len(task_list))

    # start workers
    
    worker_handle = []
    
    dev_num     = torch.cuda.device_count()
    dev_list    = np.arange(0,dev_num)
    req_num     = len(args.ug)
    req_list    = args.ug

    if req_num>0:
        dev_num  = req_num
        tmp_list = []
        for r in req_list:
            if r in dev_list:
                if r in tmp_list:
                    logging.error("gpu id '%d' is called more than once!"%r) 
                    exit(1)
                else:
                    tmp_list.append(r)
            else:
                logging.error("gpu id '%d' is not available!"%r) 
                exit(1)

        dev_list = tmp_list
        dev_num  = len(tmp_list)

    thread_num = args.tn

    logging.info("assigned gpus {}".format(dev_list)) 
    logging.info("worker(s)  {} with {} thread(s) -> {} process(es)".format(len(dev_list),thread_num[0],len(dev_list)*thread_num[0])) 

    process_num = len(dev_list)*thread_num[0]

    talk_to_bot(bot_file, ca=dev_num, th=thread_num[0], tot=tasks_queue.qsize(), pas=0, fai=0, cur=0, ph="begin", dr=idir)    

    for pid in range(process_num):
        
        wqueue_in           = Queue()
        wqueue_out          = Queue()
       
        req_id              = dev_list[pid%dev_num]
        logging.info("New process {} on GPU {}".format(pid,req_id)) 
        wprocess            = Process(target=worker, args=(wqueue_out, wqueue_in, idir, odir, req_id, pid), name="worker_{:02}".format(pid))
        wprocess.daemon     = True
        worker_handle.append({ "worker":wprocess, "pipe_out":wqueue_out, "pipe_in":wqueue_in})

    for pr in worker_handle:
        pr["worker"].start()

    for pr in worker_handle:
        pr["pipe_out"].put(["ready",None])

    tq_len = tasks_queue.qsize()

    b_pass = 0
    b_fail = 0
    b_curr = 0
    b_time_out = 0
    b_send_f = 0

    while not tasks_queue.empty():
    
        for pr in worker_handle:
            if not pr["pipe_in"].empty(): 

                save_queue(tasks_queue,output_path,"eval_waiting_jobs.log")
                save_queue(  wr_failed,output_path,"eval_failed_jobs.log")
                save_queue(  wr_passed,output_path,"eval_passed_jobs.log")

                res = pr["pipe_in"].get()
                logging.info(res)

                if res[0] == "executed" :
                    b_curr -= 1
                    
                    if res[1]:
                        wr_failed.put(res[2]) 
                    else:
                        wr_passed.put(res[2]) 

                if (res[0] == "ready" or res[0] == "executed") and (not tasks_queue.empty()):
                    b_curr += 1
                    exe = tasks_queue.get()
                    ext = [pdir,idir,odir,ses_id,args.oshp,args.plim,args.isfx]
                    flg = [args.f_box,  args.f_label, args.f_prob, args.f_prob_nl, args.f_poly, args.f_stat, args.refo, args.fholes]
                    
                    pr["pipe_out"].put(["execute",exe,ext,flg,args.upo])

                b_pass = wr_passed.qsize()
                b_fail = wr_failed.qsize()
        
                talk_to_bot(bot_file, ca=dev_num, th=thread_num[0], tot=tq_len, pas=b_pass, fai=b_fail, cur=b_curr, ph="work", dr=idir)
                b_send_f = 1
                
        #logging.info(len(tasks_queue.queue))
        time.sleep(0.1)
        if b_send_f:
            b_time_out  = 0
            b_send_f    = 0
        else:
            b_time_out += 1
        
        if b_time_out>100:
         
                b_time_out  = 0
                b_pass      = wr_passed.qsize()
                b_fail      = wr_failed.qsize()
        
                talk_to_bot(bot_file, ca=dev_num, th=thread_num[0], tot=tq_len, pas=b_pass, fai=b_fail, cur=b_curr, ph="work", dr=idir)
        

    save_queue(tasks_queue,output_path,"eval_waiting_jobs.log")
    save_queue(  wr_failed,output_path,"eval_failed_jobs.log")
    save_queue(  wr_passed,output_path,"eval_passed_jobs.log")

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

    b_pass = wr_passed.qsize()
    b_fail = wr_failed.qsize()
        
    talk_to_bot(bot_file, ca=dev_num, th=thread_num[0], tot=tq_len, pas=b_pass, fai=b_fail, cur=0, ph="end", dr=idir)

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

    save_queue(tasks_queue,output_path,"eval_waiting_jobs.log")
    save_queue(  wr_failed,output_path,"eval_failed_jobs.log")
    save_queue(  wr_passed,output_path,"eval_passed_jobs.log")

#-----------------------------------------------------------

if __name__ == '__main__':
    main()

