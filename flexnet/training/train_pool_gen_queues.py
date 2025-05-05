import os, sys
import glob
import json
import logging
import time
import re
import json
import numpy as np
import queue
#---------------------------------------------------------
from datetime           import datetime
from argparse           import ArgumentParser
#--------------------------------------------------------

def get_name_from_parameters(parameter_set):

    out_name = ""
    for line in parameter_set:
        key = list(line.keys())[0]
        # print(line[key])
        if key[0]!='#' and key[0]!='+':
            if type(line[key]) == str and line[key].find(".json")!=0:
                oname = ""
                fname = os.path.basename(line[key])
                sname = fname.split("[")
                if len(sname)>1:
                    prm_str     = sname[1].split("]")[0]
                    prm_list    = prm_str.split("+")
                    for p in prm_list:
                        oname += p.split("@")[0] + "[" + p.split("@")[1] + "]_"
                #print("json ",key,":",line[key], ">>>",oname)
                out_name += oname
            else: 
                fname = key.replace("-","").replace("-","")
                sname = fname.split("_") 
                oname = ""
                for w in sname:
                    oname += w[0]
                #print("parse ",key,":",line[key],">>>",oname)
                out_name += oname + "[" + str(line[key]) + "]_"
        #else:
            #print("skip  ",key,":",line[key])

    #print(out_name[:-1])
    if len(out_name) >0:
        return(out_name[:-1], 0)
    else:
        return("", 1)

def parse_consts(section, cfg):

    s_param_list = []
    p_param_list = []
    error_list   = []

    for key in section:
        if type(key) == dict:
            spl,ppl,erl     = parse_consts(key,cfg)
            s_param_list    = [*s_param_list, *spl]
            p_param_list    = [*p_param_list, *ppl]
            error_list      = [*error_list,   *erl]
            continue

        if key == '#unroll':
            continue

        if key == '#list' :
            spl,ppl,erl     = parse_consts(cfg[section[key]],cfg)
            s_param_list    = [*s_param_list, *spl]
            p_param_list    = [*p_param_list, *ppl]
            error_list      = [*error_list,   *erl]
        elif key[0] == '#':
            s_param_list.append({key:section[key]})
        else:
            p_param_list.append({key:section[key]})

    return(s_param_list,p_param_list,[])

#-----------------------------------------------------------

def parse_unroll(section, cfg):

    u_param_list = []
    error_list   = []

    for skey in section:
        if type(skey) == dict:
            if '#unroll' in skey:
                for sub in skey['#unroll']:
                    for key in sub:
                        #print(key)
                        if type(key) == dict:
                            if '#list' in key:
                                if key['#list'] in cfg:
                                    u_param_list.append(cfg[key['#list']])
                                else:
                                    error_list.append(key)
                            else:
                                error_list.append(key)

    return(u_param_list,error_list)

#-----------------------------------------------------------

def unroll_loop(s_param_list, p_param_list, u_param_list, stack):

    error_list   = []
    ses_list     = []
    
    if len(u_param_list)>0:
        loop = u_param_list[-1] 
        next = u_param_list[0:len(u_param_list)-1]
        for item in loop:
            tmp = stack.copy()
            tmp.append(item)
            sub_list = unroll_loop(s_param_list, p_param_list, next, tmp)
            ses_list = [*ses_list,*sub_list]
    else:
        stack = [*s_param_list,*p_param_list,*stack]
        return([stack])

    return(ses_list)

#-----------------------------------------------------------

def unroll_tasks(task_list):

    new_tasks = []
    
    for item in task_list:
        task = []
        for line in item:
            for param in line:
                ptype = type(line[param])
                if ptype==str or ptype==int or ptype==float:
                    #print("ok >",param, line[param])
                    task.append({param:line[param]})    
                elif ptype==list:
                    for arg in line[param]:
                        #print("lst>", param, arg)
                        task.append({param:arg})    
                else: 
                    print("!!!",type(line[param]), param, line[param])
                    task.append({param:line[param]})    

        new_tasks.append(task)

    return(new_tasks)

#-----------------------------------------------------------

def split_tasks(task_list,base_path):

    task_block_list = []

    # phase 1 find requirements 

    for task_block in task_list:

        set_of_req  = {}
        param_set   = {}

        #-------------------------------------------------------------------
        # set default def_* as req_*
        #-------------------------------------------------------------------

        for param in task_block:

            keys = list(param.keys())
            if len(keys)>0 and keys[0].startswith('#def_'):

                key     = keys[0]
                rkey    = key.replace("#def_","#req_")    

                if rkey in set_of_req:
                    set_of_req[rkey] = max(set_of_req[key],param[key])
                else:
                    set_of_req[rkey] = param[key]

        #-------------------------------------------------------------------
        # set req_*
        #-------------------------------------------------------------------

        for param in task_block:

            keys = list(param.keys())
            if len(keys)>0 and keys[0].startswith('#req_'):
                
                key = keys[0]

                if key in set_of_req:
                    if key!='#req_priority':
                        set_of_req[key] = max(set_of_req[key],param[key])
                    else:
                        set_of_req[key] = param[key]
                else:
                    set_of_req[key] = param[key]
            
        #-------------------------------------------------------------------
        # clear $req_* and #def_*
        #-------------------------------------------------------------------

        clear_task_block = []
        for param in task_block:

            keys = list(param.keys())
            if len(keys)>0 and not (keys[0].startswith("#req_") or keys[0].startswith("#def_")):
                #print("clear >",param,keys[0])
                clear_task_block.append(param)

        task_block_list.append({"req":set_of_req,"task":clear_task_block})

    #-------------------------------------------------------------------
    # create queues 
    #-------------------------------------------------------------------

    q_name_list = {}

    for task in task_block_list:
        name = "q"
        req  = task["req"]
        
        for key in req:

            pname = ""
            nlist = key.split("_")
            pname = nlist[1] 
            
            if nlist[1] == 'gpr':
                name += "_" + pname + "[" + '{0:+}'.format(req[key]) + "]"
            else:
                name += "_" + pname + "[" + str(req[key]) + "]"
   
        if name in q_name_list:
            q_name_list[name].append(task["task"])
        else: 
            q_name_list[name] = [task["task"]]


    print("+"*50)

    q_dir = base_path + "/scheduler/waiting"

    print("queues dir :",q_dir)    

    if not os.path.isdir(q_dir):
        try:
            os.makedirs(q_dir,0o775)
        except Exception as err:
            print('IERROR > creating "%s" directory failed, error "%s"'%(q_dir,err))
            exit(1)

    print("+"*50)


    id = 0
    for que in q_name_list:
        for t in q_name_list[que]:
						
            qname       = que              
            tname,res   = get_name_from_parameters(t)

            print(tname)
            fout        = open(q_dir + "/" + qname  + "_#_" + tname + ".json","w")
            id         +=1
			
            fout.write("{\n") 
            fout.write(" \"task\":\n") 

            send = "" 

            fout.write(send)   
            fout.write("    [\n") 
            lend = "" 
            for p in t:
                fout.write(lend)   
                for key in p:
                    if type(p[key]) == str:    
                        text = "            {\"%s\":\"%s\"}"%(key,p[key])
                    if type(p[key]) == int or type(p[key]) == float:    
                        text = "            {\"%s\":%s}"%(key,str(p[key]))
                    fout.write(text) 
                lend  = ",\n"
            fout.write("\n")   
                
            fout.write("    ]") 
            send = ",\n" 

            fout.write("\n") 
            fout.write("}\n") 
            fout.flush()
            fout.close()

    return(id)

#-----------------------------------------------------------

def cfg_gen_queues(main, cfg, base_path):

    s_param_list    = []
    p_param_list    = []
    u_param_list    = []

    errors          = 0

    s_param_list, p_param_list,errors  =  parse_consts(main, cfg)
    u_param_list              ,errors  =  parse_unroll(main, cfg)

    tasks       = unroll_loop(s_param_list,p_param_list,u_param_list,[])
    tasks       = unroll_tasks(tasks)
    queue_count = split_tasks(tasks,base_path)

    return(0,queue_count)

#-----------------------------------------------------------

def main():

    wr_waiting 	    = queue.Queue() 
    wr_passed 	    = queue.Queue() 
    wr_failed 	    = queue.Queue() 
    wr_inprogress 	= queue.Queue() 

    parser              = ArgumentParser()

    parser.add_argument("-cfg",    "--cfg" 	     	, dest="cfg"     	,help="json file with configs pool",	metavar="PATH",     required=True)
    parser.add_argument("-pDir",   "--project_dir"  , dest="pdir"    	,help="project directory",            	metavar="PATH",     required=False)
    parser.add_argument("-oDir",   "--output_dir"  	, dest="odir"    	,help="output directory",            	metavar="PATH",     required=True)
    parser.add_argument("-v",      "--verbose"    	, dest="verbose"	,help="verbose level",		                  	            required=False)

    args            	= parser.parse_args()

    #-------------------------------------------------------------------------------------------------

    root_dir            = "." if args.pdir is None else args.pdir
    root_path           = os.path.normpath(root_dir)

    cfg_dir  			= args.cfg
    cfg_path            = os.path.normpath(root_dir + "/" + cfg_dir)
    
    out_dir            	= args.odir
    out_path            = os.path.normpath(root_dir + "/" + out_dir)

    print("+"*50)
    print("current working dir  :",os.getcwd())
    print("root dir             :",root_path)
    print("cfg dir              :", cfg_path)
    print("out dir              :", out_path)
    print("+"*50)

    #-------------------------------------------------------------------------------------------------
        
    try:
        cfg_file	 	= open(cfg_path,"r");   
        print("opening training config file %s"%cfg_path)
    except:
        print("cannot open training config file %s"%cfg_path)
        exit(1)

    #-------------------------------------------------------------------------------------------------

    if not os.path.isdir(out_path):
        try:
            os.makedirs(out_path,0o775)
        except Exception as err:
            print('IERROR > creating "%s" directory failed, error "%s"'%(out_path,err))
            exit(1)

    #-------------------------------------------------------------------------------------------------
  
    cfg_json	 	    = json.load(cfg_file);     

    #-------------------------------------------------------------------------------------------------
    #loop section
    #-------------------------------------------------------------------------------------------------
    
    exe_list            = cfg_json["process"]
    base_path           = os.path.normpath(root_dir + "/" + out_dir)
    res = 0

    if not os.path.isdir(base_path):
        try:
            os.makedirs(base_path, 0o775)
        except Exception as err:
            print('ERROR   > creating "%s" directory failed, error "%s"'%(base_path, err))
            exit(1)
    
    gcnt = 0
    gerr = 0

    for exe in exe_list:

        err,cnt  = cfg_gen_queues(exe,cfg_json,base_path)

        gerr += err                 
        gcnt += cnt                 

    print("+"*50)

    if gerr!=0:
        print("configuration files error(s)")
        exit(1)
    else:
        print("%d configuration files generated successfully"%(gcnt))

    print("+"*50)

    return(0)

#-----------------------------------------------------------

if __name__ == '__main__':
    main()

