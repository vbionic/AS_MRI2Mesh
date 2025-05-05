import os
import pathlib
import sys 
import logging
import shutil
import copy
import json
#----------------------------------------------------------------------------
def print_cfg_dict(cfg_d, indent=0, skip_comments=False, max_print_len = 100):
    indent_str0 = " "*(indent-1 if indent>0 else 0)*3
    indent_str = " "*(indent)*3
    if(isinstance(cfg_d, dict)):
        #len = [cfg_d.keys()
        logging.info("{}{{".format(indent_str0))
        for key in cfg_d:
            if(skip_comments and (type(key) is str) and (key.find("_comment")!=-1)):
                continue
            val = cfg_d[key]
            if((type(val) is list) and (len(val) > 0) and (len(val)>1 or (type(val[0]) is list) or isinstance(val[0], dict))):
                logging.info("{}{}:".format(indent_str, key))
                print_cfg_list(val, indent+1, skip_comments, max_print_len)
            elif (isinstance(val, dict) and (len(val) > 0) and (len(val)>1 or (type(list(val.values())[0]) is list) or isinstance(list(val.values())[0], dict))):
                logging.info("{}{}:".format(indent_str, key))
                print_cfg_dict(val, indent+1, skip_comments, max_print_len)
            else:
                logging.info("{}{}:\t{}".format(indent_str,key,val))
        logging.info("{}}}".format(indent_str0))
        
#----------------------------------------------------------------------------
def print_cfg_list(cfg_l, indent=0, skip_comments=False, max_print_len = 100):
    indent_str0 = " "*(indent-1 if indent>0 else 0)*3
    indent_str = " "*(indent)*3
    logging.info("{}[".format(indent_str0))
    if len(cfg_l) > max_print_len:
        print_cfg_l = copy.copy(cfg_l[:max_print_len])
        print_cfg_l[-1] = "..."
    else:
        print_cfg_l = cfg_l
    for val in print_cfg_l:
        if((type(val) is list) and (len(val)>1 or (type(val[0]) is list) or isinstance(val[0], dict))):
            print_cfg_list(val, indent+1, skip_comments)
        elif (isinstance(val, dict) and (len(val)>1 or (type(list(val.values())[0]) is list) or isinstance(list(val.values())[0], dict))):
            print_cfg_dict(val, indent+1, skip_comments)
        else:
            logging.info("{}{}".format(indent_str, val))
    logging.info("{}]".format(indent_str0))

    
#----------------------------------------------------------------------------
from argparse import Action
class arg2boolAct(Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(arg2boolAct, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        val_conv = arg2bool(values)
        if val_conv is None:
            setattr(namespace, self.dest, None)
        else:
            bool_val = bool(val_conv)
            setattr(namespace, self.dest, bool_val)
#----------------------------------------------------------------------------
class arg2bool(object):
    def __bool__(self):
        return self.val
    __nonzero__=__bool__
    def __eq__(self, oth):
        return self.val == oth
    def __repr__(self):
        return self.val.__repr__()
    def __format__(self, format_spec):
        return format(str(self.val), format_spec)
        #return self.val.__format__()
    def __init__(self, input):
        found = False
        if(type(input) is str):
            if (input.lower().find("t") == 0) or (input.lower().find("y") == 0) or (input.lower().find("1") == 0):
                self.val = True
                found = True
            elif(input.lower().find("f") == 0) or (input.lower().find("n") == 0) or (input.lower().find("0") == 0):
                self.val = False
                found = True
        elif(type(input) is int):
            if(input == 0):
                self.val = False
                found = True
            elif(input == 1):
                self.val = True
                found = True
        elif(type(input) is bool):
                self.val = input
                found = True
        if not found:
            self.val = None

#----------------------------------------------------------------------------           
def convert_cfg_files_to_dicts(cfg_fn_list):
    cfg_dicts = []
    for cfg_fn in cfg_fn_list:
        logging.info(' -' * 25)
        logging.info(" Try reading config json file  {}:".format(cfg_fn))
        cfg_fn =  os.path.normpath(cfg_fn)
        if(os.path.isfile(cfg_fn) and (os.stat(cfg_fn).st_size != 0)):
            fconfig_json= open (cfg_fn)
            try:
                cfg_dict = json.load(fconfig_json)
            except json.JSONDecodeError as err:
                logging.error("Could not read from json config file: {}. \nError info: {}".format(cfg_fn, err))
                sys.exit(1)
        else:
            logging.error("Could not find dataset json config file: %s\n. Exit."%(cfg_fn))
            sys.exit(1)
    
        print_cfg_dict(cfg_dict, indent = 1, skip_comments = True)

        # skip comment entries (starting form "_" mark)
        key_list = list(cfg_dict.keys())
        for key in key_list:
            if(key.find('_') == 0):
                _ = cfg_dict.pop(key)

        cfg_dicts.append(cfg_dict)
    return cfg_dicts

#----------------------------------------------------------------------------
def convert_cmd_line_args_to_dict(cmd_line_args, recognize_negative_values = False):
    cmd_line_args_dict = {}
    key = None
    key_args = None 
    for k in cmd_line_args:
        is_negative_number = recognize_negative_values and (k.find("-") == 0) and (len(k) > 1) and k[1].isdigit()
        if (not (key is None)) and (k.find("-") == 0) and (not is_negative_number): # new key while parsing previous key - finish the previous key parsing
            cmd_line_args_dict[key] = key_args
            key = None
            key_args = None
        if(k.find("--") == 0):
            key = k[2:]
        elif(k.find("-") == 0) and (not is_negative_number):
            key = k[1:]
        else:
            if(key_args is None): # first argument
                key_args = k
            elif(not type(key_args) is list): # we already have one argument so a list mast be created
                key_args = [key_args, k]
            else: #append to arguments list
                key_args.append(k)
    if(not (key is None)):
        cmd_line_args_dict[key] = key_args
    return cmd_line_args_dict

#----------------------------------------------------------------------------
def convert_dict_to_cmd_line_args(cfg_dict, short_th = 3):
    cmd_line_args = []
    for key in cfg_dict.keys():
        if(len(key) > short_th):
            cmd_line_args.append("--{}".format(key))
        else:
            cmd_line_args.append("-{}".format(key))
        if(type(cfg_dict[key]) is list):
            for val in cfg_dict[key]:
                cmd_line_args.append("{}".format(val))
        elif(not cfg_dict[key] is None):
            cmd_line_args.append("{}".format(cfg_dict[key]))
    return cmd_line_args

#----------------------------------------------------------------------------