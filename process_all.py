import sys, os
from argparse          import ArgumentParser
from datetime import datetime
import time
from time import gmtime, strftime
import logging
from pandas.core.common import flatten

from v_utils.v_dataset import expand_session_dirs
from v_utils.v_arg import convert_dict_to_cmd_line_args, convert_cmd_line_args_to_dict, convert_cfg_files_to_dicts
from v_utils.v_arg import print_cfg_list, print_cfg_dict
#----------------------------------------------------------------------------
stsd={    
    0  :{"script":'process_st00.py'                 , "additional_params": None                                , "user_list_dir": 'as_input', "descr":"DICOM files processing"},
    1  :{"script":'process_st01.py'                 , "additional_params": None                                , "user_list_dir": 'as_data/st00_dicom_processing', "descr":"filter sessions"},
    2  :{"script":'process_st02.py'                 , "additional_params": None                                , "user_list_dir": 'as_data/st01_dicom_selecting', "descr":"use FlexNet to calculate ROI"},
    3  :{"script":'process_st03.py'                 , "additional_params": None                                , "user_list_dir": 'as_data/st02_roi', "descr":"cut to ROI"},
    5  :{"script":'process_st05.py'                 , "additional_params": None                                , "user_list_dir": 'as_data/st03_preprocessed_images',  "descr":"use FlexNet to segment all tissues"},
    6  :{"script":'process_st06.py'                 , "additional_params": None                                , "user_list_dir": 'as_data/st05_evaluated_shapes',  "descr":"images relative possition correction"},
    7  :{"script":'process_st07.py'                 , "additional_params": None                                , "user_list_dir": 'as_data/st06_shape_correction',  "descr":"CorrectShapes - Skin"},
    8  :{"script":'process_st08.py'                 , "additional_params": None                                , "user_list_dir": 'as_data/st07_postprocess',  "descr":"fill voids and convert images to polygons in json format"},
    23 :{"script":'process_st23.py'                 , "additional_params": None                                , "user_list_dir": 'as_data/st05_evaluated_shapes',  "descr":"create 3D objects from 2D polygons"},
    24 :{"script":'process_st24.py'                 , "additional_params": None                                , "user_list_dir": 'as_data/st23_preprocessed_meshes',  "descr":"remesh"},
 
}

#----------------------------------------------------------------------------
# initialize logging 
script_name = os.path.basename(__file__).split(".")[0]
log_format = "%(asctime)s [%(levelname)s] %(message)s"
time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
initial_log_fn = f"_initial_{script_name}_{time_str}_pid{os.getpid()}.log"
logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])

logging.info(f'*' * 50)
logging.info(f"script {script_name} start @ {time.ctime()}")
logging.info(f"initial log file is {initial_log_fn}")
logging.info(f"*" * 50)
logging.info(f"Parse command line arguments...")

from v_utils.v_logging_std import bind_std_2_logging
bind_std_2_logging()

start_time = time.time()
#----------------------------------------------------------------------------
logging.info("Reading configuration...")
parser = ArgumentParser()
logging.info(' -' * 25)
logging.info(" Command line arguments:\n  {}".format(' '.join(sys.argv)))
_logging_levels = logging._levelToName.keys()

cfa = parser.add_argument_group('config_file_arguments')
cfa.add_argument("--cfg" , default=[], action='append', type=str, nargs='*', required=False, metavar="PATH", help="one or more config json filenames. Further config files have higher priority. Command line arguments have the highest priority.", )
parser.add_argument("--logging_level"                   , default=logging.INFO                      , type=int          , required=False, choices=_logging_levels,     help="")

if not(("-h" in sys.argv) or ("--help" in sys.argv)):
    cfg_fns_args, cmd_line_args_rem = parser.parse_known_args(); # bez error gdy natrafi na nieznany parametr (odwrotnie niż "parse_args()")
    logging.getLogger().setLevel(cfg_fns_args.logging_level)
    cfgs = list(map(str, flatten(cfg_fns_args.cfg)))
    # read dictonaries from config files (create a list of dicts)
    cfg_dicts = convert_cfg_files_to_dicts(cfgs)

    # convert cmd_line_args_rem to dictionary so we can use it to update content of the dictonaries from config files
    cmd_line_args_rem_dict = convert_cmd_line_args_to_dict(cmd_line_args_rem)
    
    logging.info(' -' * 25)
    logging.info(" Merge config files arguments with command line arguments...")
    # merge all config dicts - later ones will overwrite entries with the same keys from the former ones
    cfg_dict_pr = {}
    for cfg_dict in cfg_dicts:
        cfg_dict_pr.update(cfg_dict)
    # finally update with the command line arguments dictionary
    cfg_dict_pr.update(cmd_line_args_rem_dict)
    
    logging.info(" Merged arguments:")
    cfg_d = cfg_dict_pr
    print_cfg_dict(cfg_d, indent = 1, skip_comments = True)

    # parse the merged dictionary
    args_list_to_parse = convert_dict_to_cmd_line_args(cfg_dict_pr)
#----------------------------------------------------------------------------
parser          = ArgumentParser()

parser.add_argument("-ses",    "--session_id"       , default = "*/*"   ,help="session id", nargs='+',	metavar="PATH",	required=False)
parser.add_argument("-sst",  "--start_at_stage",  type = int, default = list(stsd)[ 0] ,help="at which stage to start processing",    required=False)
parser.add_argument("-fst",  "--finish_at_stage", type = int, default = list(stsd)[-1] ,help="last stage to execute",    required=False)


logging.info('-' * 50)
if not(("-h" in sys.argv) or ("--help" in sys.argv)): 
    # get training arguments
    args, rem_args = parser.parse_known_args(args_list_to_parse)
    
    logging.info("Parsed configuration arguments:")
    args_d = vars(args)
    print_cfg_dict(args_d, indent = 1, skip_comments = True, max_print_len=8)

    if len(rem_args) > 0:
        logging.warning(f"Unrecognize arguments: {rem_args}")
    
else: 
    # help
    logging.info("Params:")
    logging.info(parser.format_help())
    logging.info("stages:")
    for std_key in stsd:
       logging.info(f"{std_key} : {stsd[std_key]['descr']}")
    sys.exit(1)


#----------------------------------------------------------------------------
def get_range_of_entries(d, first_key, last_key):
    result = {}
    found_first = False
    for key in d:
        if key == first_key:
            found_first = True
        if found_first:
            result[key] = d[key]
        if key == last_key:
            break
    return result
#----------------------------------------------------------------------------
stages_to_process_d = get_range_of_entries(stsd, args.start_at_stage, args.finish_at_stage)

if (not type(args.session_id) is list) and (not type(args.session_id) is tuple):
    args.session_id = [args.session_id]

for curr_stage_key, curr_stage_d in stages_to_process_d.items():
    for session in args.session_id:
        cmdl = f"python {curr_stage_d['script']} -ses {session}"
        if not curr_stage_d['additional_params'] is None:
            cmdl += f" {curr_stage_d['additional_params'] }"
        print(cmdl)
        ret = os.system(cmdl)
        if(ret != 0):
            logging.error(f"Error at stage {curr_stage_key} for session {session}, return value = {ret}. Exit...")
            break
