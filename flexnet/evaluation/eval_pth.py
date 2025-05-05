import sys, os
import pathlib
#-----------------------------------------------------------------------------------------
import os
import glob
import tracemalloc
import multiprocessing as mp
import timeit
import time
import tracemalloc
import multiprocessing
import timeit
import shutil
import logging
import copy
import time
import json
#---------------------------------------------------------
from argparse   import ArgumentParser
#-----------------------------------------------------------------------------------------
curr_script_path = os.path.dirname(os.path.abspath(__file__))
flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
from v_utils.v_json import jsonDumpSafe
from v_utils.v_dataset import MRIDataset
from v_utils.v_arg import print_cfg_list, print_cfg_dict
from flexnet.utils.gen_unet_utils import try_parse_dicts_from_file
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
    
#-----------------------------------------------------------

def modify_dsd(src_dsd, iDir, iSufix, keep_roi_iSufix, ses, oDir, f_reqref):

    logging.info("Modify the source dataset description dict...")
    timestamp = time.strftime("%y%m%d%H%M", time.gmtime())

    # modify entries as required:
    dst_dsd                                     = {}
            
    dst_dsd["session_dirs"]                     = [""]
    dst_dsd["skip_inputs_withouts_all_comps"]   = False
    dst_dsd["req_refs_level"]                   = f_reqref
    
    dst_dsd["ds_polyRefDirs_root"]          = os.path.normpath(os.path.join(src_dsd["ds_polyRefDirs_root"], ses))
    
    # find number of componets
    
    max_cmp_id                                  = MRIDataset.find_max_cmp_id(src_dsd)

    logging.info("Found {} component(s)".format(max_cmp_id+1))

    for cid in range(max_cmp_id+1):

         if iDir is not None:
             dst_dsd["cmp{}_imgDir_root".format(cid)]    = os.path.normpath((os.path.join(iDir, ses)))
         else:
             dst_dsd["cmp{}_imgDir_root".format(cid)]    = os.path.normpath((os.path.join(src_dsd["cmp{}_imgDir_root".format(cid)], ses)))

         if( keep_roi_iSufix and (src_dsd["cmp{}_imgDir_sufix".format(cid)] == 'roi')):
             logging.info("Keep roi input sufix for component {}".format(cid))
             dst_dsd["cmp{}_imgDir_sufix".format(cid)]    = 'roi'
         elif iSufix is None:
             dst_dsd["cmp{}_imgDir_sufix".format(cid)]    = ""
         else:
             dst_dsd["cmp{}_imgDir_sufix".format(cid)]    = iSufix
    
    return(dst_dsd)
    
#-----------------------------------------------------------

def use_unet(logging_level, ses, iDir, oDir, pth_path, iSufix, keep_roi_iSufix, shList, fgid, f_box, f_label, f_prob, f_prob_nl, f_mask, f_poly, f_stats, f_refreq, plim, f_fill_holes): 
    
    plpath 	= os.path.normpath('flexnet/evaluation/apply_model_to_session.py ')

    shp_list = ' '.join(shList)

    imgpath 	        = os.path.normpath(iDir)
    workpath 	        = os.path.normpath(oDir)
    
    #-----------------------------------------------------------
    #get reference database describsion dictionary that comes together with model .pth file

    logging.info("Use model from file {}".format(pth_path))

    model_dir, model_fn = os.path.split(pth_path)
    V                   = model_fn.split("_")[0]
    
    timestamp = time.strftime("%y%m%d%H%M", time.gmtime())
    
    #get reference database description dictionary that comes together with model .pth file
    model_timestamp     = model_fn.split('_')[0]
 
    logging.debug("Try reading file {} to find integrated dataset description dict...".format(pth_path))
    
    parsed_dicts        = try_parse_dicts_from_file(pth_path)
    
    logging.info("Found dicts: {}".format(parsed_dicts.keys()))

    if(not 'ds_dict' in parsed_dicts.keys()):
        logging.error("Not found integrated dataset dict!")
        sys.exit(1)

    #-----------------------------------------------------------
    # modyfikacja DSD
    
    dst_dsd             = modify_dsd(parsed_dicts["ds_dict"], iDir, iSufix, keep_roi_iSufix, ses, oDir,f_refreq)

    #-----------------------------------------------------------
    #zapis ds_dict do pliku

    #dst_dsd_fn                                  = "dst_{}_dsd_{}.json".format(timestamp,model_timestamp)
    dst_dsd_fn                                  = "dst_dsd_{}.json".format(model_timestamp)
    dst_dsd_path                                = os.path.normpath(os.path.join(oDir, dst_dsd_fn))

    logging.info("Saving required output files description to {}...".format(dst_dsd_path))
    jsonDumpSafe(dst_dsd_path, dst_dsd)     

    #-----------------------------------------------------------
    # copy loss information

    loss_path   = os.path.dirname(pth_path) + "/loss_log.json"

    for sh in shList:
        destdir     = os.path.normpath(os.path.join(oDir,sh))
        destination = os.path.normpath(os.path.join(oDir,sh,"loss_log.json"))

        try:
            if not os.path.isdir(destdir):
                pathlib.Path(destdir).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('Creating "%s" directory faild, error "%s"'%(destdir,err))
            exit(-1)

        try:
            shutil.copy(loss_path, destination) 
        except Exception as err:
            logging.warning('INFO      > skiping log_loss file')

    #-----------------------------------------------------------
    # construct cmd line

    dst_out = {
     "limit_polygons_num":   plim,
     "export_labels":        f_label,        
     "export_polygons":      f_poly,      
     "export_box":           f_box,           
     "export_prob":          f_prob, 
     "export_prob_nl":       f_prob_nl,
     "export_masks":         f_mask,
     "export_clasStats":     f_stats,
     "export_clasStatsPng":  "F",
     "export_dbg_raw":       "F",
     "export_dbg":           "F",
     "export_pngs_cropped":  "F",
     "fill_polygons_holes":  f_fill_holes,
     "fill_labels_holes":    f_fill_holes,
     "fill_masks_holes":     f_fill_holes,
     "req_refs_level":       f_refreq
    }
     
    dst_out_fn = "dst_out_{}.json".format(model_timestamp)
    dst_out_path = os.path.normpath(os.path.join(oDir, dst_out_fn))

    logging.info("Saving required output files description to {}...".format(dst_out_path))
    jsonDumpSafe(dst_out_path, dst_out)     
     
    #-----------------------------------------------------------
     
    cmd    = 'python {}'.format(plpath)
    cmd   +=  '--logging_level {}'.format(logging_level)
    cmd   += ' --out_dir {}'.format(os.path.join(oDir))
    cmd   += ' --model {}'.format(pth_path)
    cmd   += ' --force_gpu_id {} '.format(fgid)
    cmd   += ' --threshold_level 0.5 '
    if shList is not None:
        cmd   += ' --export_clss_filter_pass_list {}'.format(shp_list)
    cmd   += ' --cfg {} {}'.format(dst_dsd_path, dst_out_path)
 
    log_run ('use unet', os.path.join(oDir,ses))
    
    logging.info("Executing cmd: {}".format(cmd))
    ret = os.system(cmd)

    if ret:
        log_err ('roi shape unet', os.path.join(oDir,ses))

    return

#---------------------------------------------------------
# main
#---------------------------------------------------------

def main():

    parser          = ArgumentParser()

    parser.add_argument("-ses",  "--session_id"     , dest="ses_id"    ,help="session id",  	metavar="PATH",	required=False)
                                                                       
    parser.add_argument("-iDir", "--input_dir"      , dest="idir"      ,help="input dir",         			        required=True)
    parser.add_argument("-oShp", "--output_shapes"  , dest="oshp"      ,help="output shapes", nargs='+', type=str,  required=False)
    parser.add_argument("-oDir", "--output_dir"     , dest="odir"      ,help="output dir",         			        required=True)
    parser.add_argument("-iSfx", "--input_sufix"    , dest="isfx"      ,help="input sufix",      			        required=False)
    parser.add_argument("-kr_iSfx", "--keep_roi_input_sufix"           ,help="keep input sufix for roi",  default= True, required=False)
    parser.add_argument("-pth",  "--input_pth"      , dest="pth"       ,help="input pth",         			        required=False)
                                                                       
    parser.add_argument("-sB",   "--store_box"      , dest="f_box"     ,help="generate box file",          action='store_true',  required=False)
    parser.add_argument("-sL",   "--store_label"    , dest="f_label"   ,help="generate label file",        action='store_true',  required=False)
    parser.add_argument("-sP",   "--store_prob"     , dest="f_prob"    ,help="generate prob file",         action='store_true',  required=False)
    parser.add_argument("-sN",   "--store_prob_nl"  , dest="f_prob_nl" ,help="generate prob nLinear file", action='store_true',  required=False)
    parser.add_argument("-sM",   "--store_mask"     , dest="f_mask"    ,help="generate mask file",         action='store_true',  required=False)
    parser.add_argument("-sC",   "--store_polygons" , dest="f_poly"    ,help="generate polygons file",     action='store_true',  required=False)
    parser.add_argument("-sS",   "--store_stats"    , dest="f_stat"    ,help="generate stats file",        action='store_true',  required=False)
                                                                       
    parser.add_argument("-plim", "--polygons_lim"   , dest="plim"      ,help="limit of polygons number",                         required=False)
    parser.add_argument("-fH",   "--fill_holes"     , dest="fholes"    ,help="fill holes of polygons, masks and labels",action='store_true', required=False)
                                                    
    parser.add_argument("-refo", "--ref_only"       , dest="refo"      ,help="process images with refs only", action='store_true',         required=False)
                                                    
    parser.add_argument("-fgid", "--force_gpu_id"   , dest="fgid"      ,help="force gpu id",                       required=False)
                                                                       
    parser.add_argument("-v",    "--verbose"        , dest="verbose"   ,help="verbose level",						required=False)
                                                    
    args            = parser.parse_args()


    #---------------------------------------------------------------------------------------------------------------------------------------

    iDir        = os.path.normpath(args.idir) 
    iSfx        = "" if args.isfx is None else args.isfx
    ses_id      = "" if args.ses_id is None else args.ses_id
    oDir        = os.path.normpath(args.odir)
    ipth        = os.path.normpath(args.pth)
    
    fgid        = 0         if args.fgid == None else args.fgid

    plim        = 100       if args.plim == None else args.plim
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    # flagi

    f_box       = "T" if args.f_box     else "F"
    f_label     = "T" if args.f_label   else "F"
    f_prob      = "T" if args.f_prob    else "F"
    f_prob_nl   = "T" if args.f_prob_nl else "F"
    f_mask      = "T" if args.f_mask    else "F"
    f_poly      = "T" if args.f_poly    else "F"
    f_stats     = "T" if args.f_stat    else "F"
    fholes      = "T" if args.fholes    else "F"

    f_refreq    = "all" if args.refo    else "none"

    #---------------------------------------------------------------------------------------------------------------------------------------
    # initialize logging 

    data_dir = oDir
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

    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")

    #---------------------------------------------------------------------------------------------------------------------------------------
        
    logging.info('BEGIN     > -----------------------------------------------------')
    logging.info('INFO      > working set   : %s'%(ses_id))
    logging.info('INFO      > current dir   : %s'%os.getcwd())
    logging.info('INFO      > verbose level : %s'%args.verbose)
    logging.info('INFO      > subdirectory  :')
    logging.info('INFO      >    input images (png from dicom) : '+ iDir)
    logging.info('INFO      >    input pth                     : '+ ipth)
    logging.info('INFO      >    output shapes and metadata    : '+ oDir)
    
    try:
        if not os.path.isdir(oDir):
            pathlib.Path(oDir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('Creating "%s" directory faild, error "%s"'%(oDir,err))
        exit(-1)
    
    use_unet (logging_level, ses_id, iDir, oDir, ipth, iSfx, args.keep_roi_input_sufix, args.oshp, fgid, f_box, f_label, f_prob, f_prob_nl, f_mask, f_poly, f_stats, f_refreq, plim, fholes)

    #---------------------------------------------------------------------------------------------------------------------------------------

    logging.info("RUNS   : "+str(len(run_list)))
    logging.info("ERRORS : "+str(len(err_list)))
    if len(err_list):
        logging.info("LIST OF SESSIONS ENDED WITH ERRORS: ")
        for e in err_list:
            logging.info(e)

#-----------------------------------------------------------
    
if __name__ == '__main__':
    main()
