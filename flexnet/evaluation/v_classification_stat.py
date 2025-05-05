#!/usr/bin/env python
# coding: utf-8

# pip/conda libs:
import os,sys
import pathlib
import time
import numpy as np
import logging
import copy
from PIL import Image
import json
import multiprocessing
from multiprocessing import Process, Queue

# local libs:
#-----------------------------------------------------------------------------------------
curr_script_path = os.path.dirname(os.path.abspath(__file__))
flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
from v_utils.v_contour import *
from v_utils.v_polygons import *
from v_utils.v_dataset import MRIDataset, expand_session_dirs
#-----------------------------------------------------------------------------------------

def classification_stats_for_dataset(ds_ref_config_json_path,
                                     est_root_dir_path,
                                     work_dir = None, 
                                     out_fn = "class_stat.csv",
                                     mmpd = 0.5, margin_mm = 1.5, write_imgs = False,
                                     verbose = False):
     
    #----------------------------------------------------------------------------
    # initialize logging 
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    initial_log_fn = "_initial_training.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])

    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    logging.info("initial log file is {}".format(initial_log_fn))
    logging.info("*" * 50)
    logging.info(" ds_ref_config_json_path   : {}".format(ds_ref_config_json_path))
    logging.info(" est_root_dir_path         : {}".format(est_root_dir_path))
    if not(work_dir is None):
        logging.info(" work_dir                  : {}".format(work_dir))
    else:
        work_dir = est_root_dir_path
        logging.info(" work_dir                  : {}".format(work_dir))
    #----------------------------------------------------------------------------
    # create work dir
    work_dir = os.path.normpath(work_dir)
    try:
        if not os.path.isdir(work_dir):
            pathlib.Path(work_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
            logging.info("Created dir {}".format(work_dir))
    except Exception as err:
        logging.error("Creating dir ({}) IO error: {}".format(work_dir, err))
        sys.exit(1)
    #----------------------------------------------------------------------------
        
    logging.info("-"*30)
    logging.info("Opening csv file {} ...".format(out_fn))
    out_csv_path = os.path.join(work_dir, out_fn)
    out_csv_f = open(out_csv_path, "w") 
    logging.info("-"*30)
    #----------------------------------------------------------------------------
    # read dataset config
    
    dsd_pth =  os.path.normpath(ds_ref_config_json_path)
    logging.info("Reading ref dataset json config file: {}...".format(dsd_pth))
    if(os.path.isfile(dsd_pth) and (os.stat(dsd_pth).st_size != 0)):
        fconfig_json= open (dsd_pth)
        try:
            dataset_desc_ref = json.load(fconfig_json)
        except json.JSONDecodeError as err:
            logging.error("Could not read from json config file: {}. \nError info: {}".format(dsd_pth, err))
            sys.exit(1)
    else:
        logging.error("Could not find dataset json config file: %s\n. Exit."%(dsd_pth))
        sys.exit(1)
        
    #----------------------------------------------------------------------------
    logging.info("*" * 50)
    t_classes = dataset_desc_ref["ds_polygon_clss"]
    t_classes = list(set(t_classes)) # converting to set object leaves only the unique entries
    
    user_session_dir_l = dataset_desc_ref["session_dirs"]
    user_session_dir_l = list(set(user_session_dir_l)) # converting to set object leaves only the unique entries

    # expand "*" and "?" filters in user_session list entries
    root_dir = dataset_desc_ref["ds_polyRefDirs_root"]
    user_session_dir_l = expand_session_dirs(user_session_dir_l, root_dir)
    
    #----------------------------------------------------------------------------
    
    stat_dicts={}
    for user_ses_dir in user_session_dir_l:
        stat_dicts[user_ses_dir] = {}
        for t_class in t_classes:
            logging.info("user_session = {}, class = {}".format(user_ses_dir, t_class))
            stat_dicts[user_ses_dir][t_class]={}
            dataset_desc_ref["ds_polygon_clss"] = [t_class]
            dataset_desc_ref["session_dirs"]     = [user_ses_dir]
            dataset_desc_ref["comps"]            = [dataset_desc_ref["comps"][0]]
            dataset_desc_est = copy.copy(dataset_desc_ref)
            dataset_desc_est["ds_polyRefDirs_root"] = est_root_dir_path
            
            ds_ref = MRIDataset(dataset_desc_ref, force_size_is_multiple_of = 1, 
                                req_refs = 'all', logging_level=(logging.FATAL if not verbose else logging.INFO))
            ds_est = MRIDataset(dataset_desc_est, force_size_is_multiple_of = 1, 
                                req_refs = 'all', logging_level=(logging.FATAL if not verbose else logging.INFO))

            found = False
            if(len(ds_ref.paths) > 0 and len(ds_est.paths) > 0):
                polys_dicts = {}
                for ref in ds_ref.paths:
                    img_id = ref['src_image_id']
                    polys_dicts[img_id] = {"ref":ref['cls_envs_list'][0]['ref_polygon_path']}
                for est in ds_est.paths:
                    img_id = est['src_image_id']
                    if(img_id in polys_dicts.keys()):
                        polys_dicts[img_id]["est"] = est['cls_envs_list'][0]['ref_polygon_path']
                
                valid_poly_pairs = {}
                for img_id in polys_dicts.keys():
                    if("est" in polys_dicts[img_id].keys()):
                        valid_poly_pairs[img_id] = polys_dicts[img_id]

                if(len(valid_poly_pairs) != 0):
                    found = True
                    curr_dir = os.path.join(work_dir, user_ses_dir, t_class)

                    stats_dict = classification_stats_for_img_list(valid_poly_pairs, out_dir = curr_dir,  mmpd = mmpd, margin_mm = margin_mm, write_imgs = write_imgs)

                    logging.info(" Precision {:.3f}, TP {:.1f}, FP {:.1f}, req {:.1f}, pic_num {}".format(
                        stats_dict['Precision'], stats_dict['mm2_out_TP'], 
                        stats_dict['mm2_out_FP'], stats_dict['mm2_ref_req'], stats_dict['num_pic']))
                    stat_dicts[user_ses_dir][t_class]=stats_dict
            if not found:
                logging.info("-")
    
    # accumulate results
    for user_ses_dir in user_session_dir_l:
        stat_dicts[user_ses_dir]["avr"] = {}
        acc_stats = {}
        num_valid = 0
        for t_class in t_classes:
            if len(stat_dicts[user_ses_dir][t_class])!=0:
                num_valid += 1
                if(len(stat_dicts[user_ses_dir]["avr"]) == 0):
                    stat_dicts[user_ses_dir]["avr"] = copy.copy(stat_dicts[user_ses_dir][t_class])
                else:
                    for key in stat_dicts[user_ses_dir]["avr"].keys():
                        stat_dicts[user_ses_dir]["avr"][key] += stat_dicts[user_ses_dir][t_class][key]
        if num_valid != 0:
            stat_dicts[user_ses_dir]["avr"]['Precision'] /= num_valid
        
    #----------------------------------------------------------------------------
    params_to_dump = ['Precision', 'mm2_out_TP', 'mm2_out_FP', 'mm2_ref_req', 'num_pic']
    
    #----------------------------------------------------------------------------
    # write to csv
    for param in params_to_dump:
        out_csv_f.write(param + "\n")
        headers = ['session_sub_dir']
        for t_class in ["avr"] + t_classes:
            headers.append("{} {}".format(t_class, param))
        header_str = ""
        for header in headers:
            header_str += "{};".format(header)
        out_csv_f.write(header_str + "\n")

        for user_ses_dir in user_session_dir_l:
            out_csv_f.write("{};".format(user_ses_dir))
            for t_class in ["avr"] + t_classes:
                if len(stat_dicts[user_ses_dir][t_class])!=0:
                    out_csv_f.write("{};".format(stat_dicts[user_ses_dir][t_class][param]))
                else:
                    out_csv_f.write("-;")
            out_csv_f.write("\n")

def classification_stats_for_img_list(valid_poly_pairs, out_dir, mmpd = 0.5, margin_mm = 1.5, write_imgs = False):
     
    #----------------------------------------------------------------------------
    # create work dir
    work_dir = os.path.normpath(out_dir)
    try:
        if not os.path.isdir(work_dir):
            pathlib.Path(work_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
            logging.info("Created dir {}".format(work_dir))
    except Exception as err:
        logging.error("Creating dir ({}) IO error: {}".format(work_dir, err))
        sys.exit(1)
    #----------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------
    # read input files

    e_paths_dicts_valid = []
    for img_id in valid_poly_pairs.keys():
        r_path = valid_poly_pairs[img_id]["ref"]
        e_path = valid_poly_pairs[img_id]["est"]
        e_file_name = os.path.basename(e_path)
        file_name_pattern = os.path.splitext(e_file_name)[0]
        
        outJsonPath = os.path.normpath(os.path.join(work_dir, file_name_pattern+'clasStat.json'))
        outPngPath  = os.path.normpath(os.path.join(work_dir, file_name_pattern+'clasStat.png'))

        new_dict = {
            "img_id":img_id,
            "est_file_path":e_path,
            "est_file_name":e_file_name,
            "ref_file_path":r_path,
            "out_json_file_path":outJsonPath,
            "out_png_file_path":outPngPath,
            }
        e_paths_dicts_valid.append(new_dict)
        
    logging.debug("Found {} files.".format(len(e_paths_dicts_valid)))

    #----------------------------------------------------------------------------
    ## platform specific features
    #
    #system_name = os.name
    #if system_name != 'nt': #not Windows - require to specify that process start with spawn method
    #    multiprocessing.set_start_method('spawn')
    #
    #if 'pydevd_concurrency_analyser.pydevd_thread_wrappers' in sys.modules:
    #    logging.debug("Running in Visual Studio. Disable multiprocessing")
    #    use_multiprocessing = False
    #else:
    #    logging.debug("Running outside Visual Studio. Can use multiprocessing")
    #    use_multiprocessing = True#False
    
    #----------------------------------------------------------------------------
    # process 
    
    logging.debug('=' * 50)
    stat_dicts = []
    # evaluate on the whole dataset
    since = time.time()
    logging.debug("Start processing all inputs ...")
    use_multiprocessing = False
    if not use_multiprocessing:
        for d in e_paths_dicts_valid:

            est_polygons_org_pth = d["est_file_path"]
            ref_polygons_org_pth = d["ref_file_path"]
    
            logging.debug(" {} vs {}".format(os.path.basename(ref_polygons_org_pth), os.path.basename(est_polygons_org_pth)))

            # read reference polygons
            with open (ref_polygons_org_pth) as f:
                polygons_dict_data= json.load(f)
            ref_polygons_org    = v_polygons.from_dict(polygons_dict_data)

            # read estimated polygons
            with open (est_polygons_org_pth) as f:
                polygons_dict_data= json.load(f)
            est_polygons_org    = v_polygons.from_dict(polygons_dict_data)
        
            stat_dict, img = classification_stats_for_single_img(est_polygons_org, ref_polygons_org, mmpd, margin_mm, write_imgs)

            stat_dict["src_info"] = d 
            stat_dicts.append(stat_dict)

            img.save(d["out_png_file_path"])
            jsonDumpSafe(d["out_json_file_path"], stat_dict)

        logging.debug(" done")
         
    #----------------------------------------------------------------------------
    # process complete set of statistics
    stats_dict = {}

    mm2_ref_org      = 0
    mm2_ref_margin   = 0
    mm2_ref_req      = 0
    mm2_ref_allowed  = 0
    mm2_est_org      = 0
    mm2_out_TP       = 0
    mm2_out_TPo      = 0
    mm2_out_FP       = 0
    mm2_out_FPo      = 0
    mm2_out_FN       = 0
    mm2_out_FNo      = 0
    per_FN           = 0
    per_TP           = 0
    per_TP_no_margin = 0
    per_req          = 0

    for d in stat_dicts:
        mm2_ref_org      += d["mm2_ref_org"     ]
        mm2_ref_margin   += d["mm2_ref_margin"  ]
        mm2_ref_req      += d["mm2_ref_req"     ]
        mm2_ref_allowed  += d["mm2_ref_allowed" ]
        mm2_est_org      += d["mm2_est_org"     ]
        mm2_out_TP       += d["mm2_out_TP"      ]
        mm2_out_TPo      += d["mm2_out_TPo"     ]
        mm2_out_FP       += d["mm2_out_FP"      ]
        mm2_out_FPo      += d["mm2_out_FPo"     ]
        mm2_out_FN       += d["mm2_out_FN"      ]
        mm2_out_FNo      += d["mm2_out_FNo"     ]

    pic_num = len(stat_dicts) 
        
    #statystyki potrzebne do wyliczenia Precision okreslonej w projekcie:  
    stats_dict["-----MAIN STATS"            ] = "-----"     
    stats_dict["mm2_out_TP"                 ] = mm2_out_TP  
    stats_dict["mm2_out_FP"                 ] = mm2_out_FP   
    stats_dict["mm2_ref_req"                ] = mm2_ref_req     
    stats_dict["Precision"                  ] = mm2_out_TP   / (mm2_out_TP + mm2_out_FP) if (mm2_out_TP + mm2_out_FP)!=0 else 1
    
    stats_dict["-----AREA [mm2]"            ] = "-----"     
    #inne statystyki:
    stats_dict["mm2_ref_org"                ] = mm2_ref_org     
    stats_dict["mm2_ref_margin"             ] = mm2_ref_margin  
    stats_dict["mm2_ref_allowed"            ] = mm2_ref_allowed 
    stats_dict["mm2_est_org"                ] = mm2_est_org       
    stats_dict["mm2_out_TPo"                ] = mm2_out_TPo 
    stats_dict["mm2_out_FPo"                ] = mm2_out_FPo      
    stats_dict["mm2_out_FN"                 ] = mm2_out_FN   
    stats_dict["mm2_out_FNo"                ] = mm2_out_FNo 
    
    stats_dict["----AVERAGES "              ] = "-----"  
    stats_dict["num_pic"                    ]  = pic_num

    stats_dict["----AVERAGES [mm2]"         ] = "-----"  
    stats_dict["aver_area_per_FN"           ]  = mm2_out_FN   / mm2_ref_req if mm2_ref_req!=0 else -1
    stats_dict["aver_area_per_TP"           ]  = mm2_out_TP   / mm2_ref_req if mm2_ref_req!=0 else -1
    stats_dict["aver_area_per_TP_no_margin" ]  = mm2_out_TPo  / mm2_ref_org if mm2_ref_org!=0 else -1
    stats_dict["aver_area_per_req"          ]  = mm2_ref_req  / mm2_ref_org if mm2_ref_org!=0 else -1
    
    stats_dict["----AVERAGES [%]"           ] = "-----"  
    stats_dict["aver_image_per_FN"          ] = per_FN            /pic_num
    stats_dict["aver_image_per_TP"          ] = per_TP            /pic_num
    stats_dict["aver_image_per_TP_no_margin"] = per_TP_no_margin  /pic_num
    stats_dict["aver_image_per_req"         ] = per_req           /pic_num

    outJsonPath = os.path.normpath(os.path.join(work_dir, 'clasStat.json'))
    jsonDumpSafe(outJsonPath, stats_dict)
    
    #----------------------------------------------------------------------------
    time_elapsed = time.time() - since
    logging.debug(' Processing time {:2.0f}m:{:02.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
    
    return stats_dict


def classification_stats_for_single_img(est_polygons_org, ref_polygons_org, mmpd = 0.5, margin_mm = 1.5, write_imgs = False):
    
    margin_points = margin_mm/mmpd/2.0
    margin_points_ceil = np.int16(np.ceil(margin_points))
    points_2_area = 1.0 * mmpd * mmpd
    
    # determine roi
    org_box = ref_polygons_org["box"]
    est_box = est_polygons_org["box"]
    
    if(len(est_box) == 0) and (len(org_box) == 0):
        org_box = [0,0,1,1]
        est_box = copy.copy(org_box)
    elif(len(org_box) == 0):
        org_box = copy.copy(est_box)
    elif(len(est_box) == 0):
        est_box = copy.copy(org_box)
        
    roi_point = np.array([min([est_box[0], org_box[0]]) - margin_points_ceil - 1, min([est_box[1], org_box[1]]) - margin_points_ceil - 1], dtype=np.int16)
    roi_end   = np.array([max([est_box[2], org_box[2]]) + margin_points_ceil + 2, max([est_box[3], org_box[3]]) + margin_points_ceil + 2], dtype=np.int16)  

    roi_size  = roi_end - roi_point

    # crop to roi
    ref_polygons_org.move2point(-roi_point)
    est_polygons_org.move2point(-roi_point)

    # create polygons for estimated and reference
    est_polygons_org_np     = est_polygons_org.as_numpy_mask(fill=True , w=roi_size[0], h=roi_size[1], val=255, masks_merge_type='or')
    est_polygons_org_peri_np= est_polygons_org.as_numpy_mask(fill=False, w=roi_size[0], h=roi_size[1], val=255, masks_merge_type='or', line_type = "-")
        
    ref_polygons_org_np     = ref_polygons_org.as_numpy_mask(fill=True , w=roi_size[0], h=roi_size[1], val=255, masks_merge_type='or')
    ref_polygons_org_peri_np= ref_polygons_org.as_numpy_mask(fill=False, w=roi_size[0], h=roi_size[1], val=255, masks_merge_type='or')

    # create polygons at a margin of the reference polygons - in that area we DO NOT require match between estimated and reference results
    ref_polygons_margin    = v_polygons.from_polygons_borders(ref_polygons_org, dilation_radius = margin_points)
    ref_polygons_margin_np = ref_polygons_margin.as_numpy_mask(fill=True, w=roi_size[0], h=roi_size[1], val=128, masks_merge_type='or')
        
    # create req polygons from the reference but without margin area - in req polygon area we DO require match between estimated and reference results
    ref_polygons_req_np = np.where((ref_polygons_margin_np == 0), ref_polygons_org_np, 0)
        
    # create allowed polygons from the reference with margin area - in allowed polygon area we DO allow nonzero values of the estimated mask 
    ref_polygons_allowed_np = ref_polygons_org_np | ref_polygons_margin_np

    # create (req & est) polygons - true-positive polygons
    out_polygons_TP_np = np.where((ref_polygons_req_np != 0) & (est_polygons_org_np != 0), np.uint8(128), np.uint8(0))

    # create (org & est) polygons - true-positive polygons if no margin was allowed
    out_polygons_TPo_np = np.where((ref_polygons_org_np != 0) & (est_polygons_org_np != 0), np.uint8(255), np.uint8(0))
        
    # create (!allowed & est) polygons - false-positive polygons
    out_polygons_FP_np = np.where((ref_polygons_allowed_np == 0) & (est_polygons_org_np != 0), np.uint8(180), np.uint8(0))
        
    # create (!org & est) polygons - false-positive polygons if no margin was allowed
    out_polygons_FPo_np = np.where((ref_polygons_org_np == 0) & (est_polygons_org_np != 0), np.uint8(255), np.uint8(0))

    # create (req & !est) polygons - false-negative polygons
    out_polygons_FN_np = np.where((ref_polygons_req_np != 0) & (est_polygons_org_np == 0), np.uint8(100), np.uint8(0))

    #calc area of the polygons

    area_p_ref_org     = np.count_nonzero(ref_polygons_org_np      ) * points_2_area
    area_p_ref_margin  = np.count_nonzero(ref_polygons_margin_np   ) * points_2_area
    area_p_ref_req     = np.count_nonzero(ref_polygons_req_np      ) * points_2_area
    area_p_ref_allowed = np.count_nonzero(ref_polygons_allowed_np  ) * points_2_area
    area_p_est_org     = np.count_nonzero(est_polygons_org_np      ) * points_2_area
    area_p_out_TP      = np.count_nonzero(out_polygons_TP_np       ) * points_2_area
    area_p_out_TPo     = np.count_nonzero(out_polygons_TPo_np      ) * points_2_area
    area_p_out_FP      = np.count_nonzero(out_polygons_FP_np       ) * points_2_area
    area_p_out_FPo     = np.count_nonzero(out_polygons_FPo_np      ) * points_2_area
    area_p_out_FN      = area_p_ref_req - area_p_out_TP
    area_p_out_FNo     = area_p_ref_org - area_p_out_TPo

    out_dict = {
        "mm2_ref_org"     : area_p_ref_org    ,
        "mm2_ref_margin"  : area_p_ref_margin ,
        "mm2_ref_req"     : area_p_ref_req   ,
        "mm2_ref_allowed" : area_p_ref_allowed,
        "mm2_est_org"     : area_p_est_org    ,
        "mm2_out_TP"      : area_p_out_TP     ,
        "mm2_out_TPo"     : area_p_out_TPo    ,
        "mm2_out_FP"      : area_p_out_FP     ,
        "mm2_out_FPo"     : area_p_out_FPo    ,
        "mm2_out_FN"      : area_p_out_FN     ,
        "mm2_out_FNo"     : area_p_out_FNo    ,

        "per_FN"          : (area_p_out_FN  / area_p_ref_req) if (area_p_ref_req>0) else 0   ,
        "per_TP"          : (area_p_out_TP  / area_p_ref_req) if (area_p_ref_req>0) else 0   ,
        "per_TP_no_margin": (area_p_out_TPo / area_p_ref_org) if (area_p_ref_org>0) else 0   ,
        "per_req"         : (area_p_ref_req / area_p_ref_org) if (area_p_ref_org>0) else 0   

        }

    stats_Image = None
    if(write_imgs):
        r = out_polygons_FN_np | out_polygons_FP_np #light red - FP, strong red - FN
        g = out_polygons_TP_np # green - match with req
        #b = np.where(ref_polygons_org_peri_np!=0, np.uint8(255), ref_polygons_margin_np) #light blue-allowed, strong blue - req
        b = ref_polygons_org_peri_np | ref_polygons_margin_np; #light blue-allowed, strong blue - req
        #b = np.where(r|g, np.uint8(0), b) #light blue-allowed, strong blue - req
        b = np.where(est_polygons_org_peri_np, np.uint8(200), b) # dashed blue estimated polygon perimeter
        rgb = np.dstack([r, g, b])  # stacks 3 h x w arrays -> h x w x 3
        stats_Image = Image.fromarray(rgb)

    return out_dict, stats_Image

def classification_stats_for_single_img_np(pre_polys_np, ref_polys_np, mmpd = 0.5, write_imgs = False):
    
    points_2_area = 1.0 * mmpd * mmpd
    w = pre_polys_np.shape[1]
    h = pre_polys_np.shape[0]
    
    # create (org & est) polygons - true-positive polygons if no margin was allowed
    TP_polys_np = np.where((ref_polys_np != 0) & (pre_polys_np != 0), np.uint8(190), np.uint8(0))
               
    #calc area of the polygons
    
    area_tot = w * h                                * points_2_area
    area_ref = np.count_nonzero(ref_polys_np      ) * points_2_area
    area_pre = np.count_nonzero(pre_polys_np      ) * points_2_area
    area_TP  = np.count_nonzero(TP_polys_np       ) * points_2_area
    area_FP  = area_pre - area_TP
    area_FN  = area_ref - area_TP
    area_TN  = area_tot - area_TP - area_FP - area_FN

    PR = area_TP / (area_TP + area_FP) if(area_TP != 0) else 0.0 if(area_FP != 0) else 1.0
    RC = area_TP / (area_TP + area_FN) if(area_TP != 0) else 0.0 if(area_FN != 0) else 1.0
    AC = (area_TP + area_TN) / area_tot
    F1 = (2 * (PR * RC) / (PR + RC)) if ((PR + RC) != 0.0) else 0.0
    out_dict = {
        "tot" : area_tot  ,
        "ref" : area_ref,
        "pre" : area_pre,
        "TP"  : area_TP ,
        "FP"  : area_FP ,
        "FN"  : area_FN ,
        "TN"  : area_TN ,
        "PR"  : PR,
        "RC"  : RC,
        "F1"  : F1,
        "AC"  : AC
        }

    stats_Image = None
    if(write_imgs):
        FP_polys_np = np.where((ref_polys_np == 0) & (pre_polys_np != 0), np.uint8( 90), np.uint8(0))
        FN_polys_np = np.where((ref_polys_np != 0) & (pre_polys_np == 0), np.uint8(190), np.uint8(0))
        g = TP_polys_np                 # TP         - green
        r = FP_polys_np | FN_polys_np   # FP - light red, FN - strong red
        b = np.zeros(TP_polys_np.shape, dtype = np.uint8)
        rgb = np.dstack([r, g, b])  # stacks 3 h x w arrays -> h x w x 3
        stats_Image = Image.fromarray(rgb)

    return out_dict, stats_Image
        
def main(mmpd = 0.5, margin_mm = 1.5, write_imgs = True):
    
    from v_utils.v_json import jsonUpdate, jsonDumpSafe
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    #path = "as_data/st04_shape_processing/Bartek/00000014_forearm/skin"
    #est_polygons_org_fn = os.path.normpath(os.path.join(path,'00000148_nsi_skin_polygons_unetX.json'))
    #ref_polygons_org_fn = os.path.normpath(os.path.join(path,'00000148_skin_polygons.json'))
    
    path = "as_data/st04_shape_processing/Jakub/00000010_forearm/skin"
    est_polygons_org_fn = os.path.normpath(os.path.join(path,'00000072_nsi_skin_polygon_unetXs.json'))
    ref_polygons_org_fn = os.path.normpath(os.path.join(path,'00000072_skin_polygons.json'))
    
    # read estimated polygons
    logging.info("read estimated polygons from json file {}".format(est_polygons_org_fn))
    with open (est_polygons_org_fn) as f:
        polygons_dict_data= json.load(f)
    est_polygons    = v_polygons.from_dict(polygons_dict_data)
        
    # read reference polygons
    logging.info("read reference polygons from json file {}".format(ref_polygons_org_fn))
    with open (ref_polygons_org_fn) as f:
        polygons_dict_data= json.load(f)
    ref_polygons    = v_polygons.from_dict(polygons_dict_data)

    stats_dict, img = classification_stats_for_single_img(est_polygons, ref_polygons, mmpd, margin_mm, write_imgs)
    
    img.save("as_classification_stat_out.png")
    jsonDumpSafe("as_classification_stat_out.json", stats_dict)

if __name__ == '__main__':
    main()