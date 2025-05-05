#!/usr/bin/env python
# coding: utf-8

# pip/conda libs:
import os,sys
import time
import numpy as np
import logging
import torch
import copy
from PIL import Image
import json
from argparse import ArgumentParser

# local libs:
#-----------------------------------------------------------------------------------------
curr_script_path = os.path.dirname(os.path.abspath(__file__))
flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
from v_utils.v_contour import *
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe
from flexnet.evaluation.v_classification_stat import classification_stats_for_single_img, classification_stats_for_dataset
from v_utils.v_dataset import MRIDataset
#----------------------------------------------------------------------------
# main
from multiprocessing import Process
import multiprocessing
import subprocess

def main():
    
    parser = ArgumentParser()
    
    parser.add_argument("-eD",   "--est_dir",      dest="est_dir",    help="estimated polygons root directory",          metavar="PATH", required=False)
    parser.add_argument("-dsd",  "--ref_dsd",      dest="ref_dsd",    help="dataset description for reference polygons", metavar="PATH", required=False)
    parser.add_argument("-oD",   "--out_dir",      dest="out_dir",    help="output statistics directory",                metavar="PATH", required=False)
    parser.add_argument("-mmpd", "--mm_per_dot",   dest="mmpd",       help="number of image dots / pixels per mm",                   required=False)
    parser.add_argument("-mar",  "--margin_mm",    dest="margin_mm",  help="width of expert margin error on class border",          required=False)
    parser.add_argument("-v",    "--verbose",      dest="verbose",    help="verbose level",   action='store_true',                   required=False)


    args = parser.parse_args()
    
    est_dir_path            = args.est_dir          if not (args.est_dir   is None) else "as_data/st06_shape_processing"
    ds_ref_config_json_path = args.ref_dsd          if not (args.ref_dsd   is None) else "as_unet.cfg/dataset_snapshot_fully_done_config.json"
    out_dir                 = args.out_dir          if not (args.out_dir   is None) else est_dir_path + "_class_stats"
    mmpd                    = args.mmpd             if not (args.mmpd      is None) else 0.5  
    margin_mm               = float(args.margin_mm) if not (args.margin_mm is None) else 1.5 
    write_imgs              = True
    verbose = args.verbose

    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    since = time.time()

    #----------------------------------------------------------------------------
    # actual calculations
    classification_stats_for_dataset(ds_ref_config_json_path, est_dir_path, work_dir = out_dir,
                                    mmpd = mmpd, margin_mm = margin_mm, write_imgs = write_imgs,
                                    verbose = verbose)
    #----------------------------------------------------------------------------
    time_elapsed = time.time() - since
    logging.info(' Processing time {:2.0f}m:{:02.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

    #----------------------------------------------------------------------------
    # final EVALUATION
    logging.info("script ends @ {}".format(time.ctime()))
    logging.info('*' * 50)

    #----------------------------------------------------------------------------


if __name__ == '__main__':
    #----------------------------------------------------------------------------
    # initialize logging 
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    initial_log_fn = "_calc_accuracy_for_session.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()

    
main()
