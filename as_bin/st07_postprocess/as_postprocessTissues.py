import cv2 as cv
from matplotlib import pyplot as plt
import sys, getopt
import pydicom
import numpy as np
from PIL import Image
import json
from pydicom.tag import Tag
from pydicom.datadict import keyword_for_tag
import os
import pathlib
from argparse import ArgumentParser
import glob
import math
import shutil
import logging
import scipy
from scipy.interpolate import RectBivariateSpline
import time

#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_json import *
from v_utils.v_polygons import *
#-----------------------------------------------------------------------------------------

def read_files(iDir, ROI_file, tissue, verbose):
    tissue_file = os.path.normpath(iDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_labels.png')
    if verbose:
        print(tissue_file)
    #logging.info("opening {} file: {}".format(tissue, tissue_file))
    tissue_raw = cv.imread(tissue_file, cv.IMREAD_COLOR)
    tissue_raw = tissue_raw[:,:,2]
    tissue_raw[tissue_raw != 0] = 255
    
    tissue_poly_file = os.path.normpath(iDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_polygons.json')
    #logging.info("opening {} prob file: {}".format(tissue, tissue_prob_file))
    
    tissue_poly_h = open(tissue_poly_file)
    tissue_poly = json.load(tissue_poly_h)
    tissue_poly_h.close()

    return [tissue_raw, tissue_poly]
    
def wirte_files(oDir, ROI_file, tissue, labels, poly):
    out_path = os.path.normpath(oDir + "/"+tissue)
    out_file = os.path.normpath(oDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_labels.png')
    out_poly_file = os.path.normpath(oDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_polygons.json')
    try:
        if not os.path.isdir(out_path):
            pathlib.Path(out_path).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(out_path,err))
        exit(1)
    cv.imwrite(out_file, labels)
    jsonDumpSafe(out_poly_file, poly)
    
def process_tissue(raw_in, poly_in, configuration, tissue, verbose):
    max_poly_num = configuration[tissue]["max_polygon_num"]
    min_poly_size = configuration[tissue]["min_polygon_size"]
    holes_allowed = configuration[tissue]["holes_allowed"]
    processed_poly = {"polygons": []}
    number = 0
    for poly in poly_in["polygons"]:
        if verbose:
            print(poly["area"])
        if (poly["area"] >= min_poly_size) and (number < max_poly_num):
            processed_poly["polygons"].append(poly)
            if not holes_allowed:
                processed_poly["polygons"][-1]["inners"]=[]
        number += 1
    
    processed_poly["box"] = poly_in["box"]
    
    if verbose:
        print(processed_poly)
    
    tissue_polygons_out = v_polygons()
    tissue_polygons_out = v_polygons.from_dict(processed_poly)
    raw_out = tissue_polygons_out.as_numpy_mask(fill = True,w = raw_in.shape[1], h = raw_in.shape[0])
    return [raw_out, tissue_polygons_out]
        
def process_dir(iDir, oDir, log2, configuration, verbose):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    file_list    = glob.glob(os.path.normpath(iDir + "/roi/*_roi_labels.png"))
    file_list.sort()
    
    index = []
    coordinate_mm = []
    number = 1
    results = []
    corrections = []
    
    for ROI_file in file_list:
        #logging.info("opening ROI file: {}".format(ROI_file))
        ROI_raw = cv.imread(ROI_file, cv.IMREAD_COLOR)
        ROI_raw = ROI_raw[:,:,2]

        try:
            if not os.path.isdir(oDir + "/roi"):
                pathlib.Path(oDir + "/roi").mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('creating "%s" directory failed, error "%s"'%(oDir + "/roi",err))
            exit(1)
        cv.imwrite(oDir + "/roi/" + os.path.basename(ROI_file), ROI_raw)
        
        #for tissue in ["bones", "fat", "muscles", "skin", "vessels"]:
        occupied_by_tissues = np.zeros(ROI_raw.shape, dtype = ROI_raw.dtype)
        for tissue in configuration["tissue_order"]:
            [raw_in, poly_in] = read_files(iDir, ROI_file, tissue, verbose)
            if tissue != "skin":
                raw_in = cv.bitwise_and(raw_in, ROI_raw)
            
            #print("{}:{}  {}:{}".format(occupied_by_tissues.shape, occupied_by_tissues.dtype, raw_in.shape, raw_in.dtype))
            
            raw_in = cv.bitwise_and(raw_in, cv.bitwise_not(occupied_by_tissues))
            occupied_by_tissues = cv.bitwise_or(occupied_by_tissues, raw_in)
            #print(poly_in)
            #print(json.dumps(poly_in, indent=4))
            tissue_polygons_in = v_polygons()
            tissue_polygons_in._mask_ndarray_to_polygons(raw_in, background_val = 0, limit_polygons_num = 0)
            tissue_polygons_in_dict = tissue_polygons_in.as_dict()
            #print(json.dumps(tissue_polygons_in_dict, indent=4))
            
            [raw_out, poly_out] = process_tissue(raw_in, tissue_polygons_in_dict, configuration, tissue, verbose)
            image_out = np.zeros((raw_in.shape[0], raw_in.shape[1], 3))
            image_out[:,:,2] = raw_out
            wirte_files(oDir, ROI_file, tissue, image_out, poly_out.as_dict())


parser = ArgumentParser()

parser.add_argument("-iDir",      "--input_dir"      ,     dest="idir"   ,    help="input directory" ,    metavar="PATH", required=True)
parser.add_argument("-oDir",      "--output_dir"     ,     dest="odir"   ,    help="output directory",    metavar="PATH", required=True)
parser.add_argument("-conf",      "--configuration"  ,     dest="conffn" ,    help="configuration file name",    metavar="PATH", required=True)

parser.add_argument("-v"   ,      "--verbose"        ,     dest="verbose",    help="verbose level"   ,                    required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
iDir 	= args.idir
oDir  	= args.odir
conffn  = args.conffn

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(oDir+"/fillVoids.log",mode='w'),logging.StreamHandler(sys.stdout)])


if not os.path.isdir(iDir):
    logging.error('Error : Input directory (%s) not found !',iDir)
    exit(1)


logging.info("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_postprocessTissues.py")
logging.info("in:     "   +   iDir)
logging.info("out:    "   +   oDir)
logging.info("config: "   +   conffn)
logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")


log2 = open(oDir+"/postprocessTissues_results.log","a+")

if verbose == 'off':
    verbose = False
else:
    verbose = True

try:
    conffh = open(conffn)
    configuration = json.load(conffh)
    conffh.close()
except Exception as err:
    logging.error("Input data IO error: {}".format(err))
    sys.exit(1)


process_dir(iDir, oDir, log2, configuration, verbose)

log2.close()