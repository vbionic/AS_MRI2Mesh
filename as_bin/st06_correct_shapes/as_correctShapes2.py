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


def process_dir(iDir, oDir, log2, verbose):
    polygon_list    = glob.glob(iDir + "/*_polygons.json")
    polygon_list.sort()
    
    for polygon_file in polygon_list:
        try:
            file_data = open(polygon_file)
            data = json.load(file_data)
            file_data.close()
        except Exception as err:
            logging.error("Input data IO error: {}".format(err))
            sys.exit(1)
        
        skinOut_fn = oDir + "/" + os.path.split(polygon_file)[1].rsplit('_',1)[0]+"_labels.png"
        
        
        skin_fn = polygon_file.rsplit('_',1)[0]+"_labels.png"
        #print(oDir)
        #print(os.path.split(polygon_file)[1])
        
        logging.info("opening file: {}".format(skin_fn))
        skin_image_raw = cv.imread(skin_fn, cv.IMREAD_COLOR)
        skin_image_raw = skin_image_raw[:,:,2]
        skin_image_raw[skin_image_raw != 255] = 0

        
        try:
            [dir_name,name] = os.path.split(polygon_file)
            roi_fn = dir_name + "/../roi/" + name.split('_',1)[0]+"_roi_labels.png"
            roi_image_raw = cv.imread(roi_fn, cv.IMREAD_COLOR)
            roi_image_raw = roi_image_raw[:,:,2]
        except Exception as err:
            logging.error("ROI file error")
            sys.exit(1)

        if (roi_image_raw.shape != skin_image_raw.shape):
            logging.info("images are not the same size")
            skin_image = np.zeros((max([skin_image_raw.shape[0], roi_image_raw.shape[0]]), max([skin_image_raw.shape[1], roi_image_raw.shape[1]]),3))
            roi_image = np.zeros((max([skin_image_raw.shape[0], roi_image_raw.shape[0]]), max([skin_image_raw.shape[1], roi_image_raw.shape[1]]),3))
            for j in range(0,skin_image_raw.shape[0]):
                for i in range(0,skin_image_raw.shape[1]):
                    skin_image[j,i,:] = skin_image_raw[j,i,:]
            for j in range(0,roi_image_raw.shape[0]):
                for i in range(0,roi_image_raw.shape[1]):
                    roi_image[j,i,:] = roi_image_raw[j,i,:]
        else:
            skin_image = skin_image_raw
            roi_image = roi_image_raw
        if verbose:
            cv.imwrite(skinOut_fn+"_00_original_skin.png",skin_image.astype(np.uint8))
            
        if(len(data["polygons"])==0):
            jsonDumpSafe(skinOut_fn.rsplit('_',1)[0]+"_polygons.json", data)
            logging.info("writing file: {}".format(skinOut_fn))
            skin_image = cv.cvtColor(skin_image, cv.COLOR_GRAY2BGRA)
            for j in range(0,skin_image.shape[0]):
                for i in range(0,skin_image.shape[1]):
                    if ((skin_image[j,i,0]>0) | (skin_image[j,i,1]>0) | (skin_image[j,i,2]>0)):
                        skin_image[j,i,0] = skin_image_copy[j,i]
                        skin_image[j,i,1] = 0
                        skin_image[j,i,2] = 255
                        skin_image[j,i,3] = 255
                    else:
                        skin_image[j,i,3] = 0
            cv.imwrite(skinOut_fn,skin_image.astype(np.uint8))
            
            a = cv.imread(polygon_file.rsplit('_',1)[0]+"_prob.png")
            cv.imwrite(oDir + "/" + os.path.split(polygon_file)[1].rsplit('_',1)[0]+"_prob.png",a)
            
            a = cv.imread(polygon_file.rsplit('_',1)[0]+"_prob_nl.png")
            cv.imwrite(oDir + "/" + os.path.split(polygon_file)[1].rsplit('_',1)[0]+"_prob_nl.png",a)
            
            continue
        
        skin_image_copy = skin_image.copy()
        
        #filling holes in ROI
        roi_image_copy = roi_image.copy()
        cv.floodFill(roi_image_copy, None, (0,0),255)
        
        #cv.imwrite(skinOut_fn+"test1.png",roi_image_copy.astype(np.uint8))
        roi_image_copy = cv.bitwise_not(roi_image_copy)
        #cv.imwrite(skinOut_fn+"test2.png",roi_image_copy.astype(np.uint8))
        roi_image += roi_image_copy
        #cv.imwrite(skinOut_fn+"_01_ROI_filled.png",roi_image.astype(np.uint8))
        
        kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(3, 3))
        skin_image  = cv.morphologyEx(skin_image, cv.MORPH_OPEN, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        if verbose:
            cv.imwrite(skinOut_fn+"_02_skin_Open.png",skin_image.astype(np.uint8))
        
        kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3, 3))
        morph1  = cv.morphologyEx(roi_image, cv.MORPH_DILATE, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        wynik   = cv.bitwise_or(cv.bitwise_xor(morph1, roi_image), skin_image)
        if verbose:
            cv.imwrite(skinOut_fn+"_03_skin_with_added_lines.png",wynik.astype(np.uint8))
        #wynik = skin with at least 1 pixel thickness
        
        #------------------------------------
        #find areas with 1 pixel thickness
        #------------------------------------
        
        
        
        wynik_copy = wynik.copy()
        
        # #remove all thin lines
        # kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3, 3))
        # morph1  = cv.morphologyEx(wynik, cv.MORPH_ERODE, kernel1)
        # if verbose:
            # cv.imwrite(skinOut_fn + "_04_eroded.png",morph1.astype(np.uint8))
        
        # #thicken what is left after the removal
        # kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5, 5))
        # morph2  = cv.morphologyEx(morph1, cv.MORPH_DILATE, kernel1)
        # if verbose:
            # cv.imwrite(skinOut_fn+"_05_dilated.png",morph2.astype(np.uint8))
        
        #subtract the thickened version. What is left are tose thin lines in the original image
        wynik_copy = cv.subtract(wynik_copy, skin_image)
        if verbose:
            cv.imwrite(skinOut_fn+"_06_thin_lines.png",wynik_copy.astype(np.uint8))
        
        #original image without the thin lines
        wynik_thick = cv.subtract(wynik, wynik_copy)
        if verbose:
            cv.imwrite(skinOut_fn+"_07_thin_lines_removed.png",wynik_thick.astype(np.uint8))
        
        #treat each line separately
        number_of_segments_plus1, labels1 = cv.connectedComponents(wynik_copy)
        # print(number_of_segments_plus1-1)
        #cv.imwrite(skinOut_fn+"test_seg.png",labels1.astype(np.uint8))
        for current_segment in range(1, number_of_segments_plus1):
            # print("Segment {}".format(current_segment))
            segment_image = np.zeros(labels1.shape, dtype = np.uint8)
            segment_image[labels1==current_segment] = 255

            #cv.imwrite(skinOut_fn+"test_seg{}.png".format(current_segment),segment_image.astype(np.uint8))
            
            #-------------------------------
            #estimate the required thickness
            
            #thicken the segment
            mask_size = 11
            kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(mask_size, mask_size))
            morph_seg = cv.morphologyEx(segment_image, cv.MORPH_DILATE, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
            
            #now make AND with the original skin image
            # print(wynik.shape)
            # print(type(wynik[0,0]))
            # print(morph_seg.shape)
            # print(type(morph_seg[0,0]))
            seg_neighborhood = cv.bitwise_and(wynik_thick, morph_seg)
            #cv.imwrite(skinOut_fn+"test_neighb_seg{}.png".format(current_segment),seg_neighborhood.astype(np.uint8))
            
            #the number of 255 pixels corresponds to the required thickness
            #thickness must be >= 2
            pixel_number = max([2, sum(seg_neighborhood.flatten()) / ((mask_size-1) / 2) / 2 / 255])
            #print(pixel_number)
            
            #thicken the thin lines
            kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(int(pixel_number*2+1), int(pixel_number*2+1)))
            morph3 = cv.morphologyEx(segment_image, cv.MORPH_DILATE, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
            #cv.imwrite(skinOut_fn+"test4_{}.png".format(current_segment),morph3.astype(np.uint8))
        
            #add the thickened lines to the original image
            wynik = cv.bitwise_or(wynik, morph3)
            
        if verbose:
            cv.imwrite(skinOut_fn+"_08_added_padding.png",wynik.astype(np.uint8))
        #cv.imwrite(skinOut_fn+"test6.png",roi_image.astype(np.uint8))
        
        kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9, 9))
        wynik  = cv.morphologyEx(wynik, cv.MORPH_CLOSE, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        if verbose:
            cv.imwrite(skinOut_fn+"_09_after_closing.png",wynik.astype(np.uint8))
        
        #subtract the roi from the result, so that the skin does not expand "inwards"
        wynik = cv.subtract(wynik, roi_image)
        if verbose:
            cv.imwrite(skinOut_fn+"_10_final.png",wynik.astype(np.uint8))
        
        
        # for j in range(0,wynik.shape[0]):
            # for i in range(0,wynik.shape[1]):
                # if((wynik[j,i,0]>0) | (wynik[j,i,1]>0) |(wynik[j,i,2]>0)):
                    # wynik[j,i,1] = 255
                # wynik[j,i,0] = 0
                # wynik[j,i,2] = 0
                
        
        tissue_polygons_out = v_polygons()
        tissue_polygons_out._mask_ndarray_to_polygons(wynik, background_val = 0, limit_polygons_num = 0)
        data = tissue_polygons_out.as_dict()

        jsonDumpSafe(skinOut_fn.rsplit('_',1)[0]+"_polygons.json", data)
        
        logging.info("writing file: {}".format(skinOut_fn))
        wynik = cv.cvtColor(wynik, cv.COLOR_GRAY2BGRA)
        for j in range(0,wynik.shape[0]):
            for i in range(0,wynik.shape[1]):
                if ((wynik[j,i,0]>0) | (wynik[j,i,1]>0) | (wynik[j,i,2]>0)):
                    wynik[j,i,0] = skin_image_copy[j,i]
                    wynik[j,i,1] = 0
                    wynik[j,i,2] = 255
                    wynik[j,i,3] = 255
                else:
                    wynik[j,i,3] = 0
        cv.imwrite(skinOut_fn,wynik.astype(np.uint8))
        
        a = cv.imread(polygon_file.rsplit('_',1)[0]+"_prob.png")
        cv.imwrite(oDir + "/" + os.path.split(polygon_file)[1].rsplit('_',1)[0]+"_prob.png",a)
        
        a = cv.imread(polygon_file.rsplit('_',1)[0]+"_prob_nl.png")
        cv.imwrite(oDir + "/" + os.path.split(polygon_file)[1].rsplit('_',1)[0]+"_prob_nl.png",a)
            
            
            
            
        
parser = ArgumentParser()

parser.add_argument("-iDir",      "--input_dir",       dest="idir",    help="input directory",      metavar="PATH", required=True)
parser.add_argument("-oDir",      "--output_dir",     dest="odir",    help="output directory",        metavar="PATH", required=True)

parser.add_argument("-v",       "--verbose",        dest="verbose",     help="verbose level",                                           required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
iDir 	= args.idir
oDir  	= args.odir


logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(oDir+"/correctShapesSkin.log",mode='w'),logging.StreamHandler(sys.stdout)])


if not os.path.isdir(iDir):
    logging.error('Error : Input directory (%s) not found !',iDir)
    exit(1)


logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_correctShapes2.py")
logging.info("in:  "    +   iDir    )
logging.info("out: "   +   oDir)

log2 = open(oDir+"/correctShapes2_results.log","a+")

if verbose == 'off':
    verbose = False
else:
    verbose = True

process_dir(iDir, oDir, log2, verbose)

log2.close()