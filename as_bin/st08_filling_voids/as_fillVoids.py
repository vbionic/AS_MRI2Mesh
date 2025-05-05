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

def read_files(iDir, pDir, ROI_file, tissue, verbose):
    tissue_file = os.path.normpath(iDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_labels.png')
    if verbose:
        logging.info("opening {} file: {}".format(tissue, tissue_file))
    tissue_raw = cv.imread(tissue_file, cv.IMREAD_COLOR)
    tissue_raw = tissue_raw[:,:,2]
    tissue_raw[tissue_raw != 0] = 255
    
    tissue_prob_file = os.path.normpath(pDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_prob_nl.png')
    if verbose:
        logging.info("opening {} prob file: {}".format(tissue, tissue_prob_file))
    tissue_prob = cv.imread(tissue_prob_file, cv.IMREAD_COLOR)
    tissue_prob = tissue_prob[:,:,2]
    return [tissue_raw, tissue_prob]
    
def wirte_files(oDir, ROI_file, tissue, labels):
    out_path = os.path.normpath(oDir + "/"+tissue)
    out_file = os.path.normpath(oDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_labels.png')
    try:
        if not os.path.isdir(out_path):
            pathlib.Path(out_path).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(out_path,err))
        exit(1)
    cv.imwrite(out_file, labels)
    
def get_neighboring_both_sides1(row_mid, col_mid, bones_prob, fat_prob, muscles_prob, skin_prob, vessels_prob, or_raw):
    
    same_on_both = [False, False, False, False, False]
    different_on_both = False
    test = True
    #               l-r               u-d            tl-dr            tr-dl
    combinations = [((0,1), (0,-1)), ((1,0),(-1,0)), ((-1,-1),(1,1)), ((-1,1),(1,-1))]
    for combination in combinations:
        
        row1 = row_mid + combination[0][0]
        row2 = row_mid + combination[1][0]
        col1 = col_mid + combination[0][1]
        col2 = col_mid + combination[1][1]
        
        if row1 >=0 and row1<bones_prob.shape[0] and col1>=0 and col1<bones_prob.shape[1] and row2 >=0 and row2<bones_prob.shape[0] and col2>=0 and col2<bones_prob.shape[1]:
            
            if (bones_prob[row1,col1]   == 255) and (bones_prob[row2,col2]   == 255):
                same_on_both[0] = True
                test = False
            if (fat_prob[row1,col1]     == 255) and (fat_prob[row2,col2]     == 255):
                same_on_both[1] = True
                test = False
            if (muscles_prob[row1,col1] == 255) and (muscles_prob[row2,col2] == 255):
                same_on_both[2] = True
                test = False
            if (skin_prob[row1,col1]    == 255) and (skin_prob[row2,col2]    == 255):
                same_on_both[3] = True
                test = False
            if (vessels_prob[row1,col1] == 255) and (vessels_prob[row2,col2] == 255):
                same_on_both[4] = True
                test = False
            if (or_raw[row1,col1]       == 255) and (or_raw[row2,col2]       == 255):
                different_on_both = True
    if not test:
        different_on_both = False
    return [same_on_both, different_on_both]
    
    
def get_neighboring_both_sides2(row_mid, col_mid, bones_prob, fat_prob, muscles_prob, skin_prob, vessels_prob, or_raw):
    
    same_on_both = [False, False, False, False, False]
    different_on_both = False
    test = True
    #               l-r               u-d            tl-dr            tr-dl
    combinations = [((0,1), (0,-1)), ((1,0),(-1,0)), ((-1,-1),(1,1)), ((-1,1),(1,-1)),    
    ((-2,-1),(2,-1)),
    ((-2,-1),(2,0)),
    ((-2,-1),(2,1)),
    ((-2,-1),(1,2)),
    
    ((-2,0),(2,-1)),
    ((-2,0),(2,0)),
    ((-2,0),(2,1)),
    
    ((-2,1),(2,-1)),
    ((-2,1),(2,0)),
    ((-2,1),(2,1)),
    ((-2,1),(1,-2)),
    
    ((-1,2),(1,-2)),
    ((-1,2),(0,-2)),
    ((-1,2),(-1,-2)),
    ((-1,2),(2,-1)),
    
    ((0,2),(-1,-2)),
    ((0,2),(0,-2)),
    ((0,2),(1,-2)),
    
    
    ((1,2),(-1,-2)),
    ((1,2),(0,-2)),
    ((1,2),(1,-2)),
    
    ((2,1),(-1, -2)),
    
    ((-1,-2),(2,1))    ]
    for combination in combinations:
        
        row1 = row_mid + combination[0][0]
        row2 = row_mid + combination[1][0]
        col1 = col_mid + combination[0][1]
        col2 = col_mid + combination[1][1]
        
        if row1 >=0 and row1<bones_prob.shape[0] and col1>=0 and col1<bones_prob.shape[1] and row2 >=0 and row2<bones_prob.shape[0] and col2>=0 and col2<bones_prob.shape[1]:
            
            if (bones_prob[row1,col1]   == 255) and (bones_prob[row2,col2]   == 255):
                same_on_both[0] = True
                test = False
            if (fat_prob[row1,col1]     == 255) and (fat_prob[row2,col2]     == 255):
                same_on_both[1] = True
                test = False
            if (muscles_prob[row1,col1] == 255) and (muscles_prob[row2,col2] == 255):
                same_on_both[2] = True
                test = False
            if (skin_prob[row1,col1]    == 255) and (skin_prob[row2,col2]    == 255):
                same_on_both[3] = True
                test = False
            if (vessels_prob[row1,col1] == 255) and (vessels_prob[row2,col2] == 255):
                same_on_both[4] = True
                test = False
            if (or_raw[row1,col1]       == 255) and (or_raw[row2,col2]       == 255):
                different_on_both = True
    if not test:
        different_on_both = False
    return [same_on_both, different_on_both]
    
def check_surroundings(row_mid, col_mid, bw_img, bones_prob, fat_prob, muscles_prob, skin_prob, vessels_prob, or_raw, configuration, debug_img):
    
    neighboring_pixel_num  = np.zeros(6)
    neighboring_pixel_sum  = np.zeros(6)
    neighboring_pixel_avg  = np.zeros(6)
    neighboring_pixel_prob = np.zeros(5)
    #---------------------------------#
    #value map:
    # 0 : bones
    # 1 : fat
    # 2 : muscles
    # 3 : skin
    # 4 : vessels
    # 5 : unknown
    
    for row in range(row_mid-configuration["prob_search_range"], row_mid+configuration["prob_search_range"]+1):
        if row >=0 and row<bw_img.shape[0]:
            for col in range(col_mid-configuration["prob_search_range"], col_mid+configuration["prob_search_range"]+1):
                if col>=0 and col<bw_img.shape[1]:
                    if bones_prob[row,col] > 0:
                        neighboring_pixel_num[0] += 1
                        neighboring_pixel_sum[0] += bw_img[row, col]
                    if fat_prob[row,col] > 0:
                        neighboring_pixel_num[1] += 1
                        neighboring_pixel_sum[1] += bw_img[row, col]
                    if muscles_prob[row,col] > 0:
                        neighboring_pixel_num[2] += 1
                        neighboring_pixel_sum[2] += bw_img[row, col]
                    if skin_prob[row,col] > 0:
                        neighboring_pixel_num[3] += 1
                        neighboring_pixel_sum[3] += bw_img[row, col]
                    if vessels_prob[row,col] > 0:
                        neighboring_pixel_num[4] += 1
                        neighboring_pixel_sum[4] += bw_img[row, col]
                    if or_raw[row,col] == 0:
                        neighboring_pixel_num[5] += 1
                        neighboring_pixel_sum[5] += bw_img[row, col]
    for ind in range(0,6):
        if neighboring_pixel_num[ind]>0:
            neighboring_pixel_avg[ind] = neighboring_pixel_sum[ind] / neighboring_pixel_num[ind]
    
    curr_pix = bw_img[row_mid, col_mid]
    for ind in range(0,6):
        neighboring_pixel_avg[ind] = abs(neighboring_pixel_avg[ind]-curr_pix)
    
    #debug_img[row_mid,col_mid] = neighboring_pixel_num[1]*255/8
    #no neighborhood or too many unknowns
    if(sum(neighboring_pixel_num[0:5]) == 0) or (neighboring_pixel_num[5]>configuration["max_unknown_pixels_in_neighb"]):
        return np.zeros(5)
        
        
    for ind in range(0,5):
        #the more neighbors of certain type, the higher the probability, the smaller the difference between the brightness and the average brightness for a certain tissue, the better
        neighboring_pixel_prob[ind] = neighboring_pixel_num[ind]/(1+neighboring_pixel_avg[ind])

    if sum(neighboring_pixel_prob)>0:
        neighboring_pixel_prob /= sum(neighboring_pixel_prob)
    
    #advanced check: the same tissue or different tissues must be present on both sides
    #case for 1 pixel search range:
    #
    #   lu  u   ru
    #   l   x   r
    #   ld  d   rd
    #
    #debug_img[row_mid,col_mid] = neighboring_pixel_prob[1]*255
    
    if configuration["both_sides_required"]:
        if configuration["prob_search_range"] == 1:
            [both_sides_present_same, both_sides_present_different] = get_neighboring_both_sides1(row_mid, col_mid, bones_prob, fat_prob, muscles_prob, skin_prob, vessels_prob, or_raw)
            # if both_sides_present_different and both_sides_present_same[1]:
                # debug_img[row_mid,col_mid] = 255
            # elif both_sides_present_same[1]:
                # debug_img[row_mid,col_mid] = 200
            # elif both_sides_present_different:
                # debug_img[row_mid,col_mid] = 150
            if both_sides_present_different:
                debug_img[row_mid,col_mid] = 150
            # else:
                # debug_img[row_mid,col_mid] = 100
            for i in range(0,5):
            #    if both_sides_present_same[i]:
            #        debug_img[row_mid,col_mid] = i*50
                if configuration["treat_different_tissues_as_valid"]:
                    if (not both_sides_present_same[i]) and (not both_sides_present_different):
                        neighboring_pixel_prob[i] = 0
                        #debug_img[row_mid,col_mid] = 50
                else:
                    if (not both_sides_present_same[i]):
                        neighboring_pixel_prob[i] = 0
                        
        if configuration["prob_search_range"] == 2:
            [both_sides_present_same, both_sides_present_different] = get_neighboring_both_sides2(row_mid, col_mid, bones_prob, fat_prob, muscles_prob, skin_prob, vessels_prob, or_raw)
            # if both_sides_present_different and both_sides_present_same[1]:
                # debug_img[row_mid,col_mid] = 255
            # elif both_sides_present_same[1]:
                # debug_img[row_mid,col_mid] = 200
            # elif both_sides_present_different:
                # debug_img[row_mid,col_mid] = 150
            # else:
                # debug_img[row_mid,col_mid] = 100
            for i in range(0,5):
            #    if both_sides_present_same[i]:
            #        debug_img[row_mid,col_mid] = i*50
                if configuration["treat_different_tissues_as_valid"]:
                    if (not both_sides_present_same[i]) and (not both_sides_present_different):
                        neighboring_pixel_prob[i] = 0
                        #debug_img[row_mid,col_mid] = 50
                else:
                    if (not both_sides_present_same[i]):
                        neighboring_pixel_prob[i] = 0
    
        
    return neighboring_pixel_prob
    
    
        
def process_dir(iDir, oDir, pDir, log2, configuration, verbose):
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
        wirte_files(oDir, ROI_file, "roi", ROI_raw)
        ROI_raw = ROI_raw[:,:,2]
        ROI_raw[ROI_raw != 255] = 0
        
#BONES
        [bones_raw, bones_prob] = read_files(iDir, pDir, ROI_file, "bones", verbose)
#FAT
        [fat_raw, fat_prob] = read_files(iDir, pDir, ROI_file, "fat", verbose)
#MUSCLES
        [muscles_raw, muscles_prob] = read_files(iDir, pDir, ROI_file, "muscles", verbose)
#SKIN
        [skin_raw, skin_prob] = read_files(iDir, pDir, ROI_file, "skin", verbose)
#VESSELS
        [vessels_raw, vessels_prob] = read_files(iDir, pDir, ROI_file, "vessels", verbose)
        
        #remove specks from labels (fat and muscles should benefit from this)
        if configuration["fat"]["raw_open"]>0:
            kernel1           = cv.getStructuringElement(cv.MORPH_ELLIPSE,(configuration["fat"]["raw_open"]*2+1, configuration["fat"]["raw_open"]*2+1))
            fat_raw  = cv.morphologyEx(fat_raw, cv.MORPH_OPEN, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        
        if configuration["muscles"]["raw_open"]>0:
            kernel1           = cv.getStructuringElement(cv.MORPH_ELLIPSE,(configuration["muscles"]["raw_open"]*2+1, configuration["muscles"]["raw_open"]*2+1))
            muscles_raw  = cv.morphologyEx(muscles_raw, cv.MORPH_OPEN, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        
        if configuration["bones"]["raw_open"]>0:
            kernel1           = cv.getStructuringElement(cv.MORPH_ELLIPSE,(configuration["bones"]["raw_open"]*2+1, configuration["bones"]["raw_open"]*2+1))
            bones_raw  = cv.morphologyEx(bones_raw, cv.MORPH_OPEN, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        
        
        #remove tissues outside ROI
        bones_raw = cv.bitwise_and(ROI_raw, bones_raw)
        fat_raw = cv.bitwise_and(ROI_raw, fat_raw)
        muscles_raw = cv.bitwise_and(ROI_raw, muscles_raw)
        vessels_raw = cv.bitwise_and(ROI_raw, vessels_raw)
        
        #check where there are definite tissues
        logic_or = np.zeros(ROI_raw.shape, dtype=np.uint8)
        
        logic_or = cv.bitwise_or(logic_or, bones_raw)
        logic_or = cv.bitwise_or(logic_or, fat_raw)
        logic_or = cv.bitwise_or(logic_or, muscles_raw)
        logic_or = cv.bitwise_or(logic_or, skin_raw)
        logic_or = cv.bitwise_or(logic_or, vessels_raw)
        if verbose:
            wirte_files(oDir, ROI_file, "OR", logic_or)
        
        #remove specks from probabilities (fat and muscles should benefit from this)
        if configuration["fat"]["open"]>0:
            kernel1           = cv.getStructuringElement(cv.MORPH_ELLIPSE,(configuration["fat"]["open"]*2+1, configuration["fat"]["open"]*2+1))
            fat_prob  = cv.morphologyEx(fat_prob, cv.MORPH_OPEN, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        
        if configuration["muscles"]["open"]>0:
            kernel1           = cv.getStructuringElement(cv.MORPH_ELLIPSE,(configuration["muscles"]["open"]*2+1, configuration["muscles"]["open"]*2+1))
            muscles_prob  = cv.morphologyEx(muscles_prob, cv.MORPH_OPEN, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        
        if configuration["bones"]["open"]>0:
            kernel1           = cv.getStructuringElement(cv.MORPH_ELLIPSE,(configuration["bones"]["open"]*2+1, configuration["bones"]["open"]*2+1))
            bones_prob  = cv.morphologyEx(bones_prob, cv.MORPH_OPEN, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        
        #check where there are non-zero probabilities of at least one tissue
        logic_or_prob = np.zeros(ROI_raw.shape, dtype=np.uint8)
        
        prob_avail = np.zeros(ROI_raw.shape, dtype=np.uint8)
        prob_avail[bones_prob>0] = 255
        logic_or_prob = cv.bitwise_or(logic_or_prob, prob_avail)
        
        prob_avail = np.zeros(ROI_raw.shape, dtype=np.uint8)
        prob_avail[fat_prob>0] = 255
        logic_or_prob = cv.bitwise_or(logic_or_prob, prob_avail)
        
        prob_avail = np.zeros(ROI_raw.shape, dtype=np.uint8)
        prob_avail[muscles_prob>0] = 255
        logic_or_prob = cv.bitwise_or(logic_or_prob, prob_avail)
        
        #prob_avail = np.zeros(ROI_raw.shape, dtype=np.uint8)
        #prob_avail[skin_prob>0] = 255
        #logic_or_prob = cv.bitwise_or(logic_or_prob, prob_avail)
        
        prob_avail = np.zeros(ROI_raw.shape, dtype=np.uint8)
        prob_avail[vessels_prob>0] = 255
        logic_or_prob = cv.bitwise_or(logic_or_prob, prob_avail)
        
        logic_or_prob = cv.bitwise_and(logic_or_prob, ROI_raw)
        
        if verbose:
            wirte_files(oDir, ROI_file, "OR_prob", logic_or_prob)
        
        #find the area where no definite tissues are present, but a non-zero probability for at least one tissue exists
        logic_diff = cv.bitwise_xor(logic_or_prob, logic_or)
        
        if verbose:
            wirte_files(oDir, ROI_file, "roznica", logic_diff)
        
        unknown_raw = np.zeros(ROI_raw.shape)
        
        
        #try to fill the single (double?) pixel wide gaps
        #between the same tissue - closing operation (?)
        
        #between different tissues - in horizontal, diagonal and vertical directions - select the tissue with more similar brightness
        bw_img_file = os.path.normpath(pDir + "/images/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_nsi.png')
        bw_img = cv.imread(bw_img_file, cv.IMREAD_COLOR)
        bw_img = bw_img[:,:,2]
        
        
        debug1 = np.zeros(ROI_raw.shape)
        debug2 = np.zeros(ROI_raw.shape)
        if verbose:
            wirte_files(oDir, ROI_file, "fat_prob1", fat_prob)

        updated_prob = np.zeros((ROI_raw.shape[0], ROI_raw.shape[1], 5), dtype=np.uint8)
        
        area_to_check = cv.bitwise_and(ROI_raw, cv.bitwise_not(logic_or))
        for row in range(0,ROI_raw.shape[0]):
            for col in range(0,ROI_raw.shape[1]):
                if(area_to_check[row,col] > 0):
                    new_prob = check_surroundings(row, col, bw_img, bones_prob, fat_prob, muscles_prob, skin_prob, vessels_prob, logic_or, configuration, debug2)
                    updated_prob[row,col,0] = new_prob[0] * 128
                    updated_prob[row,col,1] = new_prob[1] * 128
                    updated_prob[row,col,2] = new_prob[2] * 128
                    updated_prob[row,col,3] = new_prob[3] * 128
                    updated_prob[row,col,4] = new_prob[4] * 128
                    
                    #bones_prob[row, col]   += new_prob[0] * 128
                    #fat_prob[row, col]     += new_prob[1] * 128
                    debug1[row,col] = 255
                    #muscles_prob[row, col] += new_prob[2]* 128
                    #skin_prob[row, col]    += new_prob[3]* 128
                    #vessels_prob[row, col] += new_prob[4]* 128
        
        bones_prob[:,:] += updated_prob[:,:,0]
        fat_prob[:,:] += updated_prob[:,:,1]
        muscles_prob[:,:] += updated_prob[:,:,2]
        skin_prob[:,:] += updated_prob[:,:,3]
        vessels_prob[:,:] += updated_prob[:,:,4]
        
        if configuration["zero_prob_when_not_in_vicinity"]:
            
            for row in range(0,ROI_raw.shape[0]):
                for col in range(0,ROI_raw.shape[1]):
                    if(area_to_check[row,col] > 0):
                        neighboring_pixel_num = np.zeros(5)
                        neighboring_pixel_sum = np.zeros(5)
                        for row_i in range(row-configuration["vicinity_range"], row+configuration["vicinity_range"]+1):
                            if row_i >=0 and row_i<bw_img.shape[0]:
                                for col_i in range(col-configuration["vicinity_range"], col+configuration["vicinity_range"]+1):
                                    if col_i>=0 and col_i<bw_img.shape[1]:
                                        if bones_raw[row_i,col_i] > 0:
                                            neighboring_pixel_num[0] += 1
                                            neighboring_pixel_sum[0] += bw_img[row_i, col_i]
                                        if fat_raw[row_i,col_i] > 0:
                                            neighboring_pixel_num[1] += 1
                                            neighboring_pixel_sum[1] += bw_img[row_i, col_i]
                                        if muscles_raw[row_i,col_i] > 0:
                                            neighboring_pixel_num[2] += 1
                                            neighboring_pixel_sum[2] += bw_img[row_i, col_i]
                                        if skin_raw[row_i,col_i] > 0:
                                            neighboring_pixel_num[3] += 1
                                            neighboring_pixel_sum[3] += bw_img[row_i, col_i]
                                        if vessels_raw[row_i,col_i] > 0:
                                            neighboring_pixel_num[4] += 1
                                            neighboring_pixel_sum[4] += bw_img[row_i, col_i]
        
                        #zero probability of the tissues that were not present in the neighborhood
                        if neighboring_pixel_num[0] == 0:
                            bones_prob[row,col] = 0
                        if neighboring_pixel_num[1] == 0:
                            fat_prob[row,col] = 0
                        if neighboring_pixel_num[2] == 0:
                            muscles_prob[row,col] = 0
                        if neighboring_pixel_num[3] == 0:
                            skin_prob[row,col] = 0
                        if neighboring_pixel_num[4] == 0:
                            vessels_prob[row,col] = 0
                            
        #produce non-zero fat probability near the skin - usually, there is fat under the skin
        #dilate skin by the configured values
        
        kernel1           = cv.getStructuringElement(cv.MORPH_ELLIPSE,(configuration["fat"]["skin_distance"]*2+1, configuration["fat"]["skin_distance"]*2+1))
        skin_raw_dilated  = cv.morphologyEx(skin_raw, cv.MORPH_DILATE, kernel1, borderType = cv.BORDER_CONSTANT, borderValue = 0)
        if verbose:
            wirte_files(oDir, ROI_file, "skin_dilated", skin_raw_dilated)
        probable_fat      = (cv.bitwise_and(skin_raw_dilated, cv.bitwise_and(ROI_raw, cv.bitwise_not(logic_or)))/10).astype(np.uint8)
        
        if verbose:
            wirte_files(oDir, ROI_file, "fat_under_skin", probable_fat)
        fat_prob += probable_fat 
        
        if verbose:
            wirte_files(oDir, ROI_file, "fat_prob2", fat_prob)
            wirte_files(oDir, ROI_file, "debug1", debug1)
            wirte_files(oDir, ROI_file, "debug2", debug2)
        
        
        
        #perform probability check
        
        for row in range(0,ROI_raw.shape[0]):
            for col in range(0,ROI_raw.shape[1]):
                if(logic_or[row,col]==0) and (ROI_raw[row,col] > 0):
                    #pixel is in ROI
                    probs = [fat_prob[row,col], muscles_prob[row,col], vessels_prob[row,col]]
                    if max(probs) > 0:
                        bestfit = probs.index(max(probs))
                        if bestfit == 0:
                            fat_raw[row,col] = 255
                        if bestfit == 1:
                            muscles_raw[row,col] = 255
                        if bestfit == 2:
                            vessels_raw[row,col] = 255
                    else:
                        unknown_raw[row,col] = 255
                    #bones are not the greatest filler - they are never scattered around
                    #skin is not a good filler - it is never inside the arm
        
        wirte_files(oDir, ROI_file, "bones", bones_raw)
        wirte_files(oDir, ROI_file, "fat", fat_raw)
        wirte_files(oDir, ROI_file, "muscles", muscles_raw)
        wirte_files(oDir, ROI_file, "skin", skin_raw)
        wirte_files(oDir, ROI_file, "vessels", vessels_raw)
        wirte_files(oDir, ROI_file, "unknown", unknown_raw)



parser = ArgumentParser()

parser.add_argument("-iDir",      "--input_dir"      ,     dest="idir"   ,    help="input directory" ,    metavar="PATH", required=True)
parser.add_argument("-pDir",      "--prob_dir"      ,      dest="pdir"   ,    help="input directory with probability images" ,    metavar="PATH", required=True)
parser.add_argument("-oDir",      "--output_dir"     ,     dest="odir"   ,    help="output directory",    metavar="PATH", required=True)
parser.add_argument("-conf",      "--configuration"  ,     dest="conffn" ,    help="configuration file name",    metavar="PATH", required=True)

parser.add_argument("-v"   ,      "--verbose"        ,     dest="verbose",    help="verbose level"   ,                    required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
iDir 	= args.idir
oDir  	= args.odir
pDir  	= args.pdir
conffn  = args.conffn

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(oDir+"/fillVoids.log",mode='w'),logging.StreamHandler(sys.stdout)])


if not os.path.isdir(iDir):
    logging.error('Error : Input directory (%s) not found !',iDir)
    exit(1)


logging.info("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_fillVoids.py")
logging.info("in:       "    +   iDir    )
logging.info("in prob:  "    +   pDir    )
logging.info("config:   "    +   conffn  )
logging.info("out:      "    +   oDir    )
logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")


log2 = open(oDir+"/fillVoids_results.log","a+")

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

process_dir(iDir, oDir, pDir, log2, configuration, verbose)

log2.close()