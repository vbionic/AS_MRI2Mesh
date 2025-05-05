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
def find_nearest(point, vector):
    difx = vector[1,:] - point[1]
    dify = vector[0,:] - point[0]
    dif = []
    for i in range(0, len(difx)):
        dif.append(math.sqrt(difx[i]*difx[i] + dify[i]*dify[i]))
    
    min_dif = min(dif)
    return vector[:, dif.index(min_dif)]
    
def find_nearest_dist(point, vector):
    difx = vector[1,:] - point[1]
    dify = vector[0,:] - point[0]
    dif = []
    for i in range(0, len(difx)):
        dif.append(math.sqrt(difx[i]*difx[i] + dify[i]*dify[i]))
    
    min_dif = min(dif)
    return min_dif
    

def process_dir(iDir, oDir, ses_06_dir, ses_01_dir, dicom_fn_long, log2, ss, psx, psy, verbose):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    label_list    = glob.glob(iDir + "/*_roi_labels.png")
    label_list.sort()
    
    fileOut_fn = oDir + "/" + "extreme_coords.csv"
    fileOut = open(fileOut_fn,"w")
    index = []
    number = 1
    results = []
    
    
    #stack up ROIs for the given session
    stack = None
    
    for label_file in label_list:
        logging.info("opening file: {}".format(label_file))
        image_raw = cv.imread(label_file, cv.IMREAD_COLOR)
        image_raw = image_raw[:,:,2]
        image_raw[image_raw != 0] = 255
        if stack is None:
            stack = image_raw.copy()
        else:
            stack[image_raw == 255] = 255
        
    if verbose:
        cv.imwrite(oDir+"/stack.png",stack.astype(np.uint8))
        
        # minx = 100000000
        # miny = 100000000
        # maxx = -1
        # maxy = -1
        # num_found = 0
        
    if verbose:
        #image_DBG = stack.copy()
        image_DBG = cv.cvtColor(stack, cv.COLOR_GRAY2BGR)
        image_DBG[:,:,0] = 0
        image_DBG[:,:,1] = 0
        
        
    try:
        dicom_h = open(dicom_fn_long)
        dicomdata = json.load(dicom_h)
        dicom_h.close()
    except Exception as err:
        logging.error('Dicom file not found: "%s", error "%s"'%(dicom_fn_long,err))
        exit(1)
    
    invert_j_axis = False
    #check direction of the longitudal scan (up/down direction)
    if dicomdata["PatientPosition"][0:2] == 'HF':
        #head first orientation
        #if Z value of the image j axis is >0, then we need to invert the j coordinates
        if dicomdata["ImageOrientationPatient"][5] > 0:
            invert_j_axis = True
    elif dicomdata["PatientPosition"][0:2] == 'FF':
        #feet first orientation
        #if Z value of the image j axis is <0, then we need to invert the j coordinates
        if dicomdata["ImageOrientationPatient"][5] < 0:
            invert_j_axis = True
    else:
        logging.error("---!!!---                unknown patient orientation                ---!!!---")
    
    if invert_j_axis:
        logging.info("Direction of j axis longitudal inverted")
        
    borders = []
    rows = []
    for j in range(0,stack.shape[0]):
        found = False
        #searching for the leftmost white pixel
        minx = 0
        for i in range(0,stack.shape[1]):
            if stack[j,i] == 255:
                minx = i
                found = True
                if verbose:
                    image_DBG[j,i,1] = 255
                    image_DBG[j,i,2] = 1
                break
        #searching for the rightmost white pixel
        maxx = 0
        for i in range(stack.shape[1]-1,-1,-1):
            if stack[j,i] == 255:
                maxx = i
                found = True
                if verbose:
                    image_DBG[j,i,0] = 255
                    image_DBG[j,i,2] = 1                    
                break
        if found:
            borders.append([minx*psx, maxx*psx])
            if invert_j_axis:
                rows.append((stack.shape[0]-1 - j)*psy)
            else:
                rows.append(j*psy)
            fileOut.write("{},{}\n".format(minx,maxx))
    if verbose:
        cv.imwrite(oDir+"/stack_DBG.png",image_DBG)
        
    jsonDumpSafe(oDir + "/" +"extremes.json", [[i[0] for i in borders], [i[1] for i in borders], rows])
    
    plt.plot(rows,[i[0] for i in borders],'r-', rows,[i[1] for i in borders],'r+')
    plt.savefig(oDir + "/" + "extreme_coords.png")
    plt.clf()
    fileOut.close()
    
    
    
    orientation_i_long = dicomdata["ImageOrientationPatient"][0:2]
    
    ses_06_list    = [os.path.basename(path) for path in glob.glob(os.path.normpath(ses_06_dir + '/*'))]
    
    logging.info('session list to process:')
    ses_06_list.sort()
    for session in ses_06_list:
        logging.info('    {}'.format(session))
        
    for session in ses_06_list:
        description_fn = os.path.normpath(ses_06_dir + '/' + session + '/extremes.json')
        dicom_fn_wildcard = os.path.normpath(ses_01_dir + '/' + session + '/upsampled/*_dicom.json')
        try:
            description_h = open(description_fn)
            extremes_trans = json.load(description_h)
            description_h.close()
        except Exception as err:
            logging.error('extremes file not found: "%s", error "%s"'%(description_fn,err))
            exit(1)
        
        try:
            dicom_fn = glob.glob(dicom_fn_wildcard)[0]
        except Exception as err:
            logging.error('dicom transversal files not found: "%s", error "%s"'%(dicom_fn_wildcard,err))
            exit(1)
        try:
            dicom_h = open(dicom_fn)
            dicomdata_trans = json.load(dicom_h)
            dicom_h.close()
        except Exception as err:
            logging.error('dicom file not found: "%s", error "%s"'%(dicom_fn,err))
            exit(1)
            
        orientation_i_trans = dicomdata_trans["ImageOrientationPatient"][0:2]
        orientation_j_trans = dicomdata_trans["ImageOrientationPatient"][3:5]
        
        #finding a projection of i and j axes onto i axis of longitudal image
        projection_i = np.linalg.norm(np.dot(orientation_i_trans, orientation_i_long)/np.linalg.norm(orientation_i_long))
        projection_j = np.linalg.norm(np.dot(orientation_j_trans, orientation_i_long)/np.linalg.norm(orientation_i_long))
        
        if projection_i > projection_j:
            #exchange transversal axes
            logging.info("Exchanging i and j axes for transversal")
            temp = extremes_trans[0]
            extremes_trans[0] = extremes_trans[2]
            extremes_trans[2] = temp
            
            temp = extremes_trans[1]
            extremes_trans[1] = extremes_trans[3]
            extremes_trans[3] = temp
        
        plt.plot(rows,[i[0] for i in borders],'c-', rows,[i[1] for i in borders],'c+', extremes_trans[4], extremes_trans[0],'r-', extremes_trans[4], extremes_trans[1],'r+', extremes_trans[4], extremes_trans[2],'g-', extremes_trans[4], extremes_trans[3],'g+')
        plt.savefig(oDir + "/" + "extreme_coords_{}.png".format(session))
        plt.clf()
        
        data_trans = np.array([extremes_trans[2]+extremes_trans[3], extremes_trans[4] + extremes_trans[4]])
        data_long = np.array([[i[0] for i in borders] + [i[1] for i in borders], rows + rows])
        print(data_trans.shape)
        print(data_long.shape)
        
        avg_x_trans = sum(data_trans[1,:])/len(data_trans[1,:])
        avg_y_trans = sum(data_trans[0,:])/len(data_trans[0,:])
        
        avg_x_long = sum(data_long[1,:])/len(data_long[1,:])
        avg_y_long = sum(data_long[0,:])/len(data_long[0,:])
        
        dif_x = avg_x_trans - avg_x_long
        dif_y = avg_y_trans - avg_y_long
        
        min_dist_sum = 1e100
        min_ofset = [0,0]
        distance_sum = 0
        searchrange = 50
        search_step = 10
        search_div = 1.0
        for ofsetx in range(-searchrange,searchrange+1,search_step):
            for ofsety in range(-searchrange,searchrange+1,search_step):
                distance_sum = 0
                for i in range(0,data_trans.shape[1]):
                    distance_sum += find_nearest_dist([data_trans[0,i]+ofsety/search_div, data_trans[1,i]+ofsetx/search_div], data_long)
                if distance_sum < min_dist_sum:
                    min_dist_sum = distance_sum
                    min_ofset = [ofsety,ofsetx]
        print("minimum: {}".format(min_ofset))
        plt.plot(rows,[i[0] for i in borders],'c-', rows,[i[1] for i in borders],'c+', np.array(extremes_trans[4])+min_ofset[1], np.array(extremes_trans[2])+min_ofset[0],'g-', np.array(extremes_trans[4])+min_ofset[1], np.array(extremes_trans[3])+min_ofset[0],'g+')
        plt.savefig(oDir + "/" + "extreme_coords_adjusted_{}.png".format(session))
        plt.clf()
    

parser = ArgumentParser()

parser.add_argument("-iDir"  ,     "--input_dir"      ,     dest="idir"    ,    help="input directory" ,            metavar="PATH", required=True)
parser.add_argument("-iDir06",     "--input_dir_06"   ,     dest="idir06"  ,    help="input directory (stage 06)" , metavar="PATH", required=True)
parser.add_argument("-iDir01",     "--input_dir_01"   ,     dest="idir01"  ,    help="input directory (stage 01)" , metavar="PATH", required=True)
parser.add_argument("-d_fn_l" ,     "--dicom_filename_long"  ,     dest="dicom_fn_long"   ,    help="longitudal dicom filename" ,              metavar="PATH", required=True)
parser.add_argument("-oDir"  ,     "--output_dir"     ,     dest="odir"    ,    help="output directory",            metavar="PATH", required=True)
parser.add_argument("-ss"    ,     "--slice_spacing"  ,     dest="ss"      ,    help="slice spacing"   ,            metavar="PATH", required=True)
parser.add_argument("-psx"   ,     "--pixel_spacing_x",     dest="psx"     ,    help="pixel spacing x" ,            metavar="PATH", required=True)
parser.add_argument("-psy"   ,     "--pixel_spacing_y",     dest="psy"     ,    help="pixel spacing y" ,            metavar="PATH", required=True)

parser.add_argument("-v"    ,     "--verbose"        ,     dest="verbose",    help="verbose level"   ,                    required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
iDir 	= args.idir
ses_06_dir 	= args.idir06
ses_01_dir 	= args.idir01
dicom_fn_long   = args.dicom_fn_long 
oDir  	= args.odir
ss      = args.ss
psx     = args.psx
psy     = args.psy

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(oDir+"/correctShapesSkin.log",mode='w'),logging.StreamHandler(sys.stdout)])


if not os.path.isdir(iDir):
    logging.error('Error : Input directory (%s) not found !',iDir)
    exit(1)


logging.info("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_correctPosition.py")
logging.info("in:  "    +   iDir    )
logging.info("in ses 06:  "    +   ses_06_dir    )
logging.info("in ses 01:  "    +   ses_01_dir    )
logging.info("dicom_fn_long:  "    +   dicom_fn_long    )
logging.info("out: "   +   oDir)
logging.info("slice spacing: "   +   ss)
logging.info("pixel spacing x: "   +   psx)
logging.info("pixel spacing y: "   +   psy)
logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")


log2 = open(oDir+"/correctPosition_results.log","a+")

if verbose == 'off':
    verbose = False
else:
    verbose = True

process_dir(iDir, oDir, ses_06_dir, ses_01_dir, dicom_fn_long, log2, float(ss), float(psx), float(psy), verbose)

log2.close()