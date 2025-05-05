import sys, os
import pathlib
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
import getopt
import logging
import numpy as np
import json 
import cv2
import glob
import tracemalloc
import multiprocessing
#-----------------------------------------------------------------------------------------
from os import path
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import warnings
import glob
#-----------------------------------------------------------------------------------------
from argparse   import ArgumentParser
#-----------------------------------------------------------------------------------------
from v_utils.v_contour  import *
from v_utils.v_polygons import *
from v_utils.v_json import *
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
from AnDiffusion import *
from MaskGenAlgo import *
from hysteresisThresholding import apply_hysteresis_threshold
#-----------------------------------------------------------------------------------------
from tqdm import tqdm
from time import sleep
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from scipy.stats import norm
from scipy.ndimage.morphology import binary_fill_holes
import cv2
from skimage import io
#-----------------------------------------------------------------------------------------
def main():

    parser = ArgumentParser()

    parser.add_argument("-imgDir",  "--img_dir",    dest="img_dir",  help="input png directory",            metavar="PATH",required=True)
    parser.add_argument("-roiDir",  "--roi_dir",    dest="roi_dir",  help="output roi shape directory",     metavar="PATH",required=True)
    parser.add_argument("-v",       "--verbose",    dest="verbose",  help="verbose level",                                 required=False)
    parser.add_argument("-ns",      "--ns",         dest="ns",       help="name of scaled image",                          required=True)

    args = parser.parse_args()

    verbose     = 'off'                 if args.verbose is None else args.verbose
    imgDir  	= args.img_dir
    roiDir  	= args.roi_dir
    ns    	= args.ns

    imgDir = os.path.normpath(imgDir)
    roiDir = os.path.normpath(roiDir)

    if not os.path.isdir(imgDir):
        logging.error('Error : Input directory (%s) with PNG files not found !'%imgDir)
        exit(1)

    try:
        if not os.path.isdir(roiDir):
            pathlib.Path(roiDir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error("Creating roi shape dir (%s) IO error: %s"%(roiDir,err))
        exit(1)

    logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(roiDir+"/"+ns+"_roi_fcn.log",mode='w'),logging.StreamHandler(sys.stdout)])

    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    gname       = imgDir + '/*_'+ns+'.png'
    gname       = os.path.normpath(gname)

    images      = glob.glob(gname)
    imid        = 0

    if images == []:
        logging.error('> cannot find *nsi.png files, propably the imgDir is not valid (%s)'%gname)
        exit(1) 

    images.sort()

    imid        = 0
    logging.info('INFO  > %d files to process'%len(images))

    all_polygons = v_polygons()

    for iname in images:

        xname           = os.path.basename(iname)
        fname, fext     = os.path.splitext(xname)
        fname, fsuf     = fname.split('_')

        if imid!= 0:
            logging.info('> -----------------------------------------------------')

        logging.info('> file name     : ' + fname)

        img_path 	    = os.path.normpath(imgDir+'/'+fname+'_'+ns+'.png')
        img_mask_path 	    = os.path.normpath(roiDir+'/'+fname+'_'+ns+'_roi_fcn_labels.png')
        img_poly_path 	    = os.path.normpath(roiDir+'/'+fname+'_'+ns+'_roi_fcn_polygons.json')

        imid += 1

        img    = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

        green  = np.zeros((img.shape[0],img.shape[1]))
        mask   = np.zeros((img.shape[0],img.shape[1]))
        fgnd   = np.zeros((img.shape[0],img.shape[1]))
        actImg = np.zeros((img.shape[0],img.shape[1]))

        selem  = disk(4)

        diff            = anisodiff(img,20,50,0.1)
        mu,sigma        = norm.fit(diff)
        htr             = apply_hysteresis_threshold(diff,mu,sigma).astype(int)
        pmask           = binary_fill_holes(htr)
        eroded          = erosion(pmask, selem).astype(int)

        eroded[eroded!=0] = 255
        eroded.astype(np.uint8)

        labels = np.stack((eroded,green,eroded),axis=2)

        print(labels.shape)
        cv2.imwrite(img_mask_path,labels) # save psudo mask

        mypolygons = v_polygons.from_ndarray(eroded)
        jsonDumpSafe(img_poly_path, mypolygons.as_dict())

#-----------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

