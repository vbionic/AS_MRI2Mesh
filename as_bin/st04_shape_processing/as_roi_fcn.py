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
from argparse import ArgumentParser
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import warnings
import glob
#-----------------------------------------------------------------------------------------

def process_using_unet_fcn():

    Ipath  = "./sample_mri_image(DICOM)/"
    Spath  = "./OutputImg/"

    print("Loading images .......... ")
    l = filelocation(Ipath)
    print(l)
    [orgImge,nofimage] =  LoadOrginalImage(l)
    print("Done! no of images: " + str(nofimage))
    print("Apply pipeline for generate masks.....")

    img_list = glob.glob(Ipath+"*.png")
    isz = io.imread(img_list[0])

    mask   = np.zeros((isz.shape[0],isz.shape[1],nofimage))
    fgnd   = np.zeros((isz.shape[0],isz.shape[1],nofimage))
    actImg = np.zeros((isz.shape[0],isz.shape[1],nofimage))

    selem  = disk(4)

    for i in tqdm(range(nofimage)):
        img             = orgImge[:,:,i]
        diff            = anisodiff(img,20,50,0.1)
        mu,sigma        = norm.fit(diff)
        htr             = apply_hysteresis_threshold(diff,mu,sigma).astype(int)
        pmask           = binary_fill_holes(htr)
        eroded          = erosion(pmask, selem)
        [fg,bg]         = foregroundBackground(eroded,img)
        mask[:,:,i]     = eroded
        fgnd[:,:,i]     = fg
        actImg[:,:,i]   = img
        sleep(0.1)


    print("save all generate image in a folder.... ")

    dictr = Spath
    for i in tqdm(range(nofimage)):
        x=l[i].split("/")
        loc1=dictr+x[-1]+'.png'
        loc2=dictr+x[-1]+'fg'+'.png'
        loc3=dictr+x[-1]+'actImg'+'.png'
        plt.imsave(loc1,mask[:,:,i],cmap = plt.cm.gray) # save psudo mask
        plt.imsave(loc2,fgnd[:,:,i],cmap = plt.cm.gray)
        plt.imsave(loc3,actImg[:,:,i],cmap = plt.cm.gray)


#-----------------------------------------------------------------------------------------
def main():

    parser = ArgumentParser()

    parser.add_argument("-imgDir",  "--img_dir",    dest="img_dir",  help="input png directory",            metavar="PATH",required=True)
    parser.add_argument("-roiDir",  "--roi_dir",    dest="roi_dir",  help="output roi shape directory",     metavar="PATH",required=True)
    parser.add_argument("-v",       "--verbose",    dest="verbose",  help="verbose level",                              required=False)

    args = parser.parse_args()

    verbose     = 'off'                 if args.verbose is None else args.verbose
    imgDir  	= args.img_dir
    roiDir  	= args.roi_dir

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

    logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(roiDir+"/roi_fcn.log",mode='w'),logging.StreamHandler(sys.stdout)])

    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    cd     =  os.path.dirname(os.path.realpath(__file__))
    cmd    = 'python3 '+cd+'/as_roi_fcn_util/as_wrapper.py ' 
    cmd   += '-v '          + verbose
    cmd   += ' -imgDir '    + imgDir
    cmd   += ' -roiDir '    + roiDir
    cmd   += ' -ns lsi '

    logging.info(cmd)
    ret = os.system(cmd)

    cd     =  os.path.dirname(os.path.realpath(__file__))
    cmd    = 'python3 '+cd+'/as_roi_fcn_util/as_wrapper.py ' 
    cmd   += '-v '          + verbose
    cmd   += ' -imgDir '    + imgDir
    cmd   += ' -roiDir '    + roiDir
    cmd   += ' -ns nsi '

    logging.info(cmd)
    ret = os.system(cmd)
    
    return    

#-----------------------------------------------------------------------------------------
   

if __name__ == '__main__':
    main()

