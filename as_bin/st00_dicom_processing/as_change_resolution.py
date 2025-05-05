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

#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_json import *
#-----------------------------------------------------------------------------------------

def calculate_3D_point(transform, point):
    a = np.matmul(transform,point)
    return [float(a[0]), float(a[1]), float(a[2])]


def process_dir(dir, outputdir, pixelspacing, verbose):
    
    global_max = 0
    global_min = 100000
    
    inputfiles = glob.glob(dir + '/*_fsi.png')
    inputfiles.sort()
    logging.info("Scanning: "+dir)
    #print(inputfiles)
    for file in inputfiles:
        if os.path.isdir(file):
            file.replace("\\",'/')
            logging.info("    Entering: "+file)
            process_dir(dir, file[len(dir)+1:], outputdir)
        else:
            file.replace("\\",'/')
            logging.info("        Processing: "+file)
            [local_min, local_max] = process_file_1(file, pixelspacing, outputdir)
            if local_min < global_min:
                global_min = local_min
            if local_max > global_max:
                global_max = local_max
    
    
    logging.debug("min: {}, max: {}".format(global_min, global_max))
    
    for file in inputfiles:
        if os.path.isdir(file):
            file.replace("\\",'/')
            logging.info("    Entering(2): "+file)
            process_dir(dir, file[len(dir)+1:], outputdir)
        else:
            file.replace("\\",'/')
            logging.info("        Processing(2): "+file)
            process_file_2(file, outputdir, global_min, global_max)
            
    name_sorted_in = dir+"/sorted.json"
    name_geometry_in = dir+"/set_data.json"
    
    name_sorted_out = outputdir+"/sorted.json"
    name_geometry_out = outputdir+"/set_data.json"
    
    try:
        file_geom_in = open(name_geometry_in)
        filedata = json.load(file_geom_in)
        file_geom_in.close()
    except Exception as err:
        logging.error("Input dicom data IO error: {}".format(err))
        sys.exit(1)
        
    filedata["pixel_spacing_x"] = pixelspacing
    filedata["pixel_spacing_y"] = pixelspacing
    
    try:
        #jsonfile = open(os.open(name_geometry_out, os.O_CREAT | os.O_WRONLY, 0o775),'w')
        #json.dump(filedata, jsonfile, indent=4)
        #jsonfile.close()
        jsonUpdate(name_geometry_out, filedata)
    except Exception as err:
        logging.error("Output JSON file IO error: {}".format(err))
        sys.exit(1)
    
    shutil.copyfile(name_sorted_in, name_sorted_out) 

#-----------------------------------------------------------------------------------------

def process_file_1(inputfile, pixelspacing, outputdir):

    try:        
        #img = cv.imread(inputfile,-1) #-1 zeby wczytac 16 bitowo
        img = cv.imread(inputfile, cv.IMREAD_ANYDEPTH)
    except Exception as err:
        logging.error("Input file IO error: {}".format(err))
        sys.exit(1)
    try:
        if not os.path.isdir(outputdir):
            pathlib.Path(outputdir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error("Output dir IO error: {}".format(err))
        sys.exit(1)


    name_dicom_out      = outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit("_",1)[0]+"_dicom.json"

    name_dicom_in       = inputfile.replace('\\','/').rsplit("_",1)[0]+"_dicom.json"

    name_16bit_out      = outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit("_",1)[0]+"_fsi.png"

    name_lsi_out        = outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit("_",1)[0]+"_lsi.png"

    name_localmax_out   = outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit("_",1)[0]+"_data.json"

    try:
        file_dicom_in = open(name_dicom_in)
        filedata = json.load(file_dicom_in)
        file_dicom_in.close()
    except Exception as err:
        logging.error("Input dicom data IO error: {}".format(err))
        sys.exit(1)
        
    Sx = filedata["ImagePositionPatient"][0]
    Sy = filedata["ImagePositionPatient"][1]
    Sz = filedata["ImagePositionPatient"][2]
    Xx = filedata["ImageOrientationPatient"][0]
    Xy = filedata["ImageOrientationPatient"][1]
    Xz = filedata["ImageOrientationPatient"][2]
    Yx = filedata["ImageOrientationPatient"][3]
    Yy = filedata["ImageOrientationPatient"][4]
    Yz = filedata["ImageOrientationPatient"][5]
    Di = filedata["PixelSpacing"][0]
    Dj = filedata["PixelSpacing"][1]
    
    _3Dtransform_matrix = [[Xx*Di, Yx*Dj ,0, Sx],[Xy*Di, Yy*Dj, 0, Sy],[Xz*Di, Yz*Dj, 0, Sz],[0, 0, 0, 1]]

    current_pixelspacing_x = filedata["PixelSpacing"][0]
    current_pixelspacing_y = filedata["PixelSpacing"][1]
    
    pixelspacing = float(pixelspacing)
    filedata["PixelSpacing"][0] = pixelspacing
    filedata["PixelSpacing"][1] = pixelspacing
    
    new_ImagePositionPatient = calculate_3D_point(_3Dtransform_matrix, [-(current_pixelspacing_x/2 - pixelspacing/2)/current_pixelspacing_x, -(current_pixelspacing_y/2 - pixelspacing/2)/current_pixelspacing_y, 0, 1])
    
    filedata["ImagePositionPatient"][0] = new_ImagePositionPatient[0]
    filedata["ImagePositionPatient"][1] = new_ImagePositionPatient[1]
    filedata["ImagePositionPatient"][2] = new_ImagePositionPatient[2]
    
    
    filedata["ImageOrientationPatient"][0] = filedata["ImageOrientationPatient"][0]
    filedata["ImageOrientationPatient"][1] = filedata["ImageOrientationPatient"][1]
    filedata["ImageOrientationPatient"][2] = filedata["ImageOrientationPatient"][2]
    filedata["ImageOrientationPatient"][3] = filedata["ImageOrientationPatient"][3]
    filedata["ImageOrientationPatient"][4] = filedata["ImageOrientationPatient"][4]
    filedata["ImageOrientationPatient"][5] = filedata["ImageOrientationPatient"][5]
    
    transform = np.zeros((2,3))
    
    transform[0][0] = current_pixelspacing_x / float(pixelspacing) 
    transform[0][1] = 0
    transform[1][0] = 0
    transform[1][1] = current_pixelspacing_y / float(pixelspacing) 
    
#    print(transform)
    
    img_rescaled16 = cv.warpAffine(img, transform, (math.ceil(float(filedata["Columns"])*current_pixelspacing_x/pixelspacing), math.ceil(float(filedata["Rows"])*current_pixelspacing_y/pixelspacing)), flags = cv.INTER_LANCZOS4)
    
    
#    cv.imshow('przeskalowane',img*256)
#    print(np.asarray(img).max())
#    cv.waitKey(0)
    
    a = np.asarray(img_rescaled16)

    try:
        Image.fromarray(a.astype(np.uint16),mode='I;16').save(name_16bit_out)
    except Exception as err:
        logging.error("Output file 16 bit IO error: {}".format(err))
        sys.exit(1)

    a = a.astype(np.float32)

    local_min = a.min() 
    local_max = a.max()

    a = a - a.min()
    if a.max()>0:
        a = (a*255)/a.max()

    image2 = np.array(a, copy=True, dtype=np.uint8)
    obraz = Image.fromarray(image2)
    filedata["Columns"] = obraz.size[0]
    filedata["Rows"] = obraz.size[1]

    try:
        obraz.save(name_lsi_out)
    except Exception as err:
        logging.error("Output file 8 bit IO error: {}".format(err))
        sys.exit(1)


    try:
        #jsonfile = open(os.open(name_dicom_out, os.O_CREAT | os.O_WRONLY, 0o775),'w')
        #json.dump(filedata, jsonfile, indent=4)
        #jsonfile.close()
        jsonUpdate(name_dicom_out, filedata)
    except Exception as err:
        logging.error("Output JSON file IO error: {}".format(err))
        sys.exit(1)

    try:
        #jsonfile = open(os.open(name_localmax_out, os.O_CREAT | os.O_WRONLY, 0o775),'w')
        #json.dump({'max_file': int(local_max), 'min_file': int(local_min)},jsonfile, indent=4)
        #jsonfile.close()
        print(name_localmax_out)
        jsonUpdate(name_localmax_out, {'max_file': int(local_max), 'min_file': int(local_min)})
    except Exception as err:
        logging.error('Output file json max file IO error: {}'.format(err))
        sys.exit(1)

    return [local_min, local_max]

#-----------------------------------------------------------------------------------------

def process_file_2(inputfile, outputdir, global_min, global_max):

    name_16bit_in      = outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit("_",1)[0]+"_fsi.png"
    
    try:        
        img = cv.imread(name_16bit_in, cv.IMREAD_ANYDEPTH)
    except Exception as err:
        logging.error("Input file IO error: {}".format(err))
        sys.exit(1)
    try:
        if not os.path.isdir(outputdir):
            pathlib.Path(outputdir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error("Output dir IO error: {}".format(err))
        sys.exit(1)

    name_gsi_out = outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit("_",1)[0]+"_gsi.png"

    name_max_out = outputdir+"/set_data.json"

    b = np.asarray(img)

    b = b - global_min
    if global_max>0:
        b = (b*255)/(global_max - global_min)

    image2a = np.array(b, copy=True, dtype=np.uint8)

    #try:
    #    Image.fromarray(image2a).save(name_gsi_out)
    #except Exception as err:
    #    logging.error("Output file 8 bit IO error: {}".format(err))
    #    sys.exit(1)

    try:
        #jsonfile = open(os.open(name_max_out, os.O_CREAT | os.O_WRONLY, 0o775),'w')
        #json.dump({'max_set': int(global_max), 'min_set': int(global_min)},jsonfile, indent=4)
        #jsonfile.close()
        jsonUpdate(name_max_out, {'max_set': int(global_max), 'min_set': int(global_min)})
    except Exception as err:
        logging.error('Output file json max file IO error: {}'.format(err))
        sys.exit(1)

#-----------------------------------------------------------------------------------------

parser = ArgumentParser()

parser.add_argument("-dprDir",  "--dpr_dir",     dest="dpr_dir",  help="output resampled png directory",       metavar="PATH",required=True)
parser.add_argument("-dpDir",  "--dp_dir",     dest="dp_dir",  help="input png directory", metavar="PATH",required=True)
parser.add_argument("-ps",  "--pix_spacing",     dest="ps",  help="pixel spacing in mm", metavar="PATH",required=True)
parser.add_argument("-v",      "--verbose",    dest="verbose", help="verbose level",                              required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
dprDir 	= args.dpr_dir
dpDir  	= args.dp_dir
pixelspacing      = args.ps


try:
	if not os.path.isdir(dprDir):
	    pathlib.Path(dprDir).mkdir(mode=0o775, parents=True, exist_ok=True)
except Exception as err:
	print("Output dir IO error: {}".format(err))
	sys.exit(1)

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(dprDir+"/change_resolution.log",mode='w'),logging.StreamHandler(sys.stdout)])


if not os.path.isdir(dpDir):
    logging.error('Error : Input directory (%s) with DICOM files not found !',dpDir)
    exit(1)

logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_change_resolution.py")
logging.info("in: "+dpDir)
logging.info("pixelspacing: "+pixelspacing)
logging.info("out: "+dprDir)



process_dir(dpDir, dprDir, float(pixelspacing), verbose)
    
#-----------------------------------------------------------------------------------------
