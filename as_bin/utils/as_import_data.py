import sys, getopt
import pydicom
import json
from pydicom.datadict import keyword_for_tag
#from pydicom.filereader import read_dicomdir
import os
import pathlib
from argparse import ArgumentParser
import glob
import math
import logging
import re
import shutil
import numpy as np

#-----------------------------------------------------------------------------------
def getDICOMheader(inputfile):
    try:        
        dataset = pydicom.dcmread(inputfile)
    except Exception as err:
        logging.error("Input file IO error: {}".format(err))
        sys.exit(1)
    return dataset
#-----------------------------------------------------------------------------------




parser = ArgumentParser()

parser.add_argument("-dst",   "--destinationDir",         dest="dst_dir",      help="Input (source of raw DICOM) directory",         metavar="PATH",required=True)
parser.add_argument("-src",  "--sourceDir",   dest="src_dir",     help="Destination directory", metavar="PATH",required=True)
parser.add_argument("-v",       "--verbose",        dest="verbose",     help="verbose level",                              required=False)


args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose

dstDir   	= args.dst_dir
srcDir   	= args.src_dir


dstDir = os.path.normpath(dstDir)
srcDir = os.path.normpath(srcDir)


try:
    if not os.path.isdir(dstDir):
        pathlib.Path(dstDir).mkdir(mode=0o775, parents=True, exist_ok=True)
except Exception as err:
    print('INFO      > creating "%s" directory failed, error "%s"'%(dstDir,err))
    sys.exit(1)

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(dstDir+"/importDicom.log",mode='w'),logging.StreamHandler(sys.stdout)])

if not os.path.isdir(srcDir):
    logging.error('Error : Input directory ({}) with raw DICOM data not found !'.format(srcDir))
    exit(1)

logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_import_data.py")
logging.info("in: "+srcDir)
logging.info("out: "+dstDir)

    
all_files = glob.glob(os.path.normpath(srcDir+'/DICOM/**'), recursive=True)
all_files.sort()
    
series = []

for file in all_files:
    if os.path.isfile(file):
        DICOMparams = getDICOMheader(file)
        if DICOMparams.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4':
            #MR images only
            if DICOMparams.SeriesNumber not in series:
                series.append(DICOMparams.SeriesNumber)
        if DICOMparams.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
            #CT images only
            if DICOMparams.SeriesNumber not in series:
                series.append(DICOMparams.SeriesNumber)
                

logging.info("Series found: {}".format(series))


image_no = [0]*len(series)

for file in all_files:
    if os.path.isfile(file):
        DICOMparams = getDICOMheader(file)
        logging.info("Processing file {}".format(file))
        if DICOMparams.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4':
            #MR images only
            series_number = series.index(DICOMparams.SeriesNumber)
            seriesDir = os.path.normpath(dstDir+'/'+"{:06d}".format(series_number))
            try:
                if not os.path.isdir(seriesDir):
                    pathlib.Path(seriesDir).mkdir(mode=0o775, parents=True, exist_ok=True)
            except Exception as err:
                logging.error('creating "%s" directory failed, error "%s"'%(seriesDir,err))
                sys.exit(1)
            try:
                shutil.copy(file, seriesDir+'/'+"{:06d}".format(image_no[series_number]))
                logging.info("Copying file {} to {}".format(file, seriesDir+'/'+"{:06d}".format(image_no[series_number])))
            except Exception as err:
                logging.error('Failed copying file {} to {}, error {}'.format(file, seriesDir+'/'+"{:06d}".format(image_no[series_number]), err))
                sys.exit(1)
            image_no[series_number] += 1
            
        if DICOMparams.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
            #CT images only
            series_number = series.index(DICOMparams.SeriesNumber)
            seriesDir = os.path.normpath(dstDir+'/'+"{:06d}".format(series_number))
            try:
                if not os.path.isdir(seriesDir):
                    pathlib.Path(seriesDir).mkdir(mode=0o775, parents=True, exist_ok=True)
            except Exception as err:
                logging.error('creating "%s" directory failed, error "%s"'%(seriesDir,err))
                sys.exit(1)
            try:
                shutil.copy(file, seriesDir+'/'+"{:06d}".format(image_no[series_number]))
                logging.info("Copying file {} to {}".format(file, seriesDir+'/'+"{:06d}".format(image_no[series_number])))
            except Exception as err:
                logging.error('Failed copying file {} to {}, error {}'.format(file, seriesDir+'/'+"{:06d}".format(image_no[series_number]), err))
                sys.exit(1)
            image_no[series_number] += 1
