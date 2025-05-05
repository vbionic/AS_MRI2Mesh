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

#-----------------------------------------------------------------------------------
def getDICOMheader_noexit(inputfile):
    try:        
        dataset = pydicom.dcmread(inputfile)
    except Exception as err:
        return None
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

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(dstDir+"/../explodeDicom.log",mode='w'),logging.StreamHandler(sys.stdout)])
#logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.StreamHandler(sys.stdout)])

if not os.path.isdir(srcDir):
    logging.error('Error : Input directory ({}) with raw DICOM data not found !'.format(srcDir))
    exit(1)

logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_import_data_explode_dicom.py")
logging.info("in: "+srcDir)
logging.info("out: "+dstDir)


all_files = glob.glob(os.path.normpath(srcDir+'/**'), recursive=True)
all_files.sort()



number = 0

for file in all_files:
    if os.path.isfile(file):
        logging.info(file)
        DICOMparams = getDICOMheader_noexit(file)
        if DICOMparams is not None:
            try:
                if DICOMparams.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4.1':
                    #Enchanced MR images only for MR
                    logging.info("File found: {} with {} images".format(file, DICOMparams.NumberOfFrames))
                    #seriesDir = os.path.normpath(dstDir+'/'+"{:06d}".format(series_number))
                    for NumerObrazu in range(0,DICOMparams.NumberOfFrames):
                        DICOMparamsCopy = getDICOMheader(file)
                        DICOMparamsCopy.NumberOfFrames = 1
                        DICOMparamsCopy.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
                        temp1 = DICOMparams[0x5200,0x9229].value[0][0x0018,0x9112].value[0]
                        #print(temp1)
                        for DataElem in temp1:
                            #print(DataElem)
                            DICOMparamsCopy.add(DataElem)
                        temp1 = DICOMparams[0x5200,0x9230].value[NumerObrazu][0x0018,0x9114].value[0]
                        #print(temp1)
                        for DataElem in temp1:
                            #print(DataElem)
                            DICOMparamsCopy.add(DataElem)
                        temp1 = DICOMparams[0x5200,0x9230].value[NumerObrazu][0x0018,0x9226].value[0]
                        #print(temp1)
                        for DataElem in temp1:
                            #print(DataElem)
                            DICOMparamsCopy.add(DataElem)
                        temp1 = DICOMparams[0x5200,0x9230].value[NumerObrazu][0x0020,0x9111].value[0]
                        #print(temp1)
                        for DataElem in temp1:
                            #print(DataElem)
                            DICOMparamsCopy.add(DataElem)
                        temp1 = DICOMparams[0x5200,0x9230].value[NumerObrazu][0x0020,0x9113].value[0]
                        #print(temp1)
                        for DataElem in temp1:
                            #print(DataElem)
                            DICOMparamsCopy.add(DataElem)
                        temp1 = DICOMparams[0x5200,0x9230].value[NumerObrazu][0x0020,0x9116].value[0]
                        #print(temp1)
                        for DataElem in temp1:
                            #print(DataElem)
                            DICOMparamsCopy.add(DataElem)
                        temp1 = DICOMparams[0x5200,0x9230].value[NumerObrazu][0x0028,0x9110].value[0]
                        #print(temp1)
                        for DataElem in temp1:
                            #print(DataElem)
                            DICOMparamsCopy.add(DataElem)
                        temp1 = DICOMparams[0x5200,0x9230].value[NumerObrazu][0x0028,0x9132].value[0]
                        #print(temp1)
                        for DataElem in temp1:
                            #print(DataElem)
                            DICOMparamsCopy.add(DataElem)
                        temp1 = DICOMparams[0x5200,0x9230].value[NumerObrazu][0x0028,0x9145].value[0]
                        #print(temp1)
                        for DataElem in temp1:
                            #print(DataElem)
                            DICOMparamsCopy.add(DataElem)
                        #DICOMparamsCopy[0x0018,0x0080].value
                        #print(DICOMparamsCopy[0x0018,0x0080].value)
                        obraz = DICOMparams.pixel_array[NumerObrazu,:,:]
                        logging.info('Image {} has dimensions: {}'.format(NumerObrazu, obraz.shape))
                        DICOMparamsCopy.PixelData = obraz.tobytes()
                        DICOMparamsCopy.save_as(dstDir+"/wynikowy{:08d}".format(number))
                        logging.info('Saved as: {}'.format(dstDir+"/wynikowy{:08d}".format(number)))
                        number += 1
                elif DICOMparams.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4':
                    #this is an old DICOM format file for MR
                    logging.info("File found: {} with {} images".format(file, 1))
                    DICOMparams.save_as(dstDir+"/wynikowy{:08d}".format(number))
                    logging.info('Saved as: {}'.format(dstDir+"/wynikowy{:08d}".format(number)))
                    number += 1
                elif DICOMparams.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
                    #this is an old DICOM file format for CT
                    logging.info("File found: {} with {} images".format(file, 1))
                    DICOMparams.save_as(dstDir+"/wynikowy{:08d}".format(number))
                    logging.info('Saved as: {}'.format(dstDir+"/wynikowy{:08d}".format(number)))
                    number += 1
                else:
                    logging.error('Unknown DICOM file: {}'.format(file))
            except Exception as err:
                logging.error('DICOM file without SOPClassUID: {}'.format(file))
        else:
            logging.error('Not a DICOM file: {}'.format(file))
                    
                    
