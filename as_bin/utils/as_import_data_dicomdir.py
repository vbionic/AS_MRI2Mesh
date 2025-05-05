import sys, getopt
import pydicom
import json
from pydicom.datadict import keyword_for_tag
from pydicom.filereader import read_dicomdir
import os
import pathlib
from argparse import ArgumentParser
import glob
import math
import logging
import re
import shutil

# File not used now. It extracts the images from dicom files based on DICOMDIR file. It is, unfortunately, not included in every scan, therefore this method is far from being universal.
# If using this method, the directory with data needs to retain its original name
#
# File left just in case there is a need to use it again some time.





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
logging.info("START:     as_import_data_dicomdir.py")
logging.info("in: "+srcDir)
logging.info("out: "+dstDir)

all_files = glob.glob(srcDir+'/**', recursive=True)
all_files_casefold = [f.casefold() for f in all_files]
    
filepath = os.path.normpath(srcDir+'/DICOMDIR')
filepath = all_files[all_files_casefold.index(filepath.casefold())]

logging.info("DICOMDIR file: {},processing".format(filepath))

    
    
dicom_dir = read_dicomdir(filepath)
base_dir = os.path.dirname(filepath)

if len(dicom_dir.patient_records) != 1:
    logging.error('Error : Expected ONE patient data, found {}'.format(len(dicom_dir.patient_records)))
    exit(1)

series_no = 1

for patient_record in dicom_dir.patient_records:
    if (hasattr(patient_record, 'PatientID') and hasattr(patient_record, 'PatientName')):
        logging.info("Patient: {}: {}".format(patient_record.PatientID, patient_record.PatientName))
    studies = patient_record.children
    for study in studies:
        
        #logging.info(" " * 4 + "Study {}: {}: {}".format(study.StudyID, study.StudyDate, study.StudyDescription))
        print("\n\n\n\n\n\n\n\n{}\n\n\n\n\n\n\n\n".format(study))
        all_series = study.children
        for series in all_series:
            print("\n\n\n\n\n\n\n\n{}\n\n\n\n\n\n\n\n".format(series))
            image_no = 1
            seriesDir = os.path.normpath(dstDir+'/'+"{:06d}".format(series_no))
            series_no = series_no + 1
            try:
                if not os.path.isdir(seriesDir):
                    pathlib.Path(seriesDir).mkdir(mode=0o775, parents=True, exist_ok=True)
            except Exception as err:
                logging.error('INFO      > creating "%s" directory failed, error "%s"'%(seriesDir,err))
                sys.exit(1)
            image_count = len(series.children)
            #logging.info(" " * 8 + "Series {}: {}: {} ({} images)".format(series.SeriesNumber, series.Modality, series.SeriesDescription,image_count))
            #logging.info(" " * 12 + "Reading images...")
            image_records = series.children

            for image_rec in image_records:
                logging.info(image_rec.ReferencedFileID)
                fileNameSrc = os.path.normpath(base_dir+'/'+os.path.join(*image_rec.ReferencedFileID))
                fileNameSrc = all_files[all_files_casefold.index(fileNameSrc.casefold())]
                try:
                    shutil.copy(fileNameSrc, seriesDir+'/'+"{:06d}".format(image_no))
                except Exception as err:
                    #logging.error('Failed copying file {} to {}, error {}'.format(base_dir+'/'+os.path.join(*image_rec.ReferencedFileID), seriesDir+'/'+"{:06d}".format(image_no), err))
                    logging.error('Failed copying file {} to {}, error {}'.format(fileNameSrc, seriesDir+'/'+"{:06d}".format(image_no), err))
                    sys.exit(1)
                image_no = image_no + 1
