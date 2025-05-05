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
import logging
import re

#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_json import *
#-----------------------------------------------------------------------------------------
    
chosenPatientPosition = None
chosenImageOrientation = None
chosenImageDimension = None
chosenPixelSpacing = None
chosenImageNames = []
    
def process_dir_gather_data(dir, outputdir, verbose):
    global chosenImageOrientation
    global chosenPatientPosition
    global chosenImageDimension
    global chosenPixelSpacing
    logging.info(verbose)
    try:
        if not os.path.isdir(outputdir):
            pathlib.Path(outputdir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error("Output dir IO error: {}".format(err))
        sys.exit(1)

    inputfiles = glob.glob(dir + '/**', recursive=True)
    inputfiles.sort()

    logging.info("Scanning: "+dir + '/*')
    DICOMparams = {}
    
    
    PatientPosition_values = []
    image_orientation = []
    image_dimension = []
    image_names = []
    pixel_spacings = []
    
    for file in inputfiles:
        if os.path.isfile(file):
            logging.info("        Dicom check gathering data: "+file)
            image_names.append(os.path.basename(file))
            DICOMparams = getDICOMheader(file)
            u_axis = [DICOMparams.ImageOrientationPatient[0], DICOMparams.ImageOrientationPatient[1], DICOMparams.ImageOrientationPatient[2]]
            v_axis = [DICOMparams.ImageOrientationPatient[3], DICOMparams.ImageOrientationPatient[4], DICOMparams.ImageOrientationPatient[5]]
            
            if hasattr(DICOMparams, "PatientPosition"):
                PatientPosition_values.append(DICOMparams.PatientPosition)
            else:
                PatientPosition_values.append("HFS")

            image_orientation.append(tuple(u_axis + v_axis))
            image_dimension.append(tuple([DICOMparams.Rows, DICOMparams.Columns]))
            pixel_spacings.append(tuple([DICOMparams.PixelSpacing[0], DICOMparams.PixelSpacing[1]]))
    
    
    max_dist = 0.1
    chosenImageOrientation = find_best_vec(image_orientation, max_dist, verbose)
    chosenPatientPosition = find_most_freq_str(PatientPosition_values, verbose)
    chosenImageDimension = find_best_vec(image_dimension, 1, verbose)
    chosenPixelSpacing = find_best_vec(pixel_spacings,1,verbose)
    


def process_dir(dir, outputdir, invertDir, forceFlip, verbose):

    try:
        if not os.path.isdir(outputdir):
            pathlib.Path(outputdir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error("Output dir IO error: {}".format(err))
        sys.exit(1)
        
    sorted_files                = outputdir+"/"+"sorted.json"
    sorted_files_old_names      = outputdir+"/"+"sorted_old_names.json"
    geometry_data               = outputdir+"/"+"set_data.json"

    inputfiles                  = glob.glob(dir + '/**', recursive = True)
    inputfiles.sort()
    
    logging.info("Scanning: "+dir + '/*')
    
    
    DICOMparams                 = {}
    
    prev_ImagePositionPatient   = []
    prev_projection             = 0
    
    PatientPosition_values      = []
    distance_between_slices     = []
    image_orientation           = []
    image_z_coord               = []
    image_coord                 = []
    image_dimension             = []
    image_names                 = []
    pixel_spacings              = []
    projections                 = []
    is_CT_scan                  = False
    
    scanType = 0
    #scanType = 0 oznacza nieznany skan, 1 oznacza plasterki poprzeczne reki
    
    for file in inputfiles:
        if os.path.isfile(file):
            logging.info("        Dicom check processing: "+file)
            image_names.append(os.path.basename(file))
            DICOMparams = getDICOMheader(file)
            u_axis = [DICOMparams.ImageOrientationPatient[0], DICOMparams.ImageOrientationPatient[1], DICOMparams.ImageOrientationPatient[2]]
            v_axis = [DICOMparams.ImageOrientationPatient[3], DICOMparams.ImageOrientationPatient[4], DICOMparams.ImageOrientationPatient[5]]
            photo_axis = np.cross(u_axis,v_axis)
            logging.info("Photo axis: {}".format(photo_axis))
            if (np.max(abs(photo_axis))==abs(photo_axis)[2]):
                #wspolrzedna Z ma najwieksza wart bezwzgledna - prawdopodobnie poprzeczne obrazki reki
                scanType = 1
                logging.info("Probably axial scan")
            
            if hasattr(DICOMparams, "PatientPosition"):
                PatientPosition_values.append(DICOMparams.PatientPosition)
            else:
                PatientPosition_values.append("HFS")
            
            curr_ImagePositionPatient = [DICOMparams.ImagePositionPatient[0], DICOMparams.ImagePositionPatient[1], DICOMparams.ImagePositionPatient[2]]
            
            projection = np.dot(curr_ImagePositionPatient, photo_axis)/np.linalg.norm(photo_axis)
            projections.append(projection)
            image_orientation.append(tuple(u_axis + v_axis))
            image_dimension.append(tuple([DICOMparams.Rows, DICOMparams.Columns]))
            pixel_spacings.append(tuple([DICOMparams.PixelSpacing[0], DICOMparams.PixelSpacing[1]]))
            image_z_coord.append(DICOMparams.ImagePositionPatient[2])
            image_coord.append([DICOMparams.ImagePositionPatient[0], DICOMparams.ImagePositionPatient[1], DICOMparams.ImagePositionPatient[2]])
            logging.info("Photo position: {}".format([DICOMparams.ImagePositionPatient[0], DICOMparams.ImagePositionPatient[1], DICOMparams.ImagePositionPatient[2]]))
            try:
                is_CT_scan = (DICOMparams.Modality == 'CT')
                if is_CT_scan:
                    logging.info("This is CT scan")
            except:
                is_CT_scan = False
            
    
    #test image_dimensions
    max_dist = 0.1
    for num in range(0, len(image_names)):
        if chosenImageDimension == image_dimension[num]:
            if chosenPatientPosition == PatientPosition_values[num]:
                if chosenPixelSpacing == pixel_spacings[num]:
                    if calc_vec_dif(chosenImageOrientation, image_orientation[num]) < max_dist:
                        chosenImageNames.append(tuple([projections[num], image_names[num], image_z_coord[num], image_coord[num]]))

    #prepare list of files
    
    if verbose: logging.info(chosenImageNames)
    logging.info('Patient Position: {}'.format(chosenPatientPosition[0:2]))
    if scanType == 1:
        if invertDir == "True":
            if chosenPatientPosition[0:2] == 'HF':
                #for "head first" position, the end of the stump will have the biggest Z value, therefore we sort in descending order
                chosenImageNames.sort(key=lambda para: para[2])
            elif chosenPatientPosition[0:2] == 'FF':
                #for "feet first" position, the end of the stump will have the smallest Z value, therefore we sort in ascending order
                chosenImageNames.sort(key=lambda para: -para[2])
            else:
                logging.error("---!!!---                unknown patient orientation                ---!!!---")
                #no idea what this is, but still needs to be sorted
                chosenImageNames.sort(key=lambda para: para[2])
        else:
            if chosenPatientPosition[0:2] == 'HF':
                #for "head first" position, the end of the stump will have the biggest Z value, therefore we sort in descending order
                chosenImageNames.sort(key=lambda para: -para[2])
            elif chosenPatientPosition[0:2] == 'FF':
                #for "feet first" position, the end of the stump will have the smallest Z value, therefore we sort in ascending order
                chosenImageNames.sort(key=lambda para: para[2])
            else:
                logging.error("---!!!---                unknown patient orientation                ---!!!---")
                #no idea what this is, but still needs to be sorted
                chosenImageNames.sort(key=lambda para: -para[2])
    elif scanType == 0:
        if invertDir == "True":
            #not the transversal scan - we sort with descending X
            chosenImageNames.sort(key=lambda para: -para[0])
        else:
            #not the transversal scan - we sort with ascending X
            chosenImageNames.sort(key=lambda para: para[0])
    
    sorted_files_list = [p[1] for p in chosenImageNames]
    
    #sort the z coordinates for logging purposes
    image_z_coord.sort()
    image_coord.sort(key=lambda coord: coord[2])
    
    first = True
    
    for dataslice in chosenImageNames:
        if first:
            first = False
            prev_projection = dataslice[0]
        else:
            distance_between_slices.append(dataslice[0]-prev_projection)
            prev_projection = dataslice[0]
    
    if verbose: logging.info(sorted_files_list)
    
    #zmiana nazw, zeby rosly w kolejnosci od dloni do barku
    numer = 1
    order_inverted = False
    new_sorted_files_list = []
    for filename in sorted_files_list:
        nowanazwa = filename+"___"
        if nowanazwa != filename:
            logging.info("Renaming file: {} to {}".format(dir+'/'+filename, dir+'/'+nowanazwa))
            if os.path.exists(dir+'/'+nowanazwa):
                logging.error("File exists: {}".format(dir+'/'+nowanazwa))
                sys.exit(1)
            try:
                logging.info('Renaming {} to {}'.format( dir+'/'+filename,  dir+'/'+nowanazwa))
                os.rename(dir+'/'+filename, dir+'/'+nowanazwa)
            except Exception as err:
                logging.error("Failed to rename source file: {}".format(err))
                sys.exit(1)
    
    for filename in sorted_files_list:
        filename = filename + "___"
        nowanazwa = '{:08d}'.format(numer)
        new_sorted_files_list.append(nowanazwa)
        logging.info('New name: {}'.format(nowanazwa))
        numer = numer + 1
        if nowanazwa != filename:
            order_inverted = True
            logging.info("Renaming file: {} to {}".format(dir+'/'+filename, dir+'/'+nowanazwa))
            if os.path.exists(dir+'/'+nowanazwa):
                logging.error("File exists: {}".format(dir+'/'+nowanazwa))
                sys.exit(1)
            try:
                #logging.info('Renaming {} to {}'.format( os.path.realpath(file),  os.path.realpath(filelocation+'/'+nowanazwa)))
                logging.info('Renaming {} to {}'.format( dir+'/'+filename,  dir+'/'+nowanazwa))
                #os.rename(os.path.realpath(file), os.path.realpath(nowanazwa))
                os.rename(dir+'/'+filename, dir+'/'+nowanazwa)
            except Exception as err:
                logging.error("Failed to rename source file: {}".format(err))
                sys.exit(1)
    
    
    
    if len(distance_between_slices) == 0:
        distance_between_slices = [1]
        
        
    best_distance_between_slices = abs(find_best_val(distance_between_slices, 0.01, verbose))
    
    flipped = False
    
    if forceFlip:
        flipped = True
    
    if scanType == 1:
        #for transversal scans chceck whether the images need to be flipped
        if photo_axis[2]<0:
            if chosenPatientPosition[0:2] == 'FF':
                flipped = True
                logging.info("Images need to be flipped.")
        else:
            if chosenPatientPosition[0:2] == 'HF':
                flipped = True
                logging.info("Images need to be flipped.")
            
        
        
        
    try:
        jsonfile = open(os.open(sorted_files, os.O_CREAT | os.O_WRONLY, 0o775),'w')
        json.dump({"sorted": new_sorted_files_list, "flipped":flipped},jsonfile, indent=4)
        jsonfile.close()
    except Exception as err:
        logging.error("Output JSON file IO error: {}".format(err))
        sys.exit(1)   
        
    try:
        jsonfile = open(os.open(sorted_files_old_names, os.O_CREAT | os.O_WRONLY, 0o775),'w')
        json.dump({"sorted": sorted_files_list},jsonfile, indent=4)
        jsonfile.close()
    except Exception as err:
        logging.error("Output JSON file IO error: {}".format(err))
        sys.exit(1)   
    try:
        #jsonfile = open(os.open(geometry_data, os.O_CREAT | os.O_WRONLY, 0o775),'w')
        #json.dump({"pixel_spacing_x": chosenPixelSpacing[0], "pixel_spacing_y": chosenPixelSpacing[1], "distance_between_slices":best_distance_between_slices},jsonfile, indent=4)
        #jsonfile.close()
        jsonUpdate(geometry_data, {"is_CT_scan":is_CT_scan, "pixel_spacing_x": chosenPixelSpacing[0], "pixel_spacing_y": chosenPixelSpacing[1], "distance_between_slices":best_distance_between_slices, "look_direction":photo_axis.tolist(), "projection":np.dot(np.array(chosenImageNames[-1][3])-np.array(chosenImageNames[0][3]), photo_axis)/np.linalg.norm(photo_axis), "patient_position":chosenPatientPosition, "coord_first":chosenImageNames[0][3], "coord_last":chosenImageNames[-1][3]})
    except Exception as err:
        logging.error("Output JSON file IO error: {}".format(err))
        sys.exit(1)   



#-----------------------------------------------------------------------------------------------------

def find_best_vec(a, max_dist, verbose):
    num_in_vicinity = []
    for i in a:
        if verbose: logging.info(i)
        count = 0
        for j in a:
            if calc_vec_dif(i,j) <= max_dist:
                count = count + 1
        num_in_vicinity.append(count)
    if verbose: logging.info(num_in_vicinity)
    best = a[num_in_vicinity.index(max(num_in_vicinity))]
    if verbose: logging.info(best)
    return best
    
def find_best_val(a, max_dist, verbose):
    num_in_vicinity = []
    for i in a:
        if verbose: logging.info(i)
        count = 0
        for j in a:
            if abs(i-j) <= max_dist:
                count = count + 1
        num_in_vicinity.append(count)
    if verbose: logging.info(num_in_vicinity)
    best = a[num_in_vicinity.index(max(num_in_vicinity))]
    if verbose: logging.info(best)
    return best
    
def find_most_freq_str(a, verbose):
    num_equal = []
    for i in a:
        if verbose: logging.info(i)
        count = 0
        for j in a:
            if i==j:
                count = count + 1
        num_equal.append(count)
    if verbose: logging.info(num_equal)
    best = a[num_equal.index(max(num_equal))]
    if verbose: logging.info(best)
    return best
    
    
def calc_vec_dif(a,b):
    res = 0
    for i in range(len(a)):
        res = res + (a[i] - b[i])**2
    return math.sqrt(res)
 
def sub_vec(a,b):
    res =[]
    for i in range(len(a)):
        res.append(a[i] - b[i])
    return res
 
def vec_len(a):
    sum = 0
    for i in a:
        sum = sum + i*i
    return math.sqrt(sum)
  
def getDICOMheader(inputfile):
    try:        
        dataset = pydicom.dcmread(inputfile)
    except Exception as err:
        logging.error("Input file IO error: {} for file {}".format(err,inputfile))
        sys.exit(1)
    return dataset
    
#-----------------------------------------------------------------------------------------

parser = ArgumentParser()

parser.add_argument("-idDir",  "--id_dir",     dest="id_dir",  help="input dicom directory",       metavar="PATH",required=True)
parser.add_argument("-ckDir",  "--ck_dir",     dest="ck_dir",  help="destination directory", metavar="PATH",required=True)
parser.add_argument("-invD",  "--invertDir",     dest="invertDir",  help="invert file order", metavar="PATH",required=True)
parser.add_argument("-fF",  "--forceFlip",     dest="forceFlip",  help="force image flip", metavar="PATH",required=True)
parser.add_argument("-v",      "--verbose",    dest="verbose", help="verbose level",                              required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
idDir  	= args.id_dir
ckDir  	= args.ck_dir
invertDir = args.invertDir
forceFlip = (args.forceFlip == "True")

idDir = os.path.normpath(idDir)
ckDir = os.path.normpath(ckDir)


try:
	if not os.path.isdir(ckDir):
	    pathlib.Path(ckDir).mkdir(mode=0o775, parents=True, exist_ok=True)
except Exception as err:
	print("Output dir IO error: {}".format(err))
	sys.exit(1)
    
logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(ckDir+"/dicom_check.log",mode='w'),logging.StreamHandler(sys.stdout)])

if not os.path.isdir(idDir):
    logging.error('Error : Input directory (%s) with DICOM files not found !',idDir)
    exit(1)
    
# logging.info('-----------------------------------------------------')
# logging.info('current dir   : %s'%os.getcwd())
# logging.info('verbose level : %s'%verbose)
# logging.info('subdirectory  :')
# logging.info('    input dicoms               : {}'.format(idDir))
# logging.info('    output images and metadata : {}'.format(ckDir))
# logging.info('-----------------------------------------------------')


logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_dicom_check.py")
logging.info("in: "+idDir)
logging.info("out: "+ckDir)


#
process_dir_gather_data(idDir, ckDir, verbose=='on')
process_dir(idDir, ckDir, invertDir, forceFlip, verbose=='on')
#-----------------------------------------------------------------------------------------
