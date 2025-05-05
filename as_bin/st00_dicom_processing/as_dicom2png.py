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
#import png
import logging

#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_json import *
#-----------------------------------------------------------------------------------------

max_set = 0
min_set = 100000

#----------------------------------------------------------------------------------------------------------------

#https://github.com/pydicom/pydicom/issues/319
def _JSON_translate(val):
    t = type(val)
    if t in (list, int, float, str):
        conv_val = val
    elif t == pydicom.multival.MultiValue:
        conv_val = list(val)
    elif t == pydicom.valuerep.DSfloat:
        conv_val = float(val)
    elif t == pydicom.valuerep.IS:
        conv_val = int(val)
    elif t == pydicom.valuerep.PersonName:
        conv_val = { "Name":str(val), "FamilyName":val.family_name, "GivenName":val.given_name, "Ideographic":val.ideographic, "MiddleName":val.middle_name, "NamePrefix":val.name_prefix, "NameSuffix":val.name_suffix, "Phonetic":val.phonetic}
    else:
        conv_val = repr(val)
    return conv_val
    

#----------------------------------------------------------------------------------------------------------------

def process_dir(inputdir, inputfiles, outputdir, flipped, min_no, max_no, cropcenterx, cropcentery, cropradius, cropx1, cropy1, cropx2, cropy2, verbose):
    for file in inputfiles:
        
        filenumber = int(file)
        
        if filenumber < min_no:
            logging.info(f"Skipping file {file}")
            continue
        if (max_no > 0) and (filenumber > max_no):
            logging.info(f"Skipping file {file}")
            continue
        
        file = inputdir + '/' +file
        file.replace("\\",'/')
        
        logging.info('Processing file       :%s '% file)
        process_file(file, outputdir, flipped, cropcenterx, cropcentery, cropradius, cropx1, cropy1, cropx2, cropy2)

#----------------------------------------------------------------------------------------------------------------

def process_dir_init(inputdir, inputfiles, outputdir, min_no, max_no, verbose):
    global max_set
    global min_set
    max_set = 0
    min_set = 100000
    name = outputdir+"/set_data.json"
    for file in inputfiles:
        filenumber = int(file)
        
        if filenumber < min_no:
            logging.info(f"Skipping file {file}")
            continue
        if (max_no > 0) and (filenumber > max_no):
            logging.info(f"Skipping file {file}")
            continue
    
    
        file = inputdir + '/' +file
        file.replace("\\",'/')
        [max_file, min_file] = process_file_init(file, outputdir)
        name_file = outputdir+"/"+file.replace('\\','/').rsplit("/",1)[1].rsplit(".",1)[0]+"_data.json"
        try:
            jsonUpdate(name_file, {'max_file': int(max_file), 'min_file': int(min_file)})
        except Exception as err:
            logging.error('Output file json data file IO error: {}'.format(err))
            sys.exit(1)
        if max_file > max_set:
            max_set = max_file
        if min_file < min_set:
            min_set = min_file
    try:
        jsonUpdate(name, {'input dicom dir': inputdir, 'max_set': int(max_set), 'min_set': int(min_set)})
    except Exception as err:
        logging.error('Output file json max file IO error: {}'.format(err))
        sys.exit(1)

#----------------------------------------------------------------------------------------------------------------

def process_file_init(inputfile, outputdir):
    data_description = {}
    try:        
        dataset = pydicom.dcmread(inputfile)
    except Exception as err:
        logging.error('Input file IO error: {}'.format(err))
        sys.exit(1)
    a = dataset.pixel_array
    return [a.max(), a.min()]

#----------------------------------------------------------------------------------------------------------------

def process_file(inputfile, outputdir, flipped, cropcenterx, cropcentery, cropradius, cropx1, cropy1, cropx2, cropy2):
    data_description = {}
    _3D_coord = []

    try:        
        dataset = pydicom.dcmread(inputfile)
    except Exception as err:
        logging.error('Input file IO error: {}'.format(err))
        sys.exit(1)

    a = dataset.pixel_array    
    
    if flipped:
        #the image needs to be flipped, so that the 3D model in not flipped inside-out
        a = a[:,[i for i in range(dataset.Columns-1,-1,-1)]]
        
    #crop the image if needed
    if (cropcenterx is not None) and (cropcentery is not None) and (cropradius is not None):
        for ii in range(dataset.Columns):
            for jj in range(dataset.Rows):
                if (((ii-cropcenterx)**2+(jj-cropcentery)**2)>cropradius**2):
                    a[jj,ii] = 0
    if (cropx1 is not None) and (cropy1 is not None) and (cropx2 is not None) and (cropy2 is not None):
        for ii in range(dataset.Columns):
            for jj in range(dataset.Rows):
                if ((ii<cropx1) or (ii>cropx2)) or ((jj<cropy1) or (jj>cropy2)):
                    a[jj,ii] = 0
        
    try:
        if not os.path.isdir(outputdir):
            pathlib.Path(outputdir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('Output file IO error: {}'.format(err))
        sys.exit(1)
        
    name1 	= outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit(".",1)[0]+"_fsi.png"
    name2_l	= outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit(".",1)[0]+"_lsi.png"
    name2_g	= outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit(".",1)[0]+"_gsi.png"
    name3 	= outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit(".",1)[0]+"_dicom.json"
    name4 	= outputdir+"/"+inputfile.replace('\\','/').rsplit("/",1)[1].rsplit(".",1)[0]+"_3Dcoord.json"

    try:
        Image.fromarray(a.astype(np.uint16),mode='I;16').save(name1)
    except Exception as err:
        logging.error('Output file 16 bit IO error: {}'.format(err))
        sys.exit(2)

    a = a.astype(np.float32)

    b = np.array(a, copy=True)

    a = a - a.min()
    if a.max()>0:
        a = (a*255)/a.max()

    #logging.info('Luma [MIN,MAX] : {} {}'.format(min_set, max_set))

    b = b - min_set
    if max_set>0:
        b = (b*255)/(max_set - min_set)

    image2_l = np.array(a, copy=True, dtype=np.uint8)
    image2_g = np.array(b, copy=True, dtype=np.uint8)

    #try:
        #Image.fromarray(image2_l).save(name2_l)
        #Image.fromarray(image2_g).save(name2_g)

    #except Exception as err:
    #    logging.error('Output file 8 bit IO error: {}'.format(err))
    #    sys.exit(1)

    unknown_idx = 0    

    for entry in dataset:
    #    print(entry)
        # print("<<-----------------------------")
        if entry.tag == (0x7fe0, 0x0010):
            continue
        # print(entry.tag)
        # print(keyword_for_tag(entry.tag))
        # print(entry.VR)
        # print(entry.VM)
        # print(entry.value)
        # print(type(entry.value))
        tag_name = keyword_for_tag(entry.tag)
        if tag_name == "":
            tag_name = "Unknown{:d}".format(unknown_idx)
            unknown_idx += 1
        #    print(tag_name)
            JSON_translated = _JSON_translate(entry.value)
            data_description[tag_name] = JSON_translated
        else:
        #    print(tag_name)
            JSON_translated = _JSON_translate(entry.value)
            data_description[tag_name] = JSON_translated
        #    print(type(JSON_translated))
        #print("----------------------------->>")

    
    if flipped:
        #the coordinates and the orientation of the image must be adjusted to be accurate after the image was flipped
        data_description["ImagePositionPatient"][0] = data_description["ImagePositionPatient"][0] + (data_description["Columns"]-1)*data_description["PixelSpacing"][0]*data_description["ImageOrientationPatient"][0]
        data_description["ImagePositionPatient"][1] = data_description["ImagePositionPatient"][1] + (data_description["Columns"]-1)*data_description["PixelSpacing"][0]*data_description["ImageOrientationPatient"][1]
        data_description["ImagePositionPatient"][2] = data_description["ImagePositionPatient"][2] + (data_description["Columns"]-1)*data_description["PixelSpacing"][0]*data_description["ImageOrientationPatient"][2]
        data_description["ImageOrientationPatient"][0] = -data_description["ImageOrientationPatient"][0]
        data_description["ImageOrientationPatient"][1] = -data_description["ImageOrientationPatient"][1]
        data_description["ImageOrientationPatient"][2] = -data_description["ImageOrientationPatient"][2]
    
    try:
        jsonfile = open(os.open(name3, os.O_CREAT | os.O_WRONLY, 0o775), 'w')
        json.dump(data_description,jsonfile, indent=4)
        jsonfile.close()
    except Exception as err:
        logging.error('Output JSON file IO error: {}'.format(err))
        sys.exit(1)   


#-----------------------------------------------------------------------------------------

parser = ArgumentParser()

parser.add_argument("-idDir",  "--id_dir",     dest="id_dir",  help="input dicom directory",       metavar="PATH",required=True)
parser.add_argument("-dpDir",  "--dp_dir",     dest="dp_dir",  help="output png directory", metavar="PATH",required=True)
parser.add_argument("-lsDir",  "--ls_dir",     dest="ls_dir",  help="input list directory", metavar="PATH",required=True)
parser.add_argument("-min",    "--min_no",     dest="min_no",  help="lower number of file to process", metavar="PATH",required=True)
parser.add_argument("-max",    "--max_no",     dest="max_no",  help="upper number of file to process", metavar="PATH",required=True)
parser.add_argument("-ccx",    "--cropcenterx",    dest="cropcenterx", help="center of cropping area x",  required=False)
parser.add_argument("-ccy",    "--cropcentery",    dest="cropcentery", help="center of cropping area y",  required=False)
parser.add_argument("-cr",     "--cropradius",    dest="cropradius", help="radius of cropping area",  required=False)
parser.add_argument("-cx1",    "--cropx1",    dest="cropx1", help="x coords of top left crop corner",  required=False)
parser.add_argument("-cx2",    "--cropx2",    dest="cropx2", help="x coords of bottom right crop corner",  required=False)
parser.add_argument("-cy1",    "--cropy1",    dest="cropy1", help="y coords of top left crop corner",  required=False)
parser.add_argument("-cy2",    "--cropy2",    dest="cropy2", help="y coords of bottom right crop corner",  required=False)

parser.add_argument("-v",      "--verbose",    dest="verbose", help="verbose level",                              required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
cropcenterx = None        if args.cropcenterx is None else int(args.cropcenterx)
cropcentery = None        if args.cropcentery is None else int(args.cropcentery)
cropradius = None        if args.cropradius is None else int(args.cropradius)
cropx1 = None        if args.cropx1 is None else int(args.cropx1)
cropx2 = None        if args.cropx2 is None else int(args.cropx2)
cropy1 = None        if args.cropy1 is None else int(args.cropy1)
cropy2 = None        if args.cropy2 is None else int(args.cropy2)

idDir  	= args.id_dir
dpDir  	= args.dp_dir
lsDir   = args.ls_dir
min_no  = int(args.min_no)
max_no  = int(args.max_no)


try:
	if not os.path.isdir(dpDir):
		pathlib.Path(dpDir).mkdir(mode=0o775, parents=True, exist_ok=True)
except Exception as err:
	print('Error: Output dir IO error: {}'.format(err))
	sys.exit(1)
    
logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(dpDir+"/dicom2png.log",mode='w'),logging.StreamHandler(sys.stdout)])


# logging.info('-----------------------------------------------------')
# logging.info('input dicom dir       :%s '% idDir)
# logging.info('output png dir        :%s '% dpDir)
# logging.info('output checklist dir  :%s '% lsDir)
# logging.info('Verbose level         :%s '% verbose)
# logging.info('-----------------------------------------------------')

logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_dicom2png.py")
logging.info("in: "+idDir)
logging.info("list: "+lsDir)
logging.info("out: "+dpDir)

if not os.path.isdir(idDir):
	logging.error('Input directory (%s) with DICOM files not found !'%iiDir)
	exit(1)

try:
	filelist_file = open(lsDir+"/sorted.json")
	filelist = json.load(filelist_file)
	filelist_file.close()
except Exception as err:
	logging.error('File list IO error: {}'.format(err))
	sys.exit(1)

process_dir_init(idDir, filelist["sorted"], dpDir, min_no, max_no, verbose)
process_dir(idDir, filelist["sorted"], dpDir, filelist["flipped"], min_no, max_no, cropcenterx, cropcentery, cropradius, cropx1, cropy1, cropx2, cropy2, verbose)
    
#-----------------------------------------------------------------------------------------
