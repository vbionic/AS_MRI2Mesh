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

def process_spectrum(spectrum, min_bin_no, threshold, width):
    for i in range(min_bin_no, (len(spectrum)//2)):
        if width == 3:
            average = (spectrum[i-1] + spectrum[i+1])/2
        elif width == 5:
            average = (spectrum[i-1] + spectrum[i+1] + spectrum[i-2] + spectrum[i+2])/4
        else:
            logging.error("incorrect width of the averaging window")
            exit(1)
        if ((np.abs(spectrum[i]) > threshold*np.abs(average))|(np.abs(spectrum[i]) < np.abs(average)*0.9)):
            spectrum[i] = average
            spectrum[len(spectrum) - i] = average.conjugate()
    return spectrum
        
        
def apply_corrections(corrections, iDir, oDir):
    poly_list    = glob.glob(os.path.normpath(iDir + "/*_polygons.json"))
    poly_list.sort()
    for poly_fn in poly_list:
        poly_fn_norm = os.path.basename(os.path.normpath(poly_fn))
        poly_num = int(poly_fn_norm.split('_',1)[0])
        try:
            poly_h = open(poly_fn)
            data = json.load(poly_h)
            poly_h.close()
        except Exception as err:
            logging.error("Input data IO error: {}".format(err))
            sys.exit(1)
        if(len(data["polygons"])==0):
            try:
                if not os.path.isdir(os.path.normpath(oDir)):
                    pathlib.Path(os.path.normpath(oDir)).mkdir(mode=0o775, parents=True, exist_ok=True)
            except Exception as err:
                logging.error('creating "%s" directory failed, error "%s"'%(oDir,err))
                exit(1)
            jsonDumpSafe(os.path.normpath(oDir + "/" + poly_fn_norm), data)
            continue
        
        corr_val_x = corrections[poly_num-1][1]
        corr_val_y = corrections[poly_num-1][2]
        imsize_x   = corrections[poly_num-1][3]
        imsize_y   = corrections[poly_num-1][4]
        
        for polygon in data["polygons"]:
            for i in range(0, len(polygon["outer"]["path"])):
                polygon["outer"]["path"][i][0] = polygon["outer"]["path"][i][0] + corr_val_x
                polygon["outer"]["path"][i][1] = polygon["outer"]["path"][i][1] + corr_val_y
            polygon["outer"]["box"][0] = polygon["outer"]["box"][0] + corr_val_x
            polygon["outer"]["box"][1] = polygon["outer"]["box"][1] + corr_val_y
            polygon["outer"]["box"][2] = polygon["outer"]["box"][2] + corr_val_x
            polygon["outer"]["box"][3] = polygon["outer"]["box"][3] + corr_val_y
            for inner in polygon["inners"]:
                for i in range(0, len(inner["path"])):
                    inner["path"][i][0] = inner["path"][i][0] + corr_val_x
                    inner["path"][i][1] = inner["path"][i][1] + corr_val_y
                inner["box"][0] = inner["box"][0] + corr_val_x
                inner["box"][1] = inner["box"][1] + corr_val_y
                inner["box"][2] = inner["box"][2] + corr_val_x
                inner["box"][3] = inner["box"][3] + corr_val_y
        data["box"][0] = data["box"][0] + corr_val_x
        data["box"][1] = data["box"][1] + corr_val_y
        data["box"][2] = data["box"][2] + corr_val_x
        data["box"][3] = data["box"][3] + corr_val_y
        
        if (data["box"][0] < 0) or (data["box"][1]<0) or (data["box"][2]>imsize_x - 1) or (data["box"][3]>imsize_y - 1):
            logging.error("The correction shift tooo large - tissue in {} goes outside the image size".format(iDir))
            exit(1)
        try:
            if not os.path.isdir(os.path.normpath(oDir)):
                pathlib.Path(os.path.normpath(oDir)).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('creating "%s" directory failed, error "%s"'%(oDir,err))
            exit(1)
        jsonDumpSafe(os.path.normpath(oDir + "/" + poly_fn_norm), data)
        
def apply_corrections_img(corrections, iDir, oDir):
    img_list    = glob.glob(os.path.normpath(iDir + "/*_labels.png"))
    img_list.sort()
    for img_fn in img_list:
        img_fn_norm = os.path.basename(os.path.normpath(img_fn))
        img_num = int(img_fn_norm.split('_',1)[0])
        corr_val_x = corrections[img_num-1][1]
        corr_val_y = corrections[img_num-1][2]
        
        img = cv.imread(img_fn, cv.IMREAD_COLOR)
        
        new_img = np.zeros((img.shape[0], img.shape[1], 3))
        
        if (corr_val_x >= 0) and (corr_val_y >= 0):
            new_img[corr_val_y:, corr_val_x:, :] = img[0:img.shape[0]-corr_val_y, 0:img.shape[1]-corr_val_x, :]
        if (corr_val_x >= 0) and (corr_val_y < 0):
            new_img[0:img.shape[0]+corr_val_y, corr_val_x:, :] = img[-corr_val_y:, 0:img.shape[1]-corr_val_x, :]
        if (corr_val_x < 0) and (corr_val_y >= 0):
            new_img[corr_val_y:, 0:img.shape[1]+corr_val_x, :] = img[0:img.shape[0]-corr_val_y, -corr_val_x:, :]
        if (corr_val_x < 0) and (corr_val_y < 0):
            new_img[0:img.shape[0]+corr_val_y, 0:img.shape[1]+corr_val_x, :] = img[-corr_val_y:, -corr_val_x:, :]
            
        try:
            if not os.path.isdir(os.path.normpath(oDir)):
                pathlib.Path(os.path.normpath(oDir)).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('creating "%s" directory failed, error "%s"'%(oDir,err))
            exit(1)
        
        cv.imwrite(os.path.normpath(oDir + "/" + img_fn_norm), new_img)
        
        img_fn_2 = img_fn.rsplit('_',1)[0]+"_prob.png"
        
        
        img = cv.imread(img_fn_2, cv.IMREAD_COLOR)
        if not img is None:
            new_img = np.zeros((img.shape[0], img.shape[1], 3))
            
            if (corr_val_x >= 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, corr_val_x:, :] = img[0:img.shape[0]-corr_val_y, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x >= 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, corr_val_x:, :] = img[-corr_val_y:, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x < 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, 0:img.shape[1]+corr_val_x, :] = img[0:img.shape[0]-corr_val_y, -corr_val_x:, :]
            if (corr_val_x < 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, 0:img.shape[1]+corr_val_x, :] = img[-corr_val_y:, -corr_val_x:, :]
            
            cv.imwrite(os.path.normpath(oDir + "/" + os.path.basename(img_fn_2)), new_img)
        
        
        img_fn_3 = img_fn.rsplit('_',1)[0]+"_prob_nl.png"
        
        img = cv.imread(img_fn_3, cv.IMREAD_COLOR)
        if not img is None:
            new_img = np.zeros((img.shape[0], img.shape[1], 3))
            
            if (corr_val_x >= 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, corr_val_x:, :] = img[0:img.shape[0]-corr_val_y, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x >= 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, corr_val_x:, :] = img[-corr_val_y:, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x < 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, 0:img.shape[1]+corr_val_x, :] = img[0:img.shape[0]-corr_val_y, -corr_val_x:, :]
            if (corr_val_x < 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, 0:img.shape[1]+corr_val_x, :] = img[-corr_val_y:, -corr_val_x:, :]
            
            cv.imwrite(os.path.normpath(oDir + "/" + os.path.basename(img_fn_3)), new_img)
            
            
            
def apply_corrections_img_orig(corrections, iDir, oDir):
    
    for img_type in ["lsi", "csi", "nsi", "gsi"]:
        img_list    = glob.glob(os.path.normpath(iDir + "/*_{}.png".format(img_type)))
        img_list.sort()
        for img_fn in img_list:
            img_fn_norm = os.path.basename(os.path.normpath(img_fn))
            img_num = int(img_fn_norm.split('_',1)[0])
            corr_val_x = corrections[img_num-1][1]
            corr_val_y = corrections[img_num-1][2]
            
            img = cv.imread(img_fn, cv.IMREAD_COLOR)
            
            new_img = np.zeros((img.shape[0], img.shape[1], 3))
            
            if (corr_val_x >= 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, corr_val_x:, :] = img[0:img.shape[0]-corr_val_y, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x >= 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, corr_val_x:, :] = img[-corr_val_y:, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x < 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, 0:img.shape[1]+corr_val_x, :] = img[0:img.shape[0]-corr_val_y, -corr_val_x:, :]
            if (corr_val_x < 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, 0:img.shape[1]+corr_val_x, :] = img[-corr_val_y:, -corr_val_x:, :]
                
            try:
                if not os.path.isdir(os.path.normpath(oDir)):
                    pathlib.Path(os.path.normpath(oDir)).mkdir(mode=0o775, parents=True, exist_ok=True)
            except Exception as err:
                logging.error('creating "%s" directory failed, error "%s"'%(oDir,err))
                exit(1)
            
            cv.imwrite(os.path.normpath(oDir + "/" + img_fn_norm), new_img)
        
        
        
def process_dir(iDir, oDir, imgDir, log2, ss, psx, psy, verbose, quit_on_small_margin):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    label_list    = glob.glob(iDir + "/*_roi_labels.png")
    label_list.sort()
    
    if verbose:
        fileOut_fn = oDir + "/" + "extreme_coords.csv"
        fileOut = open(fileOut_fn,"w")
    index = []
    coordinate_mm = []
    number = 1
    results = []
    corrections = []
    
    for label_file in label_list:
        logging.info("opening file: {}".format(label_file))
        image_raw = cv.imread(label_file, cv.IMREAD_COLOR)
        image_raw = image_raw[:,:,2]
        image_raw[image_raw != 255] = 0
        
        minx = 100000000
        miny = 100000000
        maxx = -1
        maxy = -1
        num_found = 0
        
        
        for j in range(0,image_raw.shape[0]):
            for i in range(0,image_raw.shape[1]):
                if image_raw[j,i] == 255:
                    num_found += 1
                    if j>maxy:
                        maxy = j
                    if j<miny:
                        miny = j
                    if i>maxx:
                        maxx = i
                    if i<minx:
                        minx = i
        if num_found>0:
            maxx *= psx
            minx *= psx
            maxy *= psy
            miny *= psy
            if verbose:
                fileOut.write("{},{},{},{}\n".format(minx,maxx,miny,maxy))
            if quit_on_small_margin:
                if (minx<3) or (miny<3) or (maxx>image_raw.shape[1]*psx-3) or (maxy>image_raw.shape[0]*psy-3):
                    logging.error("too small margin detected in ROI")
                    exit(1)
            results.append([minx,maxx,miny,maxy])
            coordinate_mm.append(number*ss)
            index.append(number)
        corrections.append([number,0,0, image_raw.shape[1], image_raw.shape[0]])
        number += 1
    
    pochodnax = [0]
    pochodnay = [0]
    for i in range(1,len(results)):
        pochodnax.append(((results[i][0] + results[i][1])/2-(results[i-1][0] + results[i-1][1])/2)/ss)
        pochodnay.append(((results[i][2] + results[i][3])/2-(results[i-1][2] + results[i-1][3])/2)/ss)
    
    jsonDumpSafe(oDir + "/" +"extremes.json", [[i[0] for i in results], [i[1] for i in results], [i[2] for i in results], [i[3] for i in results], coordinate_mm])
    if verbose:
        plt.plot(coordinate_mm,[i[0] for i in results],'r-', coordinate_mm,[i[1] for i in results],'r+', coordinate_mm,[i[2] for i in results],'g-', coordinate_mm,[i[3] for i in results],'g+')
        plt.savefig(oDir + "/" + "extreme_coords.png")
        plt.clf()
        plt.plot(coordinate_mm,pochodnax,'r^',coordinate_mm,pochodnay,'g^')
        plt.savefig(oDir + "/" + "extreme_coords_deriv.png")
        plt.clf()
    
    a1 = np.array([i[0] for i in results])
    a2 = np.array([i[1] for i in results])
    a3 = np.array([i[2] for i in results])
    a4 = np.array([i[3] for i in results])
    
    w1 = np.fft.fft((a1+a2)/2)
    w2 = np.fft.fft((a3+a4)/2)
    
    w1kopia = w1.copy()
    w2kopia = w2.copy()
    
    #for frequencies larger than 1/12 (?) slices remove all peaks above 2 (?) * average of neighbouring bins
    
    #calcultaing bin number for frequency 1/12
    min_bin_no = int(len(w1kopia)/2 /12 *2)
    logging.info("\nModifying spectrum from bin {}".format(min_bin_no))
    
    w1kopia = process_spectrum(w1kopia, min_bin_no, 2, 3)
    #second pass
    w1kopia = process_spectrum(w1kopia, min_bin_no, 2, 3)
    
    w2kopia = process_spectrum(w2kopia, min_bin_no, 2, 3)
    #second pass
    w2kopia = process_spectrum(w2kopia, min_bin_no, 2, 3)
    
    w1f = np.abs(np.fft.ifft(w1kopia))
    w2f = np.abs(np.fft.ifft(w2kopia))

    if verbose:
        plt.plot(coordinate_mm,[i[0] for i in results],'r-', coordinate_mm,[i[1] for i in results],'r+', coordinate_mm,[i[2] for i in results],'g-', coordinate_mm,[i[3] for i in results],'g+', coordinate_mm,((a1+a2)/2),'r.', coordinate_mm,((a3+a4)/2),'g.', coordinate_mm,w1f,'b-', coordinate_mm,w2f,'b-')
        plt.savefig(oDir + "/" + "extreme_coords_flt.png")
        plt.clf()
    
    w1[0]=0
    w2[0]=0
    w1kopia[0]=0
    w2kopia[0]=0
    w1 = np.abs(w1)
    w2 = np.abs(w2)
    w1kopia = np.abs(w1kopia)
    w2kopia = np.abs(w2kopia)
    if verbose:
        plt.plot(range(0,len(w1)),w1,'r', range(0,len(w2)),w2,'g', range(0,len(w1)), w1kopia, 'r.', range(0,len(w2)), w2kopia, 'g.')
        plt.savefig(oDir + "/" + "fftplot.png")
        plt.clf()
    if verbose:
        fileOut.close()
    
    corrections_x = w1f - ((a1+a2)/2)
    corrections_y = w2f - ((a3+a4)/2)
    
    for i in index:
        #corrections[i-1] = [i, corrections_x[index.index(i)], corrections_y[index.index(i)]]
        corrections[i-1] = [i, int(round(corrections_x[index.index(i)])), int(round(corrections_y[index.index(i)])), corrections[i-1][3], corrections[i-1][4]]
        
    #jsonDumpSafe(oDir + "/" +"corrections.json", [[i for i in corrections_x], [i for i in corrections_y]])
    jsonDumpSafe(oDir + "/" +"corrections.json", corrections)
    if verbose:
        plt.plot([i[0] for i in corrections], [i[1] for i in corrections],'r.', [i[0] for i in corrections],[i[2] for i in corrections],'g.')
        plt.savefig(oDir + "/" + "corrections.png")
        plt.clf()
        
    #perform corrections
    
    apply_corrections(corrections, iDir + "/../bones", oDir + "/bones")
    apply_corrections(corrections, iDir + "/../fat", oDir + "/fat")
    apply_corrections(corrections, iDir + "/../muscles", oDir + "/muscles")
    apply_corrections(corrections, iDir + "/../vessels", oDir + "/vessels")
    
    apply_corrections_img(corrections, iDir + "/../bones", oDir + "/bones")
    apply_corrections_img(corrections, iDir + "/../fat", oDir + "/fat")
    apply_corrections_img(corrections, iDir + "/../muscles", oDir + "/muscles")
    apply_corrections_img(corrections, iDir + "/../roi", oDir + "/roi")
    apply_corrections_img(corrections, iDir + "/../vessels", oDir + "/vessels")
    
    
    apply_corrections_img_orig(corrections, imgDir, oDir + "/images")
    
    #difficult one - shift the skin with corrected thickness
    
    apply_corrections(corrections, oDir + "/skin_closed", oDir + "/skin")
    apply_corrections_img(corrections, oDir + "/skin_closed", oDir + "/skin")
    
    
    
        
        
        

parser = ArgumentParser()

parser.add_argument("-iDir",      "--input_dir"      ,     dest="idir"   ,    help="input directory" ,    metavar="PATH", required=True)
parser.add_argument("-imgDir",      "--image_dir"      ,     dest="imgdir"   ,    help="image input directory" ,    metavar="PATH", required=True)
parser.add_argument("-oDir",      "--output_dir"     ,     dest="odir"   ,    help="output directory",    metavar="PATH", required=True)
parser.add_argument("-ss"  ,      "--slice_spacing"  ,     dest="ss"     ,    help="slice spacing"   ,    metavar="PATH", required=True)
parser.add_argument("-psx"  ,     "--pixel_spacing_x",     dest="psx"    ,    help="pixel spacing x" ,    metavar="PATH", required=True)
parser.add_argument("-psy"  ,     "--pixel_spacing_y",     dest="psy"    ,    help="pixel spacing y" ,    metavar="PATH", required=True)

parser.add_argument("-v"   ,      "--verbose"        ,     dest="verbose",    help="verbose level"   ,                    required=False)
parser.add_argument("-q"   ,      "--quit"        ,     dest="quit_on_small_margin",    help="quit_on_small_margin - 'on' to turn on"   ,                    required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
quit_on_small_margin = True                 if args.quit_on_small_margin is None else (args.quit_on_small_margin == 'on')
iDir 	= args.idir
oDir  	= args.odir
ss      = args.ss
psx     = args.psx
psy     = args.psy
imgDir     = args.imgdir

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(oDir+"/correctShapesSkin.log",mode='w'),logging.StreamHandler(sys.stdout)])


if not os.path.isdir(iDir):
    logging.error('Error : Input directory (%s) not found !',iDir)
    exit(1)


logging.info("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_correctPosition.py")
logging.info("in:  "    +   iDir    )
logging.info("image in:  "    +   imgDir    )
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

process_dir(iDir, oDir, imgDir, log2, float(ss), float(psx), float(psy), verbose, quit_on_small_margin)

log2.close()