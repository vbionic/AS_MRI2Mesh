import sys, getopt
import pydicom
import numpy as np
import json 
import os
import pathlib
import cv2 as cv
import glob
import tracemalloc
import multiprocessing
from scipy import ndimage, misc
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
#-----------------------------------------------------------------------------------------
from PIL import Image
from pydicom.tag import Tag
from pydicom.datadict import keyword_for_tag
from argparse import ArgumentParser
import logging
#-----------------------------------------------------------------------------------------

def process_file(img_path, SPSize, SPRatio, SPConn, verbose):

    im_rd       = cv.imread(img_path)
    print(img_path)
    superpix    = cv.ximgproc.createSuperpixelLSC(im_rd,SPSize,SPRatio)
    superpix.iterate(20)
    superpix.enforceLabelConnectivity(SPConn)
    mask = superpix.getLabelContourMask()
    labels = superpix.getLabels()
    number = superpix.getNumberOfSuperpixels()
    
    avgs = im_rd.copy()
    for n in range(0,number):
        suma = np.sum(im_rd[labels==n])
        temp = im_rd.copy()
        temp[labels==n]=1
        temp[labels!=n]=0
        licznosc = np.sum(temp)
        if licznosc != 0:
            avgs[labels==n] = int(suma/licznosc)
        else:
            avgs[labels==n] = 0
    
    
    return [mask, labels, avgs]

#-----------------------------------------------------------------------------------------

parser = ArgumentParser()

parser.add_argument("-dpDir",   "--dp_dir",         dest="dp_dir",      help="input png directory",         metavar="PATH",required=True)
parser.add_argument("-spxDir",  "--superpix_dir",   dest="spx_dir",     help="Output superpixel directory", metavar="PATH",required=True)
parser.add_argument("-v",       "--verbose",        dest="verbose",     help="verbose level",                              required=False)

parser.add_argument("-ns",      "--name_sufix",     dest="ns",          help="name sufix (nsi,lsi,gsi, ...)",              required=False)

parser.add_argument("-ps",      "--psize",          dest="spxsize",     help="superpixel size (LSC)",                      required=False)
parser.add_argument("-pr",      "--pratio",         dest="spxratio",    help="superpixel ratio value (LSC)",               required=False)
parser.add_argument("-pc",      "--pconn",          dest="spxconn",     help="superpixel connectivity (LSC)",              required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose

dpDir   	= args.dp_dir
spxDir   	= args.spx_dir
spxsize 	= int(args.spxsize)
spxratio 	= int(args.spxratio)
spxconn 	= int(args.spxconn)
ns      	= 'nsi' if args.ns is None else args.ns

dpDir = os.path.normpath(dpDir)
spxDir = os.path.normpath(spxDir)


try:
    if not os.path.isdir(spxDir):
        pathlib.Path(spxDir).mkdir(mode=0o775, parents=True, exist_ok=True)
except Exception as err:
    print('INFO      > creating "%s" directory faild, error "%s"'%(spxDir,err))
    sys.exit(1)

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(spxDir+"/superpixelize.log",mode='w'),logging.StreamHandler(sys.stdout)])

if not os.path.isdir(dpDir):
    logging.error('Error : Input directory ({}) with PNG files not found !'.format(dpDir))
    exit(1)

tc          = multiprocessing.cpu_count()


logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_superpixelLSC.py")
logging.info("in: "+dpDir)
logging.info("out: "+spxDir)

logging.info('> Superpixel calculation ------------------------------')
logging.info('> Detected threads      :%d '% tc)

at          = int((3*tc)/4)
cv.setNumThreads(at)

logging.info('> cv2 threads           :%d ( min(%d, 8))'%(at,tc))
logging.info('> -----------------------------------------------------')
logging.info('> Verbose level         :%s '% verbose)
logging.info('> -----------------------------------------------------')

gname       = dpDir + '/*_' + ns + '.png'
gname       = os.path.normpath(gname)

images      = glob.glob(gname)
imid        = 0

if images == []:
    logging.error('> invalid file name or path (%s)'%gname)

images.sort()

points = []

geometry = []

logging.info('LOOP  ---- calculating superpixels')

imid        = 0

for iname in images:

    xname           = os.path.basename(iname)
    fname, fext     = os.path.splitext(xname)
    print(fname)
    fname, fsuf     = fname.rsplit('_',1)

#    if imid!= 0:
#        logging.info('NEXT      > -----------------------------------------------------')

    logging.info('> file name     : {}'.format(fname))
    
    if '_' in ns:
        ns = ns.rsplit('_',1)[1]

    png_path    = os.path.normpath(dpDir+'/'+fname+'_'+ns+'.png')
   
    SPSize 		= spxsize
    SPRatio  	= spxratio
    SPConn  	= spxconn

    logging.info('> SuperPixelSize    : {}'.format(SPSize))
    logging.info('> SuperPixelRatio     : {}'.format(SPRatio))
    logging.info('> SuperPixelConn     : {}'.format(SPConn))

    [superpixelized_lsi, labels_lsi, avg_lsi] = process_file(png_path, SPSize, SPRatio/100, SPConn, verbose=="on")

    imid += 1
    img_path            = os.path.normpath(spxDir + '/' + fname + '_'+ns+ '_{:02d}{:02d}{:02d}'.format(SPSize,SPRatio,SPConn) +'_superpixel.png')
    #img_pathlabels = os.path.normpath(spxDir + '/' + fname + '_'+ns+'_superpixel_labels.png')
    #img_pathlabelsRGB = os.path.normpath(spxDir + '/' + fname + '_'+ns+'_superpixel_labelsRGB.png')
    img_pathlabelsRGB   = os.path.normpath(spxDir + '/' + fname + '_'+ns+ '_{:02d}{:02d}{:02d}'.format(SPSize,SPRatio,SPConn) +'_superpixel_labels.png')
    avg_pathlabels      = os.path.normpath(spxDir + '/' + fname + '_'+ns+ '_{:02d}{:02d}{:02d}'.format(SPSize,SPRatio,SPConn) +'_superpixel_avg.png')
    superpixelized_lsi = cv.cvtColor(superpixelized_lsi,cv.COLOR_GRAY2BGRA)
    superpixelized_lsi[:,:,3] = superpixelized_lsi[:,:,0]
    #cv.imwrite(img_path,superpixelized_lsi)
    cv.imwrite(avg_pathlabels,avg_lsi)
    labels_RGB = np.zeros((labels_lsi.shape[0], labels_lsi.shape[1], 3))
    print(labels_RGB.shape)
    labels_RGB[:,:,0] = (labels_lsi // 256).astype(np.uint8)
    labels_RGB[:,:,1] = (labels_lsi % 256).astype(np.uint8)
    #cv.imwrite(img_pathlabels,labels_lsi.astype(np.uint16))
    cv.imwrite(img_pathlabelsRGB,labels_RGB.astype(np.uint8))

#-----------------------------------------------------------------------------------------
