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

def process_file(img_path, alg, SPSize, SPRuler, SPConn, verbose):

    im_rd       = cv.imread(img_path)
    #print(img_path)
    superpix    = cv.ximgproc.createSuperpixelSLIC( im_rd,
                                                    algorithm   = alg,
                                                    region_size = SPSize,
                                                    ruler       = SPRuler
                                                    )
                                                    
    superpix.iterate(20)
    superpix.enforceLabelConnectivity(SPConn)
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
    
    
    return [labels, avgs]


#-----------------------------------------------------------------------------------------

parser = ArgumentParser()

parser.add_argument("-iDir",   "--input_dir",       dest="in_dir",      help="input png directory",         metavar="PATH",required=True)
parser.add_argument("-oDir",   "--output_dir",      dest="out_dir",     help="Output superpixel directory", metavar="PATH",required=True)
parser.add_argument("-v",       "--verbose",        dest="verbose",     help="verbose level",                              required=False)

parser.add_argument("-ns",      "--name_sufix",     dest="ns",          help="name sufix (nsi,lsi,gsi, ...)",              required=False)

parser.add_argument("-ps",      "--psize",          dest="spxsize",     help="superpixel size (LSC)",                      required=False)
parser.add_argument("-pr",      "--pratio",         dest="spxratio",    help="superpixel ratio value (LSC)",               required=False)
parser.add_argument("-pc",      "--pconn",          dest="spxconn",     help="superpixel connectivity (LSC)",              required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose

dpDir       = args.in_dir
spxDir      = args.out_dir
spxsize     = int(args.spxsize)
spxratio    = int(args.spxratio)
spxconn     = int(args.spxconn)
ns          = 'nsi' if args.ns is None else args.ns

dpDir       = os.path.normpath(dpDir)
spxDir      = os.path.normpath(spxDir)

try:
    if not os.path.isdir(spxDir):
        pathlib.Path(spxDir).mkdir(mode=0o775, parents=True, exist_ok=True)
except Exception as err:
    logging.error('Creating "%s" directory faild, error "%s"'%(spxDir,err))
    sys.exit(1)

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(spxDir+"/superpixelize.log",mode='w'),logging.StreamHandler(sys.stdout)])

if not os.path.isdir(dpDir):
    logging.error('Error : Input directory ({}) with PNG files not found !'.format(dpDir))
    exit(1)

tc          = multiprocessing.cpu_count()
logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_superpixelSLIC.py")
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

imagesX     = glob.glob(gname)
images      = []

for iname in imagesX:
    #print(iname)
    if iname.find('rect')==-1:
        images.append(iname)      
    
#print(images)
    
imid        = 0

if images == []:
    logging.error('> invalid file name or path (%s)'%gname)

images.sort()

points = []

geometry = []

logging.info('LOOP  ---- calculating superpixels')

imid        = 0
cset        = {}

for iname in images:

    xname           = os.path.basename(iname)
    fname, fext     = os.path.splitext(xname)
    fname, fsuf     = fname.rsplit('_',1)

#    if imid!= 0:
#        logging.info('NEXT      > -----------------------------------------------------')

    logging.info('> file name     : {}'.format(fname))
    
    if '_' in ns:
        ns = ns.rsplit('_',1)[1]

    png_path        = os.path.normpath(dpDir+'/'+fname+'_'+ns+'.png')
    
    SPSize          = spxsize
    SPRatio         = spxratio
    SPConn          = 12
    
    logging.info('> SuperPixelSize    : {}'.format(SPSize))
    logging.info('> SuperPixelRatio   : {}'.format(SPRatio))
    logging.info('> SuperPixelConn    : {}'.format(SPConn))
    
    # ---------------
    [labels,  avg]  = process_file(png_path, 100, SPSize, SPRatio, SPConn, verbose=="on")
    # ---------------

    imid += 1
    img_pathlabels      = os.path.normpath(spxDir + '/' + fname + '_'+ns+ '_{:02d}{:02d}'.format(int(SPSize),SPRatio) +'_slic_labels.png')
    avg_pathlabels      = os.path.normpath(spxDir + '/' + fname + '_'+ns+ '_{:02d}{:02d}'.format(int(SPSize),SPRatio) +'_slic_avg.png')

    labels_RGB = np.zeros((labels.shape[0], labels.shape[1], 3))

    logging.info("writing : %s"%img_pathlabels)
    logging.info("writing : %s"%avg_pathlabels)

    labels_RGB[:,:,0] = (labels // 256).astype(np.uint8)
    labels_RGB[:,:,1] = (labels % 256).astype(np.uint8)

    cv.imwrite(avg_pathlabels,avg)
    cv.imwrite(img_pathlabels,labels_RGB.astype(np.uint8))

#-----------------------------------------------------------------------------------------
