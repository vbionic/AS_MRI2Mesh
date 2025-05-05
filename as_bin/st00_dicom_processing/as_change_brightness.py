import sys, getopt
import pydicom 
import numpy as np
import json
import os
import pathlib
import cv2
import glob
import tracemalloc
import multiprocessing
#-----------------------------------------------------------------------------------------
from scipy import ndimage, misc
from skimage import data, img_as_float
from skimage import exposure
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
#-----------------------------------------------------------------------------------------
from argparse import ArgumentParser
import logging
#-----------------------------------------------------------------------------------------
from scipy.interpolate import (BSpline, BPoly, PPoly, make_interp_spline,

        make_lsq_spline, _bspl, splev, splrep, splprep, splder, splantider,

         sproot, splint, insert)
#-----------------------------------------------------------------------------------------

def process_img(img_path,lut,verbose):
    
    logging.info(img_path)

    img 	= np.load(img_path)

    return(imgs)

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    mask = np.uint8(mask)
    #print(mask)
    return mask

def create_lut(node_x,node_y, lut_x):
   bspl 	= make_interp_spline(node_x,node_y,k=2)
   y     	= bspl(lut_x)
   return(lut_x,y)

#-----------------------------------------------------------------------------------------

parser = ArgumentParser()

parser.add_argument("-imgDir",  "--img_dir",     dest="img_dir",  help="input img directory",         metavar="PATH",required=True)
parser.add_argument("-outDir",  "--out_dir",     dest="out_dir",  help="output directory",           metavar="PATH",required=True)
parser.add_argument("-v",       "--verbose",     dest="verbose", help="verbose level",                              required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
imgDir  	= args.img_dir
outDir  	= args.out_dir

imgDir = os.path.normpath(imgDir)
outDir = os.path.normpath(outDir)

try:
	if not os.path.isdir(outDir):
	    pathlib.Path(outDir).mkdir(mode=0o775, parents=True, exist_ok=True)
except Exception as err:
	print("Output dir (%s) IO error: {}"%(outDir,err))
	sys.exit(1)

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(outDir+"/change_brihtness.log",mode='w'),logging.StreamHandler(sys.stdout)])    
    
if not os.path.isdir(imgDir):
    logging.error('Error : Input directory (%s) with PNG files not found !'%imgDir)
    exit(1)
logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_change_brightness.py")
logging.info("in: "+imgDir)
logging.info("out: "+outDir)


np.set_printoptions(linewidth=128)

tc          = multiprocessing.cpu_count()

logging.info('-----------------------------------------------------')
logging.info('Detected threads      :%d '% tc)

at          = int((3*tc)/4)
cv2.setNumThreads(at)

logging.info('CV2 threads           :%d ( min(%d, 8))'%(at,tc))
logging.info('-----------------------------------------------------')
logging.info('Verbose level         :%s '% verbose)
logging.info('-----------------------------------------------------')

gname       = imgDir + '/*_fsi.png'
gname       = os.path.normpath(gname)

images      = glob.glob(gname)
imid        = 0

if len(images)==0:
    logging.error('invalid file name or path (%s)'%gname)
    exit(1)

images.sort()

sd_path	= outDir + '/set_data.json'
sd_path	= os.path.normpath(sd_path)

try:
    sdx 	= open(sd_path,"r");     
except:
    exit(1)

#----------------------------------------------------
# json
#----------------------------------------------------

jd	= json.load(sdx); 

pixel_spacing = jd['pixel_spacing_x']   

logging.info('pixel spacing : %0.2f'%pixel_spacing)

img_list	= []

status_path	= os.path.normpath(outDir+'/'+'status.json')
status_data = {}

images.sort()

points = []

logging.info('FIRST LOOP  ---- finding of the maximum pixel value')

maxpixv0    = 0
maxid0      = 0

maxpixvM    = 0
maxidM      = 0

imid        = 0
kernel 		= create_circular_mask(11,11)

for iname in images:

    xname           = os.path.basename(iname)
    fname, fext     = os.path.splitext(xname)
    fname, fsuf     = fname.split('_')

    #if imid!= 0:
        #logging.info('NEXT      > -----------------------------------------------------')

    npy_path 	= os.path.normpath(imgDir+'/'+fname+'_fsi.png')
    im16  	= cv2.imread(npy_path, cv2.IMREAD_ANYDEPTH)
    locmax0     = im16.max()

    im16 	= cv2.morphologyEx(im16, cv2.MORPH_OPEN, kernel)
    locmaxM     = im16.max()
    
    logging.info('file name %s, local maximum pixel value (original,filtered): %d,%d'%(fname,locmax0,locmaxM))
    
    if maxpixv0 < locmax0:
        maxpixv0 = locmax0
        maxid0   = imid

    if maxpixvM < locmaxM:
        maxpixvM = locmaxM
        maxidM   = imid

    imid += 1
  
logging.info('global maximum pixel value (org): %s'%maxpixv0)
logging.info('global maximum pixel value (fil): %s'%maxpixvM)


if maxpixv0<64:
    maxpixvM = maxpixv0
 
imid        = 0
for iname in images:

    xname           = os.path.basename(iname)
    fname, fext     = os.path.splitext(xname)
    fname, fsuf     = fname.split('_')

#    if imid!= 0:
#        logging.info('NEXT      > -----------------------------------------------------')

    npy_path 	= os.path.normpath(imgDir+'/'+fname+'_fsi.png')
    im16  	= cv2.imread(npy_path, cv2.IMREAD_ANYDEPTH)
    im16        = im16.astype(int)

    hist,bins = np.histogram(im16,maxpixv0,[0,maxpixv0-1])
    z = 0

    logging.info('file name %s'%(fname))
    imid += 1

mx 	= (maxpixvM+1)*0.40
ex 	= (maxpixvM+1)

my  = 255*0.65
ey	= 255

node_x = [8, mx ,ex]
node_y = [0, my ,ey]
lut_b  = np.arange(0, maxpixv0+2)
lut_r  = np.arange(0, maxpixv0+2)

lut_x,lut_y = create_lut(node_x,node_y,lut_b[8:maxpixvM+1])

for n in range(0,8):
    lut_r[n] = 0
for n in range(8,maxpixvM+1):
    lut_r[n] = lut_y[n-8]
for n in range(maxpixvM+1,maxpixv0+2):
    lut_r[n] = 255

for i in range(0,len(lut_r)):
    if lut_r[i]>255 : lut_r[i] = 255
    
for iname in images:

    #if imid!= 0:
    #    logging.info('NEXT      > -----------------------------------------------------')

    xname           = os.path.basename(iname)
    fname, fext     = os.path.splitext(xname)
    fname, fsuf     = fname.split('_')
    imid += 1

    npy_path 		= os.path.normpath(imgDir+'/'+fname + '_fsi.png' )
    out_path 		= os.path.normpath(outDir+'/'+fname + '_nsi.png')
    cla_path 		= os.path.normpath(outDir+'/'+fname + '_csi.png')

    img 		= cv2.imread(npy_path, cv2.IMREAD_ANYDEPTH)


    lut_int 	    = lut_r.astype(int)
    img16_1D        = np.reshape(img.copy(),img.shape[0]*img.shape[1]).astype(int)
    img16_1D        = lut_int[img16_1D]
    img8            = np.reshape(img16_1D,(img.shape[0],img.shape[1]) )
    img8int         = img8.astype(np.uint8)

    cv2.imwrite(out_path,img8int)

    fsize 	    = int(0.4 + 3/pixel_spacing)
    clahe 	    = cv2.createCLAHE(clipLimit=(6.0), tileGridSize=(fsize,fsize))
    cla             = clahe.apply(img8int)
    
    cv2.imwrite(cla_path,cla.astype(np.uint8))

print(maxpixv0,maxpixvM)

#-----------------------------------------------------------------------------------------
